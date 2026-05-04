#include "../ops.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/dispatch/dispatch_radix_sort.cuh>

#include <cuda_runtime.h>

#include <stdexcept>

namespace polysplat {
namespace {

static uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// ---------------------------------------------------------------------------
// Pinned host slot + busy-wait event for async curr_offset readback.
//
// Synchronous cudaMemcpy(D2H) implies (a) an internal cudaStreamSynchronize
// and (b) a pageable-memory staging copy.  Using cudaMemcpyAsync into pinned
// host memory plus a non-blocking event saves ~tens of us of host overhead
// between preprocess-end and the first CUB sort kernel.
// ---------------------------------------------------------------------------
struct PinnedNumRenderedSlot
{
	int* host_ptr = nullptr;
	cudaEvent_t event = nullptr;

	PinnedNumRenderedSlot()
	{
		if (cudaHostAlloc(reinterpret_cast<void**>(&host_ptr), sizeof(int),
		                  cudaHostAllocDefault) != cudaSuccess) {
			host_ptr = nullptr;
			return;
		}
		*host_ptr = 0;
		// cudaEventDisableTiming + (no BlockingSync) -> spin-wait in user space.
		if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
			event = nullptr;
		}
	}

	~PinnedNumRenderedSlot()
	{
		if (event) cudaEventDestroy(event);
		if (host_ptr) cudaFreeHost(host_ptr);
	}
};

PinnedNumRenderedSlot& getPinnedSlot()
{
	thread_local PinnedNumRenderedSlot slot;
	return slot;
}

// ---------------------------------------------------------------------------
// Pre-bound CUB radix-sort dispatch for SM90.
//
// cub::DeviceRadixSort::SortPairs() -> DispatchRadixSort::Dispatch() does
//   (1) PtxVersion(...)               // per-device cached query
//   (2) MaxPolicyT::Invoke(ptx_version, dispatch)  // chained policy walk
//       -> dispatch.template Invoke<Policy900>()   // SM90 path
// Skip (1) and (2) by constructing the dispatch struct directly with Policy900.
// ---------------------------------------------------------------------------
template <typename KeyT, typename ValueT, typename OffsetT>
static cudaError_t dispatchRadixSortSm90(
	void* d_temp_storage, size_t& temp_storage_bytes,
	const KeyT* d_keys_in, KeyT* d_keys_out,
	const ValueT* d_values_in, ValueT* d_values_out,
	OffsetT num_items, int begin_bit, int end_bit,
	cudaStream_t stream)
{
	using DispatchT = cub::DispatchRadixSort<false, KeyT, ValueT, OffsetT>;
	using SelectedPolicy = typename DispatchT::MaxPolicy;  // = Policy900

	cub::DoubleBuffer<KeyT>   d_keys(const_cast<KeyT*>(d_keys_in), d_keys_out);
	cub::DoubleBuffer<ValueT> d_values(const_cast<ValueT*>(d_values_in), d_values_out);

	DispatchT dispatch(
		d_temp_storage, temp_storage_bytes,
		d_keys, d_values,
		num_items, begin_bit, end_bit,
		/*is_overwrite_okay=*/false,
		stream,
		/*ptx_version=*/900);

	return dispatch.template Invoke<SelectedPolicy>();
}

} // namespace

void sort_gaussian(int num_rendered,
    int width, int height, int block_x, int block_y,
	char* list_sorting_space, size_t sorting_size,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted, cudaStream_t stream)
{
	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);
	const int end_bit = 32 + (int)getHigherMsb(grid.x * grid.y);
	size_t temp_storage_bytes = sorting_size;
	auto status = dispatchRadixSortSm90<uint64_t, uint32_t, int>(
		list_sorting_space, temp_storage_bytes,
		gaussian_keys_unsorted, gaussian_keys_sorted,
		gaussian_values_unsorted, gaussian_values_sorted,
		num_rendered, 0, end_bit, stream);
	if (status != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(status));
	}
}

size_t get_sort_buffer_size(int num_rendered, cudaStream_t stream)
{
    size_t sort_buffer_size = 0;
	cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>(
		nullptr, sort_buffer_size,
		nullptr, nullptr,
		nullptr, nullptr, num_rendered, 0, sizeof(uint64_t) * 8, stream);
	// Account for ES paths: uint32 keys (depth or tile), uint32 values.
	size_t es_size = 0;
	cub::DeviceRadixSort::SortPairs<uint32_t, uint32_t>(
		nullptr, es_size,
		nullptr, nullptr,
		nullptr, nullptr, num_rendered, 0, sizeof(uint32_t) * 8, stream);
	if (es_size > sort_buffer_size) sort_buffer_size = es_size;
	// Also float key depth sort (using __float_as_uint reinterpretation):
	size_t es_depth_size = 0;
	cub::DeviceRadixSort::SortPairs<float, uint32_t>(
		nullptr, es_depth_size,
		nullptr, nullptr,
		nullptr, nullptr, num_rendered, 0, sizeof(float) * 8, stream);
	if (es_depth_size > sort_buffer_size) sort_buffer_size = es_depth_size;
    return sort_buffer_size;
}

// =====================================================================
// ES orchestration: depth sort + gather + scan + Pass-B emit + tile sort.
// All intermediate buffers are passed in by caller (pre-allocated).
// =====================================================================
namespace {

// Small utility kernels for ES pipeline.
__global__ void fill_identity_kernel(int P, uint32_t* __restrict__ identity)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < P) identity[i] = (uint32_t)i;
}

__global__ void gather_counts_kernel(int P,
	const uint32_t* __restrict__ perm,
	const int32_t* __restrict__ counts_nat,
	int32_t* __restrict__ counts_sorted)
{
	int s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s < P) counts_sorted[s] = counts_nat[perm[s]];
}

// Rebuild 64-bit sorted keys from 32-bit tile keys (upper 32 bits = tile_id,
// lower 32 bits = 0 — depth not needed; render reads depth from rgb_depth[gid].w).
__global__ void rebuild_tile_keys_kernel(int M,
	const uint32_t* __restrict__ tile32,
	uint64_t* __restrict__ keys64)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < M) keys64[i] = ((uint64_t)tile32[i]) << 32;
}

} // anonymous namespace

size_t get_es_scan_buffer_size(int P, cudaStream_t stream)
{
	size_t bytes = 0;
	// int32 exclusive sum
	cub::DeviceScan::ExclusiveSum(nullptr, bytes, (int32_t*)nullptr, (int32_t*)nullptr, P, stream);
	return bytes;
}

// Small helper: write num_rendered = cum[P-1] + counts[P-1] in one thread.
__global__ static void compute_total_kernel(int P,
	const int32_t* __restrict__ cum,
	const int32_t* __restrict__ counts,
	int32_t* __restrict__ out)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*out = (P > 0) ? (cum[P - 1] + counts[P - 1]) : 0;
	}
}

// Publicly expose: depth sort + gather + scan + compute_total.
// Returns num_rendered via `total_num_rendered_out` (device int scalar).
void es_depth_sort_and_scan(
	int P,
	const int32_t* tiles_per_gauss,
	const float* depth_natural,
	uint32_t* identity_buf,
	uint32_t* perm_buf,
	float* depth_sorted_buf,
	int32_t* tiles_per_gauss_sorted_buf,
	int32_t* cum_offsets_sorted_buf,
	int32_t* total_num_rendered_out,
	char* cub_sort_scratch, size_t cub_sort_scratch_bytes,
	char* cub_scan_scratch, size_t cub_scan_scratch_bytes,
	cudaStream_t stream)
{
	if (P <= 0) { return; }

	// 1. identity
	{ int BS = 256; int nb = (P + BS - 1) / BS;
	  fill_identity_kernel<<<nb, BS, 0, stream>>>(P, identity_buf); }

	// 2. depth sort (float key + uint32 value)
	{
		size_t tmp_bytes = cub_sort_scratch_bytes;
		cudaError_t status = cub::DeviceRadixSort::SortPairs<float, uint32_t>(
			cub_sort_scratch, tmp_bytes,
			depth_natural, depth_sorted_buf,
			identity_buf, perm_buf,
			P, 0, 32, stream);
		if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
	}

	// 3. gather tiles_per_gauss by perm
	{ int BS = 256; int nb = (P + BS - 1) / BS;
	  gather_counts_kernel<<<nb, BS, 0, stream>>>(P, perm_buf, tiles_per_gauss, tiles_per_gauss_sorted_buf); }

	// 4. exclusive scan
	{
		size_t tmp_bytes = cub_scan_scratch_bytes;
		cudaError_t status = cub::DeviceScan::ExclusiveSum(
			cub_scan_scratch, tmp_bytes,
			tiles_per_gauss_sorted_buf, cum_offsets_sorted_buf,
			P, stream);
		if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
	}

	// 5. compute total = cum[P-1] + counts[P-1]
	compute_total_kernel<<<1, 32, 0, stream>>>(P, cum_offsets_sorted_buf, tiles_per_gauss_sorted_buf, total_num_rendered_out);
}

// 32-bit stable tile sort + rebuild 64-bit sorted keys for render consumption.
void es_tile_sort(
	int num_rendered,
	int width, int height, int block_x, int block_y,
	uint32_t* tile_keys_unsorted,
	uint32_t* gauss_values_unsorted,
	uint32_t* tile_keys_sorted,
	uint32_t* gauss_values_sorted,
	uint64_t* gaussian_keys_sorted_out,  // rebuilt 64-bit keys (tile << 32)
	char* cub_sort_scratch, size_t cub_sort_scratch_bytes,
	cudaStream_t stream)
{
	if (num_rendered <= 0) return;
	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);
	const int tile_end_bit = (int)getHigherMsb(grid.x * grid.y);

	size_t tmp_bytes = cub_sort_scratch_bytes;
	cudaError_t status = cub::DeviceRadixSort::SortPairs<uint32_t, uint32_t>(
		cub_sort_scratch, tmp_bytes,
		tile_keys_unsorted, tile_keys_sorted,
		gauss_values_unsorted, gauss_values_sorted,
		num_rendered, 0, tile_end_bit, stream);
	if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));

	// Rebuild 64-bit keys for render kernel's find_tile_range (upper 32 bits = tile_id).
	{
		int BS = 256;
		int nb = (num_rendered + BS - 1) / BS;
		rebuild_tile_keys_kernel<<<nb, BS, 0, stream>>>(num_rendered, tile_keys_sorted, gaussian_keys_sorted_out);
	}
}

// Async readback of curr_offset via pinned host memory + busy-wait event.
// Replaces the synchronous cudaMemcpy(D2H) used by callers in pybind.cpp.
void fetch_num_rendered_async(const int* d_curr_offset, int* out_num_rendered, cudaStream_t stream)
{
	auto& slot = getPinnedSlot();
	if (slot.host_ptr == nullptr || slot.event == nullptr) {
		cudaMemcpy(out_num_rendered, d_curr_offset, sizeof(int), cudaMemcpyDeviceToHost);
		return;
	}
	cudaMemcpyAsync(slot.host_ptr, d_curr_offset, sizeof(int), cudaMemcpyDeviceToHost, stream);
	cudaEventRecord(slot.event, stream);
	cudaEventSynchronize(slot.event);  // user-space spin (event has no BlockingSync)
	*out_num_rendered = *slot.host_ptr;
}

} // namespace polysplat
