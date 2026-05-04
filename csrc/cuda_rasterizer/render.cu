#include "../ops.h"
#include <cstdio>
#include <cub/device/device_radix_sort.cuh>

namespace polysplat {
namespace {

// Binary-search the sorted key array to find [start, end) for a given tile.
// Replaces the former identifyTileRanges kernel + ranges buffer.
__forceinline__ __device__ int2 find_tile_range(
	const uint64_t* __restrict__ sorted_keys, int num_rendered, uint32_t tile_id)
{
	// Lower bound: first index where key>>32 >= tile_id
	int lo = 0, hi = num_rendered;
	while (lo < hi)
	{
		int mid = (lo + hi) >> 1;
		if ((__ldg(&sorted_keys[mid]) >> 32) < tile_id)
			lo = mid + 1;
		else
			hi = mid;
	}
	int start = lo;
	// Upper bound: first index where key>>32 > tile_id
	hi = num_rendered;
	while (lo < hi)
	{
		int mid = (lo + hi) >> 1;
		if ((__ldg(&sorted_keys[mid]) >> 32) <= tile_id)
			lo = mid + 1;
		else
			hi = mid;
	}
	return make_int2(start, lo);
}

// Warp-cooperative 32-way partitioned search.
// All 32 lanes participate: each iteration narrows the search range by ~32x.
//
// Algorithm (lower-bound = first idx with key_hi >= tile_id):
//   stride = ceil((hi - lo) / 32)
//   lane l samples pos_l = lo + l * stride (guard pos_l < hi).
//   below_l = (pos_l < hi) && key(pos_l) < tile_id
//   Array is sorted => below_l is monotone decreasing in l.
//   mask = __ballot_sync(below) has pattern 1...10...0.
//   k = popc(mask) = first lane where below==false.
//     - k == 0: lane 0's key already >= target => lower bound == lo.
//     - k >= 1: transition between lane (k-1) and lane k.
//               lower bound is in (lo + (k-1)*stride, lo + k*stride].
//               Narrow to new_lo = lo + (k-1)*stride + 1, new_hi = lo + k*stride + 1.
// Converges in <= ceil(log_32(N)) iterations.
__forceinline__ __device__ int2 find_tile_range_warp(
	const uint64_t* __restrict__ sorted_keys, int num_rendered, uint32_t tile_id,
	unsigned lane)
{
	// ---- Lower bound ----
	int lo = 0, hi = num_rendered;
	while (hi - lo > 32)
	{
		int span = hi - lo;
		int stride = (span + 31) >> 5;  // >= 1
		int pos = lo + (int)lane * stride;
		bool below = (pos < hi) &&
			((uint32_t)(__ldg(&sorted_keys[pos]) >> 32) < tile_id);
		unsigned mask = __ballot_sync(~0u, below);
		int k = __popc(mask);
		int new_lo, new_hi;
		if (k == 0)
		{
			// lane 0 already >= target => lower bound = lo, break out.
			break;
		}
		else
		{
			new_lo = lo + (k - 1) * stride + 1;
			new_hi = lo + k * stride + 1;
		}
		if (new_hi > hi) new_hi = hi;
		if (new_lo > hi) new_lo = hi;
		lo = new_lo;
		hi = new_hi;
	}
	// Final narrow (<= 32 elements): straightforward ballot of `below`
	{
		int pos = lo + (int)lane;
		bool below = (pos < hi) &&
			((uint32_t)(__ldg(&sorted_keys[pos]) >> 32) < tile_id);
		unsigned mask = __ballot_sync(~0u, below);
		lo = lo + __popc(mask);
	}
	int start = lo;

	// ---- Upper bound: first idx where key_hi > tile_id ----
	hi = num_rendered;
	while (hi - lo > 32)
	{
		int span = hi - lo;
		int stride = (span + 31) >> 5;
		int pos = lo + (int)lane * stride;
		bool le = (pos < hi) &&
			((uint32_t)(__ldg(&sorted_keys[pos]) >> 32) <= tile_id);
		unsigned mask = __ballot_sync(~0u, le);
		int k = __popc(mask);
		int new_lo, new_hi;
		if (k == 0)
		{
			break;
		}
		else
		{
			new_lo = lo + (k - 1) * stride + 1;
			new_hi = lo + k * stride + 1;
		}
		if (new_hi > hi) new_hi = hi;
		if (new_lo > hi) new_lo = hi;
		lo = new_lo;
		hi = new_hi;
	}
	{
		int pos = lo + (int)lane;
		bool le = (pos < hi) &&
			((uint32_t)(__ldg(&sorted_keys[pos]) >> 32) <= tile_id);
		unsigned mask = __ballot_sync(~0u, le);
		lo = lo + __popc(mask);
	}
	return make_int2(start, lo);
}

__forceinline__ __device__ float fast_ex2_ftz_f32(float x)
{
	float y;
	asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}


__forceinline__ __host__ __device__ int align_up_pow2(int value, int alignment)
{
	return (value + alignment - 1) & ~(alignment - 1);
}

__forceinline__ __device__ void cp_async_ca_shared_global_8(void* smem_dst, const void* gmem_src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
	asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(smem_addr), "l"(gmem_src));
#else
	*reinterpret_cast<float2*>(smem_dst) = *reinterpret_cast<const float2*>(gmem_src);
#endif
}

__forceinline__ __device__ void cp_async_ca_shared_global_16(void* smem_dst, const void* gmem_src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
	asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_src));
#else
	*reinterpret_cast<float4*>(smem_dst) = *reinterpret_cast<const float4*>(gmem_src);
#endif
}

__forceinline__ __device__ void cp_async_commit_group()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__forceinline__ __device__ void cp_async_wait_all()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	asm volatile("cp.async.wait_all;\n" ::);
#endif
}

// Step 03 tuning knobs for the persistent 16x16 top-k smem kernel.
// Keep these in one place so each experiment only changes three constants.
#define POLYSPLAT_TOPK_SMEM_PERSISTENT_WPB 20
#define POLYSPLAT_TOPK_SMEM_PERSISTENT_LAUNCH_THREADS (32 * POLYSPLAT_TOPK_SMEM_PERSISTENT_WPB)
#define POLYSPLAT_TOPK_SMEM_PERSISTENT_LAUNCH_MIN_BLOCKS 0
#define POLYSPLAT_TOPK_SMEM_PERSISTENT_MAXNREG 88

// V2 kernel: pure persistent work-stealing, NO top-k overhead.
// Same inner loop as default kernel but with persistent tile dispatch.
// Key advantage: dynamic load balancing across tiles with varying gaussian counts.
#define POLYSPLAT_TOPK_SMEM_PERSISTENT_V2_WPB 32
#define POLYSPLAT_TOPK_SMEM_PERSISTENT_V2_MAXNREG 64

// Named barrier: synchronize only threads in a specific pair.
// barrier_id: 0-15 (SM 90 supports 16 named barriers per block)
// thread_count: number of threads participating (must be constant across calls)
__forceinline__ __device__ void bar_sync_pair(int barrier_id, int thread_count)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(thread_count));
#else
	__syncthreads();
#endif
}

#if POLYSPLAT_TOPK_SMEM_PERSISTENT_MAXNREG > 0
#define POLYSPLAT_TOPK_SMEM_PERSISTENT_LAUNCH_BOUNDS
#define POLYSPLAT_TOPK_SMEM_PERSISTENT_MAXNREG_ATTR __maxnreg__(POLYSPLAT_TOPK_SMEM_PERSISTENT_MAXNREG)
#else
#if POLYSPLAT_TOPK_SMEM_PERSISTENT_LAUNCH_MIN_BLOCKS > 0
#define POLYSPLAT_TOPK_SMEM_PERSISTENT_LAUNCH_BOUNDS \
	__launch_bounds__(POLYSPLAT_TOPK_SMEM_PERSISTENT_LAUNCH_THREADS, POLYSPLAT_TOPK_SMEM_PERSISTENT_LAUNCH_MIN_BLOCKS)
#else
#define POLYSPLAT_TOPK_SMEM_PERSISTENT_LAUNCH_BOUNDS \
	__launch_bounds__(POLYSPLAT_TOPK_SMEM_PERSISTENT_LAUNCH_THREADS)
#endif
#define POLYSPLAT_TOPK_SMEM_PERSISTENT_MAXNREG_ATTR
#endif

__forceinline__ __device__ void pixel_shader(float3& C, float& T, float pixf_x, float pixf_y, float2 xy, float4 con_o, float3 rgb)
{
	float dx = xy.x - pixf_x;
	float dy = xy.y - pixf_y;
	float power = con_o.w + con_o.x * dx * dx + con_o.z * dy * dy + con_o.y * dx * dy;
	float alpha;
	asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(alpha) : "f"(power));
	C.x += rgb.x * (alpha * T);
	C.y += rgb.y * (alpha * T);
	C.z += rgb.z * (alpha * T);
	T -= alpha * T;
}

template<int THREAD_X, int THREAD_Y>
__forceinline__ __device__ void pixel_shader_axis_shared_16x16(
	float3 (&C)[THREAD_Y][THREAD_X],
	float (&T)[THREAD_Y][THREAD_X],
	int tile_origin_x, int tile_origin_y,
	int thread_col_base, int thread_row_base,
	float2 xy, float4 con_o, float3 rgb)
{
	float x_sq_term[THREAD_X];
	float x_cross_term[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
	{
		float dx = xy.x - (float)(tile_origin_x + thread_col_base + j);
		x_sq_term[j] = con_o.x * dx * dx;
		x_cross_term[j] = con_o.y * dx;
	}

	float y_delta[THREAD_Y];
	float y_sq_term[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
		float dy = xy.y - (float)(tile_origin_y + thread_row_base + i);
		y_delta[i] = dy;
		y_sq_term[i] = con_o.z * dy * dy;
	}

#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
	#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			float power = fmaf(x_cross_term[j], y_delta[i], con_o.w + x_sq_term[j] + y_sq_term[i]);
			float alpha = fast_ex2_ftz_f32(power);
			float weight = alpha * T[i][j];
			C[i][j].x += rgb.x * weight;
			C[i][j].y += rgb.y * weight;
			C[i][j].z += rgb.z * weight;
			T[i][j] -= weight;
		}
	}
}

template<int THREAD_X, int THREAD_Y>
__forceinline__ __device__ void pixel_shader_axis_shared_16x16_unroll2(
	float3 (&C)[THREAD_Y][THREAD_X],
	float (&T)[THREAD_Y][THREAD_X],
	int tile_origin_x, int tile_origin_y,
	int thread_col_base, int thread_row_base,
	float2 xy, float4 con_o, float3 rgb)
{
	float x_sq_term[THREAD_X];
	float x_cross_term[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
	{
		float dx = xy.x - (float)(tile_origin_x + thread_col_base + j);
		x_sq_term[j] = con_o.x * dx * dx;
		x_cross_term[j] = con_o.y * dx;
	}

	float y_delta[THREAD_Y];
	float y_sq_term[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
		float dy = xy.y - (float)(tile_origin_y + thread_row_base + i);
		y_delta[i] = dy;
		y_sq_term[i] = con_o.z * dy * dy;
	}

#pragma unroll 2
	for (int i = 0; i < THREAD_Y; i++)
	{
	#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			float power = fmaf(x_cross_term[j], y_delta[i], con_o.w + x_sq_term[j] + y_sq_term[i]);
			float alpha = fast_ex2_ftz_f32(power);
			float weight = alpha * T[i][j];
			C[i][j].x += rgb.x * weight;
			C[i][j].y += rgb.y * weight;
			C[i][j].z += rgb.z * weight;
			T[i][j] -= weight;
		}
	}
}

// Dynamic Thresholding variant of pixel_shader_axis_shared_16x16.
// Checks `power < log2(1/255) = -7.994353375f` BEFORE ex2 to skip invisible
// Gaussians (alpha < 1/255) without computing the exp2. For Gaussians that are
// too faint to matter, this saves the ex2 + 3 FMAs for weight/color/T updates.
// The threshold -7.994353375f == log2(1/255). con_o.w already absorbs log2_opacity,
// so the check is equivalent to `opacity * exp(-0.5 * sigma) < 1/255`.
template<int THREAD_X, int THREAD_Y>
__forceinline__ __device__ void pixel_shader_axis_shared_16x16_dt(
	float3 (&C)[THREAD_Y][THREAD_X],
	float (&T)[THREAD_Y][THREAD_X],
	int tile_origin_x, int tile_origin_y,
	int thread_col_base, int thread_row_base,
	float2 xy, float4 con_o, float3 rgb)
{
	float x_sq_term[THREAD_X];
	float x_cross_term[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
	{
		float dx = xy.x - (float)(tile_origin_x + thread_col_base + j);
		x_sq_term[j] = con_o.x * dx * dx;
		x_cross_term[j] = con_o.y * dx;
	}

	float y_delta[THREAD_Y];
	float y_sq_term[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
		float dy = xy.y - (float)(tile_origin_y + thread_row_base + i);
		y_delta[i] = dy;
		y_sq_term[i] = con_o.z * dy * dy;
	}

	// log2(1/255) = -log2(255) ≈ -7.994353375
	constexpr float LOG2_INV255 = -7.994353375f;

#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
	#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			float power = fmaf(x_cross_term[j], y_delta[i], con_o.w + x_sq_term[j] + y_sq_term[i]);
			if (power >= LOG2_INV255)
			{
				float alpha = fast_ex2_ftz_f32(power);
				float weight = alpha * T[i][j];
				C[i][j].x += rgb.x * weight;
				C[i][j].y += rgb.y * weight;
				C[i][j].z += rgb.z * weight;
				T[i][j] -= weight;
			}
		}
	}
}

__forceinline__ __device__ uint8_t encode(float x)
{
	return (uint8_t)min(max(0.0f, x * 255.0f), 255.0f);
}

__forceinline__ __device__ void write_color(uchar3* __restrict__ out_color,
	float3 bg_color, int2 pix, int width, int height, float3 C, float T)
{
	if (pix.x < width && pix.y < height)
	{
		int pix_id = width * pix.y + pix.x;
		if (T < 0.0001f)
		{
			T = 0.0f;
		}
		out_color[pix_id].x = encode(C.x + T * bg_color.x);
		out_color[pix_id].y = encode(C.y + T * bg_color.y);
		out_color[pix_id].z = encode(C.z + T * bg_color.z);
	}
}

struct render_load_info
{
	const void* data[POLYSPLAT_WARP_SIZE] = { nullptr };
	const void* topk_data[POLYSPLAT_WARP_SIZE] = { nullptr }; // alternative pointers for topk packed hits
	int lg2_scale[POLYSPLAT_WARP_SIZE] = { 0 };

	render_load_info(const uint32_t* point_list, const float2* points_xy, const float4* rgb_depth, const float4* conic_opacity,
		const float2* topk_xy = nullptr, const float4* topk_rgb_depth = nullptr, const float4* topk_conic = nullptr)
	{
		for (int lane = 0; lane < 32; lane++)
		{
			switch (lane)
			{
			case 0:
				data[lane] = point_list;
				lg2_scale[lane] = 2;
				break;
			case 4:
				data[lane] = point_list;
				lg2_scale[lane] = 2;
				break;
			case 8:
				data[lane] = &points_xy->x;
				lg2_scale[lane] = 3;
				break;
			case 9:
				data[lane] = &points_xy->y;
				lg2_scale[lane] = 3;
				break;
				break;
			case 12:
				data[lane] = &points_xy->x;
				lg2_scale[lane] = 3;
				break;
			case 13:
				data[lane] = &points_xy->y;
				lg2_scale[lane] = 3;
				break;
			case 16:
				data[lane] = &rgb_depth->x;
				lg2_scale[lane] = 4;
				break;
			case 17:
				data[lane] = &rgb_depth->y;
				lg2_scale[lane] = 4;
				break;
			case 18:
				data[lane] = &rgb_depth->z;
				lg2_scale[lane] = 4;
				break;
			case 19:
				data[lane] = nullptr;
				lg2_scale[lane] = 0;
				break;
			case 20:
				data[lane] = &rgb_depth->x;
				lg2_scale[lane] = 4;
				break;
			case 21:
				data[lane] = &rgb_depth->y;
				lg2_scale[lane] = 4;
				break;
			case 22:
				data[lane] = &rgb_depth->z;
				lg2_scale[lane] = 4;
				break;
			case 23:
				data[lane] = nullptr;
				lg2_scale[lane] = 0;
				break;
			case 24:
				data[lane] = &conic_opacity->x;
				lg2_scale[lane] = 4;
				break;
			case 25:
				data[lane] = &conic_opacity->y;
				lg2_scale[lane] = 4;
				break;
			case 26:
				data[lane] = &conic_opacity->z;
				lg2_scale[lane] = 4;
				break;
			case 27:
				data[lane] = &conic_opacity->w;
				lg2_scale[lane] = 4;
				break;
			case 28:
				data[lane] = &conic_opacity->x;
				lg2_scale[lane] = 4;
				break;
			case 29:
				data[lane] = &conic_opacity->y;
				lg2_scale[lane] = 4;
				break;
			case 30:
				data[lane] = &conic_opacity->z;
				lg2_scale[lane] = 4;
				break;
			case 31:
				data[lane] = &conic_opacity->w;
				lg2_scale[lane] = 4;
				break;
			}
			// Populate topk_data: same layout as data but pointing to compact topk arrays.
			// lg2_scale is shared between data and topk_data.
			if (topk_xy)
			{
				switch (lane)
				{
				case 8: case 12: topk_data[lane] = &topk_xy->x; break;
				case 9: case 13: topk_data[lane] = &topk_xy->y; break;
				case 16: case 20: topk_data[lane] = &topk_rgb_depth->x; break;
				case 17: case 21: topk_data[lane] = &topk_rgb_depth->y; break;
				case 18: case 22: topk_data[lane] = &topk_rgb_depth->z; break;
				case 24: case 28: topk_data[lane] = &topk_conic->x; break;
				case 25: case 29: topk_data[lane] = &topk_conic->y; break;
				case 26: case 30: topk_data[lane] = &topk_conic->z; break;
				case 27: case 31: topk_data[lane] = &topk_conic->w; break;
				}
			}
		}
	}

	// Packed features constructor: features stored in 32B-aligned packed struct
	// Layout per gaussian: [xy.x, xy.y, r, g, b, con.x, con.y, con.z] (8 floats = 32B)
	// con.w (log2_opacity) is read from the original conic_opacity array
	// Feature lanes use lg2_scale=5 (stride 32B), con.w uses lg2_scale=4 (stride 16B)
	render_load_info(const uint32_t* point_list, const float* packed_features, const float4* conic_opacity)
	{
		const char* base = reinterpret_cast<const char*>(packed_features);
		for (int lane = 0; lane < 32; lane++)
		{
			switch (lane)
			{
			case 0: case 4: // ID prefetch
				data[lane] = point_list;
				lg2_scale[lane] = 2;
				break;
			// G0 feature lanes
			case 8:  data[lane] = base + 0;  lg2_scale[lane] = 5; break; // xy.x
			case 9:  data[lane] = base + 4;  lg2_scale[lane] = 5; break; // xy.y
			case 16: data[lane] = base + 8;  lg2_scale[lane] = 5; break; // r
			case 17: data[lane] = base + 12; lg2_scale[lane] = 5; break; // g
			case 18: data[lane] = base + 16; lg2_scale[lane] = 5; break; // b
			case 24: data[lane] = base + 20; lg2_scale[lane] = 5; break; // con.x
			case 25: data[lane] = base + 24; lg2_scale[lane] = 5; break; // con.y
			case 26: data[lane] = base + 28; lg2_scale[lane] = 5; break; // con.z
			case 27: data[lane] = &conic_opacity->w; lg2_scale[lane] = 4; break; // con.w from original
			// G1 feature lanes
			case 12: data[lane] = base + 0;  lg2_scale[lane] = 5; break;
			case 13: data[lane] = base + 4;  lg2_scale[lane] = 5; break;
			case 20: data[lane] = base + 8;  lg2_scale[lane] = 5; break;
			case 21: data[lane] = base + 12; lg2_scale[lane] = 5; break;
			case 22: data[lane] = base + 16; lg2_scale[lane] = 5; break;
			case 28: data[lane] = base + 20; lg2_scale[lane] = 5; break;
			case 29: data[lane] = base + 24; lg2_scale[lane] = 5; break;
			case 30: data[lane] = base + 28; lg2_scale[lane] = 5; break;
			case 31: data[lane] = &conic_opacity->w; lg2_scale[lane] = 4; break; // con.w from original
			default: // lanes 1-3, 5-7, 10-11, 14-15, 19, 23: unused
				data[lane] = nullptr;
				lg2_scale[lane] = 0;
				break;
			}
		}
	}
};

struct topk_smem_buffers
{
	float* xy_x;
	float* xy_y;
	float* rgb_r;
	float* rgb_g;
	float* rgb_b;
	float* con_x;
	float* con_y;
	float* con_z;
	float* con_w;
};

__forceinline__ __device__ topk_smem_buffers make_topk_smem_buffers(float* smem_base, int num_topk)
{
	topk_smem_buffers out;
	out.xy_x = smem_base;
	out.xy_y = out.xy_x + num_topk;
	out.rgb_r = out.xy_y + num_topk;
	out.rgb_g = out.rgb_r + num_topk;
	out.rgb_b = out.rgb_g + num_topk;
	out.con_x = out.rgb_b + num_topk;
	out.con_y = out.con_x + num_topk;
	out.con_z = out.con_y + num_topk;
	out.con_w = out.con_z + num_topk;
	return out;
}

__forceinline__ __device__ float load_lane_value(const void* data, int lg2_scale, int offset)
{
	const float* ptr = reinterpret_cast<const float*>(
		reinterpret_cast<const char*>(data) + ((uint32_t)offset << lg2_scale));
	return __ldg(ptr);
}

// Load with packed topk support via global memory compact arrays (no smem needed).
// For feature lanes: if offset has bit 31 set, bits 0..30 = slot index into topk_data.
// Otherwise offset is a plain gaussian_id loaded from data.
// For lanes 0/4: offset is a point_list array index, never has bit 31 set.
__forceinline__ __device__ float load_lane_value_topk_packed(
	const void* data, const void* topk_data, int lg2_scale, int offset, int lane)
{
	if (lane >= 8 && lane != 19 && lane != 23 && (offset & POLYSPLAT_TOPK_FLAG_BIT))
	{
		int slot = offset & 0x7FFFFFFF;
		return load_lane_value(topk_data, lg2_scale, slot);
	}
	return load_lane_value(data, lg2_scale, offset);
}

// Fast path with precomputed is_feature_lane
__forceinline__ __device__ float load_lane_value_topk_packed_fast(
	const void* data, const void* topk_data, int lg2_scale, int offset, bool is_feature_lane)
{
	if (is_feature_lane && (offset & POLYSPLAT_TOPK_FLAG_BIT))
	{
		int slot = offset & 0x7FFFFFFF;
		return load_lane_value(topk_data, lg2_scale, slot);
	}
	return load_lane_value(data, lg2_scale, offset);
}

//READ：把 top-k 个 Gaussian 的参数，从全局内存搬到 shared memory 里，供后续更快访问。
__forceinline__ __device__ void stage_topk_gaussians_to_smem(
	int lane,
	int num_topk,
	const float2* __restrict__ topk_xy,
	const float4* __restrict__ topk_rgb_depth,
	const float4* __restrict__ topk_conic,
	const topk_smem_buffers& smem)
{
	for (int i = lane; i < num_topk; i += POLYSPLAT_WARP_SIZE)
	{
		float2 xy = __ldg(&topk_xy[i]);
		smem.xy_x[i] = xy.x;
		smem.xy_y[i] = xy.y;

		float4 rgb_depth = __ldg(&topk_rgb_depth[i]);
		smem.rgb_r[i] = rgb_depth.x;
		smem.rgb_g[i] = rgb_depth.y;
		smem.rgb_b[i] = rgb_depth.z;

		float4 conic = __ldg(&topk_conic[i]);
		smem.con_x[i] = conic.x;
		smem.con_y[i] = conic.y;
		smem.con_z[i] = conic.z;
		smem.con_w[i] = conic.w;
	}
	__syncthreads();
}

__forceinline__ __device__ bool lane_uses_gaussian_features(int lane)
{
	return lane >= 8 && lane != 19 && lane != 23;
}

// Returns the per-lane smem base pointer for topk lookups.
// Called once before the main loop to avoid the switch on every iteration.
__forceinline__ __device__ const float* get_topk_smem_lane_ptr(
	int lane,
	const topk_smem_buffers& smem)
{
	switch (lane)
	{
	case 8:
	case 12:
		return smem.xy_x;
	case 9:
	case 13:
		return smem.xy_y;
	case 16:
	case 20:
		return smem.rgb_r;
	case 17:
	case 21:
		return smem.rgb_g;
	case 18:
	case 22:
		return smem.rgb_b;
	case 24:
	case 28:
		return smem.con_x;
	case 25:
	case 29:
		return smem.con_y;
	case 26:
	case 30:
		return smem.con_z;
	case 27:
	case 31:
		return smem.con_w;
	default:
		return nullptr;
	}
}

__forceinline__ __device__ float load_topk_smem_value(
	int lane,
	int slot,
	const topk_smem_buffers& smem)
{
	switch (lane)
	{
	case 8:
	case 12:
		return smem.xy_x[slot];
	case 9:
	case 13:
		return smem.xy_y[slot];
	case 16:
	case 20:
		return smem.rgb_r[slot];
	case 17:
	case 21:
		return smem.rgb_g[slot];
	case 18:
	case 22:
		return smem.rgb_b[slot];
	case 24:
	case 28:
		return smem.con_x[slot];
	case 25:
	case 29:
		return smem.con_y[slot];
	case 26:
	case 30:
		return smem.con_z[slot];
	case 27:
	case 31:
		return smem.con_w[slot];
	default:
		return 0.0f;
	}
}

__forceinline__ __device__ float load_lane_value_topk_smem(
	const void* data,
	int lg2_scale,
	int offset,
	int lane,
	int num_topk,
	const topk_smem_buffers& smem)
{
	// For feature lanes: offset is either a plain gaussian_id (bit 31=0)
	// or a packed topk marker (bit 31=1, bits 0..30 = smem slot index).
	// For lanes 0/4: offset is a point_list array index (never has bit 31 set).
	if (lane_uses_gaussian_features(lane) && num_topk > 0 && (offset & POLYSPLAT_TOPK_FLAG_BIT))
	{
		int slot = offset & 0x7FFFFFFF;
		return load_topk_smem_value(lane, slot, smem);
	}
	return load_lane_value(data, lg2_scale, offset);
}

// Fast path: uses precomputed per-lane smem pointer (avoids switch in hot loop).
__forceinline__ __device__ float load_lane_value_topk_smem_fast(
	const void* data,
	int lg2_scale,
	int offset,
	bool is_feature_lane,
	int num_topk,
	const float* smem_lane_ptr)
{
	if (is_feature_lane && num_topk > 0 && (offset & POLYSPLAT_TOPK_FLAG_BIT))
	{
		int slot = offset & 0x7FFFFFFF;
		return smem_lane_ptr[slot];
	}
	return load_lane_value(data, lg2_scale, offset);
}

struct topk_async_smem_buffers
{
	float2* xy;
	float4* rgbd;
	float4* conic;
};

struct topk_async_smem_lane_view
{
	const float* base;
	int lg2_stride;
};

__forceinline__ __host__ __device__ int topk_async_smem_bytes_required(int num_topk)
{
	int xy_bytes = num_topk * static_cast<int>(sizeof(float2));
	int xy_bytes_aligned = align_up_pow2(xy_bytes, static_cast<int>(sizeof(float4)));
	return xy_bytes_aligned + num_topk * static_cast<int>(sizeof(float4) + sizeof(float4));
}

__forceinline__ __device__ topk_async_smem_buffers make_topk_async_smem_buffers(void* smem_base, int num_topk)
{
	topk_async_smem_buffers out;
	char* base = reinterpret_cast<char*>(smem_base);
	int xy_bytes = num_topk * static_cast<int>(sizeof(float2));
	int xy_bytes_aligned = align_up_pow2(xy_bytes, static_cast<int>(sizeof(float4)));
	out.xy = reinterpret_cast<float2*>(base);
	out.rgbd = reinterpret_cast<float4*>(base + xy_bytes_aligned);
	out.conic = out.rgbd + num_topk;
	return out;
}

__forceinline__ __device__ topk_async_smem_lane_view get_topk_async_smem_lane_view(
	int lane,
	const topk_async_smem_buffers& smem)
{
	const float* xy_base = reinterpret_cast<const float*>(smem.xy);
	const float* rgbd_base = reinterpret_cast<const float*>(smem.rgbd);
	const float* conic_base = reinterpret_cast<const float*>(smem.conic);
	switch (lane)
	{
	case 8:
	case 12:
		return { xy_base + 0, 1 };
	case 9:
	case 13:
		return { xy_base + 1, 1 };
	case 16:
	case 20:
		return { rgbd_base + 0, 2 };
	case 17:
	case 21:
		return { rgbd_base + 1, 2 };
	case 18:
	case 22:
		return { rgbd_base + 2, 2 };
	case 24:
	case 28:
		return { conic_base + 0, 2 };
	case 25:
	case 29:
		return { conic_base + 1, 2 };
	case 26:
	case 30:
		return { conic_base + 2, 2 };
	case 27:
	case 31:
		return { conic_base + 3, 2 };
	default:
		return { nullptr, 0 };
	}
}

__forceinline__ __device__ void stage_topk_gaussians_to_async_smem(
	int thread_id,
	int total_threads,
	int num_topk,
	const float2* __restrict__ topk_xy,
	const float4* __restrict__ topk_rgb_depth,
	const float4* __restrict__ topk_conic,
	const topk_async_smem_buffers& smem)
{
	for (int i = thread_id; i < num_topk; i += total_threads)
	{
		cp_async_ca_shared_global_8(&smem.xy[i], &topk_xy[i]);
		cp_async_ca_shared_global_16(&smem.rgbd[i], &topk_rgb_depth[i]);
		cp_async_ca_shared_global_16(&smem.conic[i], &topk_conic[i]);
	}
	cp_async_commit_group();
}

__forceinline__ __device__ float load_lane_value_topk_async_smem_fast(
	const void* data,
	int lg2_scale,
	int offset,
	bool is_feature_lane,
	int num_topk,
	const float* smem_lane_ptr,
	int smem_lane_lg2_stride)
{
	if (is_feature_lane && num_topk > 0 && (offset & POLYSPLAT_TOPK_FLAG_BIT))
	{
		int slot = offset & 0x7FFFFFFF;
		return smem_lane_ptr[slot << smem_lane_lg2_stride];
	}
	return load_lane_value(data, lg2_scale, offset);
}

__forceinline__ __device__ void get_gaussian_features(float2& xy, float3& rgb, float4& con_o, float buf, int offset)
{
	xy = {
		__shfl_sync(~0, buf, 8 + offset),
		__shfl_sync(~0, buf, 9 + offset)
	};
	rgb = {
		__shfl_sync(~0, buf, 16 + offset),
		__shfl_sync(~0, buf, 17 + offset),
		__shfl_sync(~0, buf, 18 + offset)
	};
	con_o = {
		__shfl_sync(~0, buf, 24 + offset),
		__shfl_sync(~0, buf, 25 + offset),
		__shfl_sync(~0, buf, 26 + offset),
		__shfl_sync(~0, buf, 27 + offset)
	};
}

// Default render kernel with precomputed tile ranges (avoids binary search).
// Identical inner loop as renderCUDA, same register count (64), same occupancy.
template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y, bool USE_UNROLL2 = false, bool HAS_TOPK_PACKED = false>
__global__ __launch_bounds__(32) void renderCUDA_preranges(
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int2* __restrict__ tile_ranges,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	uint32_t tile_id = blockIdx.y * x_blocks + blockIdx.x;
	int lane = threadIdx.y * blockDim.x + threadIdx.x;

	int2 range = make_int2(0, 0);
	if (lane == 0)
		range = __ldg(&tile_ranges[tile_id]);
	range.x = __shfl_sync(~0, range.x, 0);
	range.y = __shfl_sync(~0, range.y, 0);

	int tile_origin_x = (int)blockIdx.x * BLOCK_X;
	int tile_origin_y = (int)blockIdx.y * BLOCK_Y;
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	int pix_x[THREAD_X];
	float pixf_x[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
	{
		pix_x[j] = tile_origin_x + thread_col_base + j;
		pixf_x[j] = (float)pix_x[j];
	}

	int pix_y[THREAD_Y];
	float pixf_y[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
		pix_y[i] = tile_origin_y + thread_row_base + i;
		pixf_y[i] = (float)pix_y[i];
	}

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			T[i][j] = 1.0f;

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			C[i][j] = { 0.0f, 0.0f, 0.0f };

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
			offset = point_id + 2;
		else if (lane == 4)
			offset = point_id + 3;
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
			offset = point_list[point_id + 0];
		else if (point_id + 1 < range.y)
			offset = point_list[point_id + 1];

		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
			load_enable = load_enable && point_id + 2 < range.y;
		else if (lane == 4)
			load_enable = load_enable && point_id + 3 < range.y;
		else if ((lane & 4) == 0)
			load_enable = load_enable && point_id + 0 < range.y;
		else
			load_enable = load_enable && point_id + 1 < range.y;

		if (load_enable)
			buf = load_lane_value(data, lg2_scale, offset);

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			if constexpr (BLOCK_X == 16 && BLOCK_Y == 16)
			{
				pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
					C, T, tile_origin_x, tile_origin_y,
					thread_col_base, thread_row_base, xy, con_o, rgb);
			}
			else
			{
#pragma unroll
				for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
					for (int j = 0; j < THREAD_X; j++)
						pixel_shader(C[i][j], T[i][j], pixf_x[j], pixf_y[i], xy, con_o, rgb);
			}

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			if constexpr (BLOCK_X == 16 && BLOCK_Y == 16)
			{
				pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
					C, T, tile_origin_x, tile_origin_y,
					thread_col_base, thread_row_base, xy, con_o, rgb);
			}
			else
			{
#pragma unroll
				for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
					for (int j = 0; j < THREAD_X; j++)
						pixel_shader(C[i][j], T[i][j], pixf_x[j], pixf_y[i], xy, con_o, rgb);
			}

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			point_id += 2;
			buf = ldg_buf;
		}

		// Tail loop with bounds checking
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			if (lane == 0)
				load_enable = load_enable && point_id + 4 < range.y;
			else if (lane == 4)
				load_enable = load_enable && point_id + 5 < range.y;
			else if ((lane & 4) == 0)
				load_enable = load_enable && point_id + 2 < range.y;
			else
				load_enable = load_enable && point_id + 3 < range.y;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			if constexpr (BLOCK_X == 16 && BLOCK_Y == 16)
			{
				pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
					C, T, tile_origin_x, tile_origin_y,
					thread_col_base, thread_row_base, xy, con_o, rgb);
			}
			else
			{
#pragma unroll
				for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
					for (int j = 0; j < THREAD_X; j++)
						pixel_shader(C[i][j], T[i][j], pixf_x[j], pixf_y[i], xy, con_o, rgb);
			}

			if (point_id + 1 >= range.y) break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			if constexpr (BLOCK_X == 16 && BLOCK_Y == 16)
			{
				pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
					C, T, tile_origin_x, tile_origin_y,
					thread_col_base, thread_row_base, xy, con_o, rgb);
			}
			else
			{
#pragma unroll
				for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
					for (int j = 0; j < THREAD_X; j++)
						pixel_shader(C[i][j], T[i][j], pixf_x[j], pixf_y[i], xy, con_o, rgb);
			}

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix_x[j] < width && pix_y[i] < height)
				{
					int pix_id = width * pix_y[i] + pix_x[j];
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
	}
}

// Naive direct-load render kernel with preranges.
// Same 32-gaussian batch loop structure as renderCUDA_preranges_smem,
// but uses plain __ldg synchronous loads per gaussian (no cp.async, no shared memory).
// Ablation: isolates whether speedup comes from cp.async or from eliminating render_load_info.
template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA_preranges_naive(
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int2* __restrict__ tile_ranges,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BATCH = 32;
	const uint32_t tile_id = blockIdx.y * x_blocks + blockIdx.x;
	const int lane = threadIdx.y * blockDim.x + threadIdx.x;

	int2 range;
	if (lane == 0) range = __ldg(&tile_ranges[tile_id]);
	range.x = __shfl_sync(~0, range.x, 0);
	range.y = __shfl_sync(~0, range.y, 0);

	const int tile_origin_x = (int)blockIdx.x * BLOCK_X;
	const int tile_origin_y = (int)blockIdx.y * BLOCK_Y;
	const int thread_col_base = (int)threadIdx.x * THREAD_X;
	const int thread_row_base = (int)threadIdx.y * THREAD_Y;

	float T[THREAD_Y][THREAD_X];
	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++) { T[i][j] = 1.0f; C[i][j] = {0.0f, 0.0f, 0.0f}; }

	bool early_exit = false;
	for (int pos = range.x; pos < range.y && !early_exit; pos += BATCH)
	{
		const int bsize = min(BATCH, range.y - pos);
		for (int g = 0; g < bsize; g++)
		{
			const uint32_t gid = __ldg(&point_list[pos + g]);
			const float2 xy = __ldg(&points_xy[gid]);
			const float4 con_o = __ldg(&conic_opacity[gid]);
			const float4 rgbd = __ldg(&rgb_depth[gid]);
			const float3 rgb = { rgbd.x, rgbd.y, rgbd.z };
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			if ((g & 7) == 7) {
				bool sat = true;
#pragma unroll
				for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
					for (int j = 0; j < THREAD_X; j++)
						sat = sat && T[i][j] < 0.0001f;
				if (__all_sync(~0, sat)) { early_exit = true; break; }
			}
		}
	}
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			write_color(out_color, bg_color,
				{ tile_origin_x + thread_col_base + j, tile_origin_y + thread_row_base + i },
				width, height, C[i][j], T[i][j]);
}

// Shared-memory batched render kernel with preranges.
// Key optimization: cooperative batch loading of 32 gaussians by 32 threads
// into shared memory, with cp.async double buffering to overlap loads with compute.
// Eliminates render_load_info overhead and reduces per-gaussian memory requests.
template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA_preranges_smem(
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int2* __restrict__ tile_ranges,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BATCH = 32;

	// Double-buffered shared memory for gaussian features
	__shared__ float2 s_xy[2][BATCH];
	__shared__ float4 s_rgb[2][BATCH];   // .x,.y,.z = RGB, .w unused
	__shared__ float4 s_con[2][BATCH];   // conic_opacity (x,y,z,w)

	const uint32_t tile_id = blockIdx.y * x_blocks + blockIdx.x;
	const int lane = threadIdx.y * blockDim.x + threadIdx.x;

	// Load tile range (lane 0 reads, broadcast to warp)
	int2 range;
	if (lane == 0)
		range = __ldg(&tile_ranges[tile_id]);
	range.x = __shfl_sync(~0, range.x, 0);
	range.y = __shfl_sync(~0, range.y, 0);

	const int tile_origin_x = (int)blockIdx.x * BLOCK_X;
	const int tile_origin_y = (int)blockIdx.y * BLOCK_Y;
	const int thread_col_base = (int)threadIdx.x * THREAD_X;
	const int thread_row_base = (int)threadIdx.y * THREAD_Y;

	float T[THREAD_Y][THREAD_X];
	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			T[i][j] = 1.0f;
			C[i][j] = { 0.0f, 0.0f, 0.0f };
		}

	const int total = range.y - range.x;
	if (total > 0)
	{
		// ---- Prologue: load first batch into buffer 0 ----
		const int b0 = min(BATCH, total);
		if (lane < b0)
		{
			const int gid = __ldg(&point_list[range.x + lane]);
			cp_async_ca_shared_global_8(&s_xy[0][lane], &points_xy[gid]);
			cp_async_ca_shared_global_16(&s_rgb[0][lane], &rgb_depth[gid]);
			cp_async_ca_shared_global_16(&s_con[0][lane], &conic_opacity[gid]);
		}
		cp_async_commit_group();
		cp_async_wait_all();
		__syncwarp();

		int cur = 0;
		int pos = range.x;

		while (pos < range.y)
		{
			const int bsize = min(BATCH, range.y - pos);
			const int nxt = 1 - cur;
			const int npos = pos + BATCH;
			const int nsize = max(0, min(BATCH, range.y - npos));

			// Issue cp.async for NEXT batch (overlaps with current compute)
			if (nsize > 0)
			{
				if (lane < nsize)
				{
					const int gid = __ldg(&point_list[npos + lane]);
					cp_async_ca_shared_global_8(&s_xy[nxt][lane], &points_xy[gid]);
					cp_async_ca_shared_global_16(&s_rgb[nxt][lane], &rgb_depth[gid]);
					cp_async_ca_shared_global_16(&s_con[nxt][lane], &conic_opacity[gid]);
				}
				cp_async_commit_group();
			}

			// Process current batch from shared memory
			bool early_exit = false;
			for (int g = 0; g < bsize; g++)
			{
				const float2 xy = s_xy[cur][g];
				const float4 con_o = s_con[cur][g];
				const float4 rgbd = s_rgb[cur][g];
				const float3 rgb = { rgbd.x, rgbd.y, rgbd.z };

				pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
					C, T, tile_origin_x, tile_origin_y,
					thread_col_base, thread_row_base, xy, con_o, rgb);

				// Check saturation every 8 gaussians
				if ((g & 7) == 7)
				{
					bool sat = true;
#pragma unroll
					for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
						for (int j = 0; j < THREAD_X; j++)
							sat = sat && T[i][j] < 0.0001f;
					if (__all_sync(~0, sat))
					{
						early_exit = true;
						break;
					}
				}
			}

			// Wait for next batch async copies before swapping
			if (nsize > 0)
			{
				cp_async_wait_all();
				__syncwarp();
			}

			if (early_exit) break;
			pos += BATCH;
			cur = nxt;
		}

		// Write final pixel colors
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color,
					{ tile_origin_x + thread_col_base + j, tile_origin_y + thread_row_base + i },
					width, height, C[i][j], T[i][j]);
	}
	else
	{
		// Empty tile - write background
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int px = tile_origin_x + thread_col_base + j;
				int py = tile_origin_y + thread_row_base + i;
				if (px < width && py < height)
				{
					int pix_id = width * py + px;
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
	}
}

// Non-inline wrapper for find_tile_range to prevent the binary search's
// complex control flow from polluting the render loop's code generation.
__noinline__ __device__ int2 find_tile_range_noinline(
	const uint64_t* __restrict__ sorted_keys, int num_rendered, uint32_t tile_id)
{
	return find_tile_range(sorted_keys, num_rendered, tile_id);
}

// V2: Fused binary search + cp.async smem batched render.
// Eliminates the separate computeTileRanges kernel by doing the binary search
// inline in the prologue (lane 0 only, hidden by warp scheduling).
template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA_preranges_smem_v2(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BATCH = 32;

	// Double-buffered shared memory for gaussian features
	__shared__ float2 s_xy[2][BATCH];
	__shared__ float4 s_rgb[2][BATCH];
	__shared__ float4 s_con[2][BATCH];

	const uint32_t tile_id = blockIdx.y * x_blocks + blockIdx.x;
	const int lane = threadIdx.y * blockDim.x + threadIdx.x;

	// Fused binary search: lane 0 finds tile range, broadcast to warp
	// Use __noinline__ wrapper to prevent binary search from polluting
	// the render loop's register allocation and instruction scheduling.
	int2 range;
	if (lane == 0)
		range = find_tile_range_noinline(sorted_keys, num_rendered, tile_id);
	range.x = __shfl_sync(~0, range.x, 0);
	range.y = __shfl_sync(~0, range.y, 0);

	const int tile_origin_x = (int)blockIdx.x * BLOCK_X;
	const int tile_origin_y = (int)blockIdx.y * BLOCK_Y;
	const int thread_col_base = (int)threadIdx.x * THREAD_X;
	const int thread_row_base = (int)threadIdx.y * THREAD_Y;

	float T[THREAD_Y][THREAD_X];
	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			T[i][j] = 1.0f;
			C[i][j] = { 0.0f, 0.0f, 0.0f };
		}

	const int total = range.y - range.x;
	if (total > 0)
	{
		// ---- Prologue: load first batch into buffer 0 ----
		const int b0 = min(BATCH, total);
		if (lane < b0)
		{
			const int gid = __ldg(&point_list[range.x + lane]);
			cp_async_ca_shared_global_8(&s_xy[0][lane], &points_xy[gid]);
			cp_async_ca_shared_global_16(&s_rgb[0][lane], &rgb_depth[gid]);
			cp_async_ca_shared_global_16(&s_con[0][lane], &conic_opacity[gid]);
		}
		cp_async_commit_group();
		cp_async_wait_all();
		__syncwarp();

		int cur = 0;
		int pos = range.x;

		while (pos < range.y)
		{
			const int bsize = min(BATCH, range.y - pos);
			const int nxt = 1 - cur;
			const int npos = pos + BATCH;
			const int nsize = max(0, min(BATCH, range.y - npos));

			// Issue cp.async for NEXT batch (overlaps with current compute)
			if (nsize > 0)
			{
				if (lane < nsize)
				{
					const int gid = __ldg(&point_list[npos + lane]);
					cp_async_ca_shared_global_8(&s_xy[nxt][lane], &points_xy[gid]);
					cp_async_ca_shared_global_16(&s_rgb[nxt][lane], &rgb_depth[gid]);
					cp_async_ca_shared_global_16(&s_con[nxt][lane], &conic_opacity[gid]);
				}
				cp_async_commit_group();
			}

			// Process current batch from shared memory
			bool early_exit = false;
			for (int g = 0; g < bsize; g++)
			{
				const float2 xy = s_xy[cur][g];
				const float4 con_o = s_con[cur][g];
				const float4 rgbd = s_rgb[cur][g];
				const float3 rgb = { rgbd.x, rgbd.y, rgbd.z };

				pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
					C, T, tile_origin_x, tile_origin_y,
					thread_col_base, thread_row_base, xy, con_o, rgb);

				// Check saturation every 8 gaussians
				if ((g & 7) == 7)
				{
					bool sat = true;
#pragma unroll
					for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
						for (int j = 0; j < THREAD_X; j++)
							sat = sat && T[i][j] < 0.0001f;
					if (__all_sync(~0, sat))
					{
						early_exit = true;
						break;
					}
				}
			}

			// Wait for next batch async copies before swapping
			if (nsize > 0)
			{
				cp_async_wait_all();
				__syncwarp();
			}

			if (early_exit) break;
			pos += BATCH;
			cur = nxt;
		}

		// Write final pixel colors
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color,
					{ tile_origin_x + thread_col_base + j, tile_origin_y + thread_row_base + i },
					width, height, C[i][j], T[i][j]);
	}
	else
	{
		// Empty tile - write background
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int px = tile_origin_x + thread_col_base + j;
				int py = tile_origin_y + thread_row_base + i;
				if (px < width && py < height)
				{
					int pix_id = width * py + px;
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
	}
}

// Atomic counter for dynamic tile scheduling in persistent kernels.
// (Forward declaration; also referenced by later persistent kernels.)
__device__ unsigned int g_tile_counter;

// ============================================================================
// Persistent work-stealing + cp.async smem batched render.
// Combines the latency-hiding benefits of preranges_smem (64 regs, 50% occ)
// with the tile scheduling benefits of reordered_persistent (spatial locality
// via Morton+zigzag ordering, dynamic load balance via work-stealing).
//
// Key advantage over reordered_persistent: uses cp.async smem loading instead
// of render_load_info pointer arrays, avoiding the +8 reg overhead that
// dropped occupancy from 50% to 43.75%.
// ============================================================================
template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA_preranges_smem_persistent(
	const uint32_t* __restrict__ point_list,
	int width, int height,
	int total_tiles,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int4* __restrict__ tile_desc,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BATCH = 32;

	// Double-buffered shared memory for gaussian features
	__shared__ float2 s_xy[2][BATCH];
	__shared__ float4 s_rgb[2][BATCH];
	__shared__ float4 s_con[2][BATCH];

	const int lane = threadIdx.y * blockDim.x + threadIdx.x;
	const int thread_col_base = (int)threadIdx.x * THREAD_X;
	const int thread_row_base = (int)threadIdx.y * THREAD_Y;

	// First-tile claim: static blockIdx.x mapping (avoids atomic contention
	// storm at kernel launch where all warps would hammer the same counter).
	int idx = (int)blockIdx.x;

	while (idx < total_tiles)
	{
		// Load packed tile descriptor: {col, row, range_start, range_end}
		int4 desc = make_int4(0, 0, 0, 0);
		if (lane == 0)
			desc = __ldg(&tile_desc[idx]);
		desc.x = __shfl_sync(~0u, desc.x, 0);
		desc.y = __shfl_sync(~0u, desc.y, 0);
		desc.z = __shfl_sync(~0u, desc.z, 0);
		desc.w = __shfl_sync(~0u, desc.w, 0);

		const int tile_origin_x = desc.x * BLOCK_X;
		const int tile_origin_y = desc.y * BLOCK_Y;
		const int range_start = desc.z;
		const int range_end = desc.w;

		float T[THREAD_Y][THREAD_X];
		float3 C[THREAD_Y][THREAD_X];
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				T[i][j] = 1.0f;
				C[i][j] = { 0.0f, 0.0f, 0.0f };
			}

		const int total = range_end - range_start;
		if (total > 0)
		{
			// ---- Prologue: load first batch into buffer 0 ----
			const int b0 = min(BATCH, total);
			if (lane < b0)
			{
				const int gid = __ldg(&point_list[range_start + lane]);
				cp_async_ca_shared_global_8(&s_xy[0][lane], &points_xy[gid]);
				cp_async_ca_shared_global_16(&s_rgb[0][lane], &rgb_depth[gid]);
				cp_async_ca_shared_global_16(&s_con[0][lane], &conic_opacity[gid]);
			}
			cp_async_commit_group();
			cp_async_wait_all();
			__syncwarp();

			int cur = 0;
			int pos = range_start;

			while (pos < range_end)
			{
				const int bsize = min(BATCH, range_end - pos);
				const int nxt = 1 - cur;
				const int npos = pos + BATCH;
				const int nsize = max(0, min(BATCH, range_end - npos));

				// Issue cp.async for NEXT batch (overlaps with current compute)
				if (nsize > 0)
				{
					if (lane < nsize)
					{
						const int gid = __ldg(&point_list[npos + lane]);
						cp_async_ca_shared_global_8(&s_xy[nxt][lane], &points_xy[gid]);
						cp_async_ca_shared_global_16(&s_rgb[nxt][lane], &rgb_depth[gid]);
						cp_async_ca_shared_global_16(&s_con[nxt][lane], &conic_opacity[gid]);
					}
					cp_async_commit_group();
				}

				// Process current batch from shared memory
				bool early_exit = false;
				for (int g = 0; g < bsize; g++)
				{
					const float2 xy = s_xy[cur][g];
					const float4 con_o = s_con[cur][g];
					const float4 rgbd = s_rgb[cur][g];
					const float3 rgb = { rgbd.x, rgbd.y, rgbd.z };

					pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
						C, T, tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base, xy, con_o, rgb);

					// Check saturation every 8 gaussians
					if ((g & 7) == 7)
					{
						bool sat = true;
#pragma unroll
						for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
							for (int j = 0; j < THREAD_X; j++)
								sat = sat && T[i][j] < 0.0001f;
						if (__all_sync(~0u, sat))
						{
							early_exit = true;
							break;
						}
					}
				}

				// Wait for next batch async copies before swapping
				if (nsize > 0)
				{
					cp_async_wait_all();
					__syncwarp();
				}

				if (early_exit) break;
				pos += BATCH;
				cur = nxt;
			}
		}

		// Write pixel colors for this tile
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int px = tile_origin_x + thread_col_base + j;
				int py = tile_origin_y + thread_row_base + i;
				if (total > 0)
					write_color(out_color, bg_color, { px, py }, width, height, C[i][j], T[i][j]);
				else if (px < width && py < height)
				{
					int pix_id = width * py + px;
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}

		// Claim next tile via work-stealing. Offset by gridDim.x so the
		// atomic range covers tiles beyond the static first wave.
		if (lane == 0)
			idx = (int)atomicAdd(&g_tile_counter, 1) + (int)gridDim.x;
		idx = __shfl_sync(~0u, idx, 0);
	}
}

// ============================================================================
// Lightweight persistent variant: uses precomputed tile_ranges directly
// (no CUB sort, no tile_desc). Zero extra preprocessing overhead vs
// preranges_smem. Benefits from work-stealing load balance only.
// ============================================================================
template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA_preranges_smem_persistent_lite(
	const uint32_t* __restrict__ point_list,
	int width, int height,
	int x_blocks, int total_tiles,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int2* __restrict__ tile_ranges,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BATCH = 32;

	__shared__ float2 s_xy[2][BATCH];
	__shared__ float4 s_rgb[2][BATCH];
	__shared__ float4 s_con[2][BATCH];

	const int lane = threadIdx.y * blockDim.x + threadIdx.x;
	const int thread_col_base = (int)threadIdx.x * THREAD_X;
	const int thread_row_base = (int)threadIdx.y * THREAD_Y;

	int idx = (int)blockIdx.x;

	while (idx < total_tiles)
	{
		const int tile_col = idx % x_blocks;
		const int tile_row = idx / x_blocks;
		const int tile_origin_x = tile_col * BLOCK_X;
		const int tile_origin_y = tile_row * BLOCK_Y;

		int2 range = make_int2(0, 0);
		if (lane == 0)
			range = __ldg(&tile_ranges[idx]);
		range.x = __shfl_sync(~0u, range.x, 0);
		range.y = __shfl_sync(~0u, range.y, 0);

		float T[THREAD_Y][THREAD_X];
		float3 C[THREAD_Y][THREAD_X];
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				T[i][j] = 1.0f;
				C[i][j] = { 0.0f, 0.0f, 0.0f };
			}

		const int total = range.y - range.x;
		if (total > 0)
		{
			const int b0 = min(BATCH, total);
			if (lane < b0)
			{
				const int gid = __ldg(&point_list[range.x + lane]);
				cp_async_ca_shared_global_8(&s_xy[0][lane], &points_xy[gid]);
				cp_async_ca_shared_global_16(&s_rgb[0][lane], &rgb_depth[gid]);
				cp_async_ca_shared_global_16(&s_con[0][lane], &conic_opacity[gid]);
			}
			cp_async_commit_group();
			cp_async_wait_all();
			__syncwarp();

			int cur = 0;
			int pos = range.x;

			while (pos < range.y)
			{
				const int bsize = min(BATCH, range.y - pos);
				const int nxt = 1 - cur;
				const int npos = pos + BATCH;
				const int nsize = max(0, min(BATCH, range.y - npos));

				if (nsize > 0)
				{
					if (lane < nsize)
					{
						const int gid = __ldg(&point_list[npos + lane]);
						cp_async_ca_shared_global_8(&s_xy[nxt][lane], &points_xy[gid]);
						cp_async_ca_shared_global_16(&s_rgb[nxt][lane], &rgb_depth[gid]);
						cp_async_ca_shared_global_16(&s_con[nxt][lane], &conic_opacity[gid]);
					}
					cp_async_commit_group();
				}

				bool early_exit = false;
				for (int g = 0; g < bsize; g++)
				{
					const float2 xy = s_xy[cur][g];
					const float4 con_o = s_con[cur][g];
					const float4 rgbd = s_rgb[cur][g];
					const float3 rgb = { rgbd.x, rgbd.y, rgbd.z };

					pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
						C, T, tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base, xy, con_o, rgb);

					if ((g & 7) == 7)
					{
						bool sat = true;
#pragma unroll
						for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
							for (int j = 0; j < THREAD_X; j++)
								sat = sat && T[i][j] < 0.0001f;
						if (__all_sync(~0u, sat))
						{
							early_exit = true;
							break;
						}
					}
				}

				if (nsize > 0)
				{
					cp_async_wait_all();
					__syncwarp();
				}

				if (early_exit) break;
				pos += BATCH;
				cur = nxt;
			}
		}

#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int px = tile_origin_x + thread_col_base + j;
				int py = tile_origin_y + thread_row_base + i;
				if (total > 0)
					write_color(out_color, bg_color, { px, py }, width, height, C[i][j], T[i][j]);
				else if (px < width && py < height)
				{
					int pix_id = width * py + px;
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}

		if (lane == 0)
			idx = (int)atomicAdd(&g_tile_counter, 1) + (int)gridDim.x;
		idx = __shfl_sync(~0u, idx, 0);
	}
}

// ============================================================================
// Dynamic Thresholding variant of persistent_lite.
// Inner loop skips ex2 + weight/color/T updates for Gaussians whose
// power (log2-space alpha) is below log2(1/255) = -7.994353375.
// ============================================================================
template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA_preranges_smem_persistent_lite_dt(
	const uint32_t* __restrict__ point_list,
	int width, int height,
	int x_blocks, int total_tiles,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int2* __restrict__ tile_ranges,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BATCH = 32;

	__shared__ float2 s_xy[2][BATCH];
	__shared__ float4 s_rgb[2][BATCH];
	__shared__ float4 s_con[2][BATCH];

	const int lane = threadIdx.y * blockDim.x + threadIdx.x;
	const int thread_col_base = (int)threadIdx.x * THREAD_X;
	const int thread_row_base = (int)threadIdx.y * THREAD_Y;

	int idx = (int)blockIdx.x;

	while (idx < total_tiles)
	{
		const int tile_col = idx % x_blocks;
		const int tile_row = idx / x_blocks;
		const int tile_origin_x = tile_col * BLOCK_X;
		const int tile_origin_y = tile_row * BLOCK_Y;

		int2 range = make_int2(0, 0);
		if (lane == 0)
			range = __ldg(&tile_ranges[idx]);
		range.x = __shfl_sync(~0u, range.x, 0);
		range.y = __shfl_sync(~0u, range.y, 0);

		float T[THREAD_Y][THREAD_X];
		float3 C[THREAD_Y][THREAD_X];
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				T[i][j] = 1.0f;
				C[i][j] = { 0.0f, 0.0f, 0.0f };
			}

		const int total = range.y - range.x;
		if (total > 0)
		{
			const int b0 = min(BATCH, total);
			if (lane < b0)
			{
				const int gid = __ldg(&point_list[range.x + lane]);
				cp_async_ca_shared_global_8(&s_xy[0][lane], &points_xy[gid]);
				cp_async_ca_shared_global_16(&s_rgb[0][lane], &rgb_depth[gid]);
				cp_async_ca_shared_global_16(&s_con[0][lane], &conic_opacity[gid]);
			}
			cp_async_commit_group();
			cp_async_wait_all();
			__syncwarp();

			int cur = 0;
			int pos = range.x;

			while (pos < range.y)
			{
				const int bsize = min(BATCH, range.y - pos);
				const int nxt = 1 - cur;
				const int npos = pos + BATCH;
				const int nsize = max(0, min(BATCH, range.y - npos));

				if (nsize > 0)
				{
					if (lane < nsize)
					{
						const int gid = __ldg(&point_list[npos + lane]);
						cp_async_ca_shared_global_8(&s_xy[nxt][lane], &points_xy[gid]);
						cp_async_ca_shared_global_16(&s_rgb[nxt][lane], &rgb_depth[gid]);
						cp_async_ca_shared_global_16(&s_con[nxt][lane], &conic_opacity[gid]);
					}
					cp_async_commit_group();
				}

				bool early_exit = false;
				for (int g = 0; g < bsize; g++)
				{
					const float2 xy = s_xy[cur][g];
					const float4 con_o = s_con[cur][g];
					const float4 rgbd = s_rgb[cur][g];
					const float3 rgb = { rgbd.x, rgbd.y, rgbd.z };

					pixel_shader_axis_shared_16x16_dt<THREAD_X, THREAD_Y>(
						C, T, tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base, xy, con_o, rgb);

					if ((g & 7) == 7)
					{
						bool sat = true;
#pragma unroll
						for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
							for (int j = 0; j < THREAD_X; j++)
								sat = sat && T[i][j] < 0.0001f;
						if (__all_sync(~0u, sat))
						{
							early_exit = true;
							break;
						}
					}
				}

				if (nsize > 0)
				{
					cp_async_wait_all();
					__syncwarp();
				}

				if (early_exit) break;
				pos += BATCH;
				cur = nxt;
			}
		}

#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int px = tile_origin_x + thread_col_base + j;
				int py = tile_origin_y + thread_row_base + i;
				if (total > 0)
					write_color(out_color, bg_color, { px, py }, width, height, C[i][j], T[i][j]);
				else if (px < width && py < height)
				{
					int pix_id = width * py + px;
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}

		if (lane == 0)
			idx = (int)atomicAdd(&g_tile_counter, 1) + (int)gridDim.x;
		idx = __shfl_sync(~0u, idx, 0);
	}
}

// ============================================================================
// E2E-fused persistent_lite: inline binary search per tile (no precomputed tile_ranges).
// Eliminates the separate computeTileRanges kernel from the pipeline.
// Functionally identical to renderCUDA_preranges_smem_persistent_lite when tile_ranges
// equals find_tile_range output.
// ============================================================================
template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA_preranges_smem_persistent_lite_fused(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	int x_blocks, int total_tiles,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BATCH = 32;

	__shared__ float2 s_xy[2][BATCH];
	__shared__ float4 s_rgb[2][BATCH];
	__shared__ float4 s_con[2][BATCH];

	const int lane = threadIdx.y * blockDim.x + threadIdx.x;
	const int thread_col_base = (int)threadIdx.x * THREAD_X;
	const int thread_row_base = (int)threadIdx.y * THREAD_Y;

	int idx = (int)blockIdx.x;

	while (idx < total_tiles)
	{
		const int tile_col = idx % x_blocks;
		const int tile_row = idx / x_blocks;
		const int tile_origin_x = tile_col * BLOCK_X;
		const int tile_origin_y = tile_row * BLOCK_Y;

		// Inline fused binary search using WARP-COOPERATIVE 32-way partitioning.
		// All 32 lanes participate => log32(N) rounds instead of log2(N),
		// so the search cost is ~5x lower per tile vs lane-0-only.
		int2 range = find_tile_range_warp(sorted_keys, num_rendered, (uint32_t)idx,
			(unsigned)lane);

		float T[THREAD_Y][THREAD_X];
		float3 C[THREAD_Y][THREAD_X];
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				T[i][j] = 1.0f;
				C[i][j] = { 0.0f, 0.0f, 0.0f };
			}

		const int total = range.y - range.x;
		if (total > 0)
		{
			const int b0 = min(BATCH, total);
			if (lane < b0)
			{
				const int gid = __ldg(&point_list[range.x + lane]);
				cp_async_ca_shared_global_8(&s_xy[0][lane], &points_xy[gid]);
				cp_async_ca_shared_global_16(&s_rgb[0][lane], &rgb_depth[gid]);
				cp_async_ca_shared_global_16(&s_con[0][lane], &conic_opacity[gid]);
			}
			cp_async_commit_group();
			cp_async_wait_all();
			__syncwarp();

			int cur = 0;
			int pos = range.x;

			while (pos < range.y)
			{
				const int bsize = min(BATCH, range.y - pos);
				const int nxt = 1 - cur;
				const int npos = pos + BATCH;
				const int nsize = max(0, min(BATCH, range.y - npos));

				if (nsize > 0)
				{
					if (lane < nsize)
					{
						const int gid = __ldg(&point_list[npos + lane]);
						cp_async_ca_shared_global_8(&s_xy[nxt][lane], &points_xy[gid]);
						cp_async_ca_shared_global_16(&s_rgb[nxt][lane], &rgb_depth[gid]);
						cp_async_ca_shared_global_16(&s_con[nxt][lane], &conic_opacity[gid]);
					}
					cp_async_commit_group();
				}

				bool early_exit = false;
				for (int g = 0; g < bsize; g++)
				{
					const float2 xy = s_xy[cur][g];
					const float4 con_o = s_con[cur][g];
					const float4 rgbd = s_rgb[cur][g];
					const float3 rgb = { rgbd.x, rgbd.y, rgbd.z };

					pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
						C, T, tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base, xy, con_o, rgb);

					if ((g & 15) == 15)
					{
						bool sat = true;
#pragma unroll
						for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
							for (int j = 0; j < THREAD_X; j++)
								sat = sat && T[i][j] < 0.0001f;
						if (__all_sync(~0u, sat))
						{
							early_exit = true;
							break;
						}
					}
				}

				if (nsize > 0)
				{
					cp_async_wait_all();
					__syncwarp();
				}

				if (early_exit) break;
				pos += BATCH;
				cur = nxt;
			}
		}

#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int px = tile_origin_x + thread_col_base + j;
				int py = tile_origin_y + thread_row_base + i;
				if (total > 0)
					write_color(out_color, bg_color, { px, py }, width, height, C[i][j], T[i][j]);
				else if (px < width && py < height)
				{
					int pix_id = width * py + px;
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}

		if (lane == 0)
			idx = (int)atomicAdd(&g_tile_counter, 1) + (int)gridDim.x;
		idx = __shfl_sync(~0u, idx, 0);
	}
}

template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y, bool USE_UNROLL2 = false, bool HAS_TOPK_PACKED = false>
__global__ __launch_bounds__(32) void renderCUDA(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	uint32_t tile_id = blockIdx.y * x_blocks + blockIdx.x;
	int2 range = find_tile_range(sorted_keys, num_rendered, tile_id);
	// 找到”当前 block 对应的 tile” ，再取出这个 tile 该处理的 Gaussian 范围 [start, end)
	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	int tile_origin_x = (int)blockIdx.x * BLOCK_X;
	int tile_origin_y = (int)blockIdx.y * BLOCK_Y;
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	const void* data = info.data[lane];
	[[maybe_unused]] const void* topk_data_ptr = info.topk_data[lane];
	int lg2_scale = info.lg2_scale[lane];
	[[maybe_unused]] bool is_feature_lane = lane_uses_gaussian_features(lane);

	int pix_x[THREAD_X];
	float pixf_x[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
	{
		pix_x[j] = tile_origin_x + thread_col_base + j;
		pixf_x[j] = (float)pix_x[j];
	}

	int pix_y[THREAD_Y];
	float pixf_y[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
		pix_y[i] = tile_origin_y + thread_row_base + i;
		pixf_y[i] = (float)pix_y[i];
	}

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			T[i][j] = 1.0f;
		}
	}

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			C[i][j] = { 0.0f, 0.0f, 0.0f };
		}
	}
	//准备好了这个tile处理的【4*2】个像素，接下来就要计算这些像素的数值啦！

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
		{
			offset = point_id + 2;
		}
		else if (lane == 4)
		{
			offset = point_id + 3;
		}
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
		{
			offset = point_list[point_id + 0];
		}
		else if (point_id + 1 < range.y)
		{
			offset = point_list[point_id + 1];
		}
		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
		{
			load_enable = load_enable && point_id + 2 < range.y;
		}
		else if (lane == 4)
		{
			load_enable = load_enable && point_id + 3 < range.y;
		}
		else if ((lane & 4) == 0)
		{
			load_enable = load_enable && point_id + 0 < range.y;
		}
		else
		{
			load_enable = load_enable && point_id + 1 < range.y;
		}
		if (load_enable)
		{
			if constexpr (HAS_TOPK_PACKED)
				buf = load_lane_value_topk_packed_fast(data, topk_data_ptr, lg2_scale, offset, is_feature_lane);
			else
				buf = load_lane_value(data, lg2_scale, offset); // 0: point_list[point_id + 2], 4: point_list[point_id + 3], 8: features[point_list[point_id + 0]], 12: features[point_list[point_id + 1]]
		}

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

#ifdef _DEBUG
			if (lane == 0)
			{
				printf("point_id = %d\n", point_id);
			}
#endif
			float ldg_buf;
			if (load_enable)
			{
				if constexpr (HAS_TOPK_PACKED)
					ldg_buf = load_lane_value_topk_packed_fast(data, topk_data_ptr, lg2_scale, offset, is_feature_lane);
				else
					ldg_buf = load_lane_value(data, lg2_scale, offset); // 0: point_list[point_id + 4], 4: point_list[point_id + 5], 8: features[point_list[point_id + 2]], 12: features[point_list[point_id + 3]]
#ifdef _DEBUG
				if (lane == 0 && __float_as_int(ldg_buf) != point_list[point_id + 4])
				{
					printf("error1\n");
				}
				else if (lane == 4 && __float_as_int(ldg_buf) != point_list[point_id + 5])
				{
					printf("error2\n");
				}
				else if (lane == 8 && ldg_buf != points_xy[point_list[point_id + 2]].x)
				{
					printf("error3\n");
				}
				else if (lane == 12 && ldg_buf != points_xy[point_list[point_id + 3]].x)
				{
					printf("error4\n");
				}
#endif
			}

			get_gaussian_features(xy, rgb, con_o, buf, 0);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 0]].x)
			{
				printf("error5\n");
			}
#endif

			if constexpr (BLOCK_X == 16 && BLOCK_Y == 16)
			{
				if constexpr (USE_UNROLL2)
				{
					pixel_shader_axis_shared_16x16_unroll2<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
				else
				{
					pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
			}
			else
			{
	#pragma unroll
				for (int i = 0; i < THREAD_Y; i++)
				{
	#pragma unroll
					for (int j = 0; j < THREAD_X; j++)
					{
						pixel_shader(C[i][j], T[i][j], pixf_x[j], pixf_y[i], xy, con_o, rgb);
					}
				}
			}

			get_gaussian_features(xy, rgb, con_o, buf, 4);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 1]].x)
			{
				printf("error6\n");
			}
#endif

			if constexpr (BLOCK_X == 16 && BLOCK_Y == 16)
			{
				if constexpr (USE_UNROLL2)
				{
					pixel_shader_axis_shared_16x16_unroll2<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
				else
				{
					pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
			}
			else
			{
	#pragma unroll
				for (int i = 0; i < THREAD_Y; i++)
				{
	#pragma unroll
					for (int j = 0; j < THREAD_X; j++)
					{
						pixel_shader(C[i][j], T[i][j], pixf_x[j], pixf_y[i], xy, con_o, rgb);
					}
				}
			}

			done = true;
	#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
	#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}

			point_id += 2;
			buf = ldg_buf;
		}
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

			if (lane == 0)
			{
				load_enable = load_enable && point_id + 4 < range.y;
			}
			else if (lane == 4)
			{
				load_enable = load_enable && point_id + 5 < range.y;
			}
			else if ((lane & 4) == 0)
			{
				load_enable = load_enable && point_id + 2 < range.y;
			}
			else
			{
				load_enable = load_enable && point_id + 3 < range.y;
			}

#ifdef _DEBUG
			if (lane == 0)
			{
				printf("point_id = %d\n", point_id);
			}
#endif
			float ldg_buf;
			if (load_enable)
			{
				if constexpr (HAS_TOPK_PACKED)
					ldg_buf = load_lane_value_topk_packed_fast(data, topk_data_ptr, lg2_scale, offset, is_feature_lane);
				else
					ldg_buf = load_lane_value(data, lg2_scale, offset); // 0: point_list[point_id + 4], 4: point_list[point_id + 5], 8: features[point_list[point_id + 2]], 12: features[point_list[point_id + 3]]
#ifdef _DEBUG
				if (lane == 0 && __float_as_int(ldg_buf) != point_list[point_id + 4])
				{
					printf("error1\n");
				}
				else if (lane == 4 && __float_as_int(ldg_buf) != point_list[point_id + 5])
				{
					printf("error2\n");
				}
				else if (lane == 8 && ldg_buf != points_xy[point_list[point_id + 2]].x)
				{
					printf("error3\n");
				}
				else if (lane == 12 && ldg_buf != points_xy[point_list[point_id + 3]].x)
				{
					printf("error4\n");
				}
#endif
			}

			get_gaussian_features(xy, rgb, con_o, buf, 0);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 0]].x)
			{
				printf("error5\n");
			}
#endif

			if constexpr (BLOCK_X == 16 && BLOCK_Y == 16)
			{
				if constexpr (USE_UNROLL2)
				{
					pixel_shader_axis_shared_16x16_unroll2<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
				else
				{
					pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
			}
			else
			{
#pragma unroll
				for (int i = 0; i < THREAD_Y; i++)
				{
#pragma unroll
					for (int j = 0; j < THREAD_X; j++)
					{
						pixel_shader(C[i][j], T[i][j], pixf_x[j], pixf_y[i], xy, con_o, rgb);
					}
				}
			}

			if (point_id + 1 >= range.y)
				break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 1]].x)
			{
				printf("error6\n");
			}
#endif

			if constexpr (BLOCK_X == 16 && BLOCK_Y == 16)
			{
				if constexpr (USE_UNROLL2)
				{
					pixel_shader_axis_shared_16x16_unroll2<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
				else
				{
					pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
			}
			else
			{
#pragma unroll
				for (int i = 0; i < THREAD_Y; i++)
				{
#pragma unroll
					for (int j = 0; j < THREAD_X; j++)
					{
						pixel_shader(C[i][j], T[i][j], pixf_x[j], pixf_y[i], xy, con_o, rgb);
					}
				}
			}

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
			}
		}
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix_x[j] < width && pix_y[i] < height)
				{
					int pix_id = width * pix_y[i] + pix_x[j];
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
		}
	}
}
// Persistent kernel: one thread block per SM, multiple warps per block.
// Each warp independently processes tiles; all warps share the same smem.
// threadIdx.z selects warp, (threadIdx.y, threadIdx.x) gives lane 0-31.
template<int THREAD_X, int THREAD_Y, int WARPS_PER_BLOCK = 4, bool USE_UNROLL2 = false>
__global__ POLYSPLAT_TOPK_SMEM_PERSISTENT_LAUNCH_BOUNDS POLYSPLAT_TOPK_SMEM_PERSISTENT_MAXNREG_ATTR
void renderCUDA16x16TopKSmem(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks, int total_tiles,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const float2* __restrict__ topk_xy,
	const float4* __restrict__ topk_rgb_depth,
	const float4* __restrict__ topk_conic,
	int num_topk,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;
	static_assert(WARPS_PER_BLOCK == POLYSPLAT_TOPK_SMEM_PERSISTENT_WPB,
		"Persistent top-k kernel launch bounds expect the configured WPB.");

	extern __shared__ float4 topk_async_smem_storage[];
	topk_async_smem_buffers smem = make_topk_async_smem_buffers(topk_async_smem_storage, num_topk);

	int warp_id = (int)threadIdx.z;
	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x; // 0-31 within each warp
	int thread_id = warp_id * 32 + lane; // global thread id within block
	constexpr int TOTAL_THREADS = 32 * WARPS_PER_BLOCK;

	if (num_topk > 0)
	{
		stage_topk_gaussians_to_async_smem(
			thread_id, TOTAL_THREADS, num_topk,
			topk_xy, topk_rgb_depth, topk_conic, smem);
	}

	// Per-warp invariants
	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;
	bool is_feature_lane = lane_uses_gaussian_features(lane);
	topk_async_smem_lane_view smem_lane_view = get_topk_async_smem_lane_view(lane, smem);
	const float* smem_lane_ptr = smem_lane_view.base;
	int smem_lane_lg2_stride = smem_lane_view.lg2_stride;

	// Overlap top-k staging with the first tile fetch and binary search.
	int tile_id = 0;
	if (lane == 0)
	{
		tile_id = (int)atomicAdd(&g_tile_counter, 1);
	}
	tile_id = __shfl_sync(~0, tile_id, 0);

	int2 range = make_int2(0, 0);
	if (tile_id < total_tiles && lane == 0)
	{
		range = find_tile_range(sorted_keys, num_rendered, (uint32_t)tile_id);
	}
	range.x = __shfl_sync(~0, range.x, 0);
	range.y = __shfl_sync(~0, range.y, 0);

	if (num_topk > 0)
	{
		cp_async_wait_all();
	}
	__syncthreads();

	// Persistent tile loop: each warp atomically grabs the next tile.
	// Lane 0 does the atomic; result is broadcast via __shfl_sync.
	while (tile_id < total_tiles)
	{
		int tile_col = tile_id % x_blocks;
		int tile_row = tile_id / x_blocks;
		int tile_origin_x = tile_col * BLOCK_X;
		int tile_origin_y = tile_row * BLOCK_Y;

		int pix_x[THREAD_X];
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			pix_x[j] = tile_origin_x + thread_col_base + j;

		int pix_y[THREAD_Y];
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
			pix_y[i] = tile_origin_y + thread_row_base + i;

		float T[THREAD_Y][THREAD_X];
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				T[i][j] = 1.0f;
		}

		float3 C[THREAD_Y][THREAD_X];
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				C[i][j] = { 0.0f, 0.0f, 0.0f };
		}

		int point_id = range.x;
		if (point_id < range.y)
		{
			int offset;
			float2 xy;
			float3 rgb;
			float4 con_o;
			if (lane == 0)
			{
				offset = point_id + 2;
			}
			else if (lane == 4)
			{
				offset = point_id + 3;
			}
			else if ((lane & 4) == 0 && point_id + 0 < range.y)
			{
				offset = point_list[point_id + 0];
			}
			else if (point_id + 1 < range.y)
			{
				offset = point_list[point_id + 1];
			}
			float buf;
			bool load_enable = data != nullptr;
			if (lane == 0)
			{
				load_enable = load_enable && point_id + 2 < range.y;
			}
			else if (lane == 4)
			{
				load_enable = load_enable && point_id + 3 < range.y;
			}
			else if ((lane & 4) == 0)
			{
				load_enable = load_enable && point_id + 0 < range.y;
			}
			else
			{
				load_enable = load_enable && point_id + 1 < range.y;
			}
			if (load_enable)
			{
				buf = load_lane_value_topk_async_smem_fast(
					data, lg2_scale, offset, is_feature_lane, num_topk, smem_lane_ptr, smem_lane_lg2_stride);
			}

			load_enable = data != nullptr;

			bool done = false;
			while (__any_sync(~0, point_id + 5 < range.y && !done))
			{
				offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
				if (lane == 0)
				{
					offset = point_id + 4;
				}
				if (lane == 4)
				{
					offset = point_id + 5;
				}

				float ldg_buf;
				if (load_enable)
				{
					ldg_buf = load_lane_value_topk_async_smem_fast(
						data, lg2_scale, offset, is_feature_lane, num_topk, smem_lane_ptr, smem_lane_lg2_stride);
				}

				get_gaussian_features(xy, rgb, con_o, buf, 0);
				if constexpr (USE_UNROLL2)
				{
					pixel_shader_axis_shared_16x16_unroll2<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
				else
				{
					pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}

				get_gaussian_features(xy, rgb, con_o, buf, 4);
				if constexpr (USE_UNROLL2)
				{
					pixel_shader_axis_shared_16x16_unroll2<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
				else
				{
					pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}

				done = true;
#pragma unroll
				for (int i = 0; i < THREAD_Y; i++)
				{
#pragma unroll
					for (int j = 0; j < THREAD_X; j++)
					{
						done = done && T[i][j] < 0.0001f;
					}
				}

				point_id += 2;
				buf = ldg_buf;
			}
			while (__any_sync(~0, point_id < range.y && !done))
			{
				offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
				if (lane == 0)
				{
					offset = point_id + 4;
				}
				if (lane == 4)
				{
					offset = point_id + 5;
				}

				if (lane == 0)
				{
					load_enable = load_enable && point_id + 4 < range.y;
				}
				else if (lane == 4)
				{
					load_enable = load_enable && point_id + 5 < range.y;
				}
				else if ((lane & 4) == 0)
				{
					load_enable = load_enable && point_id + 2 < range.y;
				}
				else
				{
					load_enable = load_enable && point_id + 3 < range.y;
				}

				float ldg_buf;
				if (load_enable)
				{
					ldg_buf = load_lane_value_topk_async_smem_fast(
						data, lg2_scale, offset, is_feature_lane, num_topk, smem_lane_ptr, smem_lane_lg2_stride);
				}

				get_gaussian_features(xy, rgb, con_o, buf, 0);
				if constexpr (USE_UNROLL2)
				{
					pixel_shader_axis_shared_16x16_unroll2<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
				else
				{
					pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}

				if (point_id + 1 >= range.y)
					break;

				get_gaussian_features(xy, rgb, con_o, buf, 4);
				if constexpr (USE_UNROLL2)
				{
					pixel_shader_axis_shared_16x16_unroll2<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}
				else
				{
					pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
						C, T,
						tile_origin_x, tile_origin_y,
						thread_col_base, thread_row_base,
						xy, con_o, rgb);
				}

				done = true;
#pragma unroll
				for (int i = 0; i < THREAD_Y; i++)
				{
#pragma unroll
					for (int j = 0; j < THREAD_X; j++)
					{
						done = done && T[i][j] < 0.0001f;
					}
				}
				point_id += 2;
				buf = ldg_buf;
			}
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
				}
			}
		}
		else
		{
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					if (pix_x[j] < width && pix_y[i] < height)
					{
						int pix_id = width * pix_y[i] + pix_x[j];
						out_color[pix_id].x = encode(bg_color.x);
						out_color[pix_id].y = encode(bg_color.y);
						out_color[pix_id].z = encode(bg_color.z);
					}
				}
			}
		}

		if (lane == 0)
		{
			tile_id = (int)atomicAdd(&g_tile_counter, 1);
		}
		tile_id = __shfl_sync(~0, tile_id, 0);

		range = make_int2(0, 0);
		if (tile_id < total_tiles && lane == 0)
		{
			range = find_tile_range(sorted_keys, num_rendered, (uint32_t)tile_id);
		}
		range.x = __shfl_sync(~0, range.x, 0);
		range.y = __shfl_sync(~0, range.y, 0);
	}
}

// V2 Persistent kernel: pure work-stealing, NO top-k overhead.
// Same inner loop as default kernel but with persistent tile dispatch.
// Each warp independently grabs tiles via atomicAdd — dynamic load balancing.
// Key: 1 warp per block + __launch_bounds__(32) — same reg budget as default.
// The tile processing is in a __noinline__ function to prevent the compiler from
// keeping T/C accumulator registers live across the tile loop back-edge.

template<int THREAD_X, int THREAD_Y>
__noinline__ __device__ void renderTileV2(
	int tile_id, int x_blocks,
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	const void* data, int lg2_scale,
	int thread_col_base, int thread_row_base,
	float3 bg_color,
	uchar3* __restrict__ out_color,
	int2 range)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;
	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;

	int tile_col = tile_id % x_blocks;
	int tile_row = tile_id / x_blocks;
	int tile_origin_x = tile_col * BLOCK_X;
	int tile_origin_y = tile_row * BLOCK_Y;

	int pix_x[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
		pix_x[j] = tile_origin_x + thread_col_base + j;

	int pix_y[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
		pix_y[i] = tile_origin_y + thread_row_base + i;

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			T[i][j] = 1.0f;
	}

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			C[i][j] = { 0.0f, 0.0f, 0.0f };
	}

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
			offset = point_id + 2;
		else if (lane == 4)
			offset = point_id + 3;
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
			offset = point_list[point_id + 0];
		else if (point_id + 1 < range.y)
			offset = point_list[point_id + 1];

		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
			load_enable = load_enable && point_id + 2 < range.y;
		else if (lane == 4)
			load_enable = load_enable && point_id + 3 < range.y;
		else if ((lane & 4) == 0)
			load_enable = load_enable && point_id + 0 < range.y;
		else
			load_enable = load_enable && point_id + 1 < range.y;

		if (load_enable)
			buf = load_lane_value(data, lg2_scale, offset);

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
				offset = point_id + 4;
			if (lane == 4)
				offset = point_id + 5;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			}

			point_id += 2;
			buf = ldg_buf;
		}
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
				offset = point_id + 4;
			if (lane == 4)
				offset = point_id + 5;

			if (lane == 0)
				load_enable = load_enable && point_id + 4 < range.y;
			else if (lane == 4)
				load_enable = load_enable && point_id + 5 < range.y;
			else if ((lane & 4) == 0)
				load_enable = load_enable && point_id + 2 < range.y;
			else
				load_enable = load_enable && point_id + 3 < range.y;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			if (point_id + 1 >= range.y)
				break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			}
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
		}
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix_x[j] < width && pix_y[i] < height)
				{
					int pix_id = width * pix_y[i] + pix_x[j];
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
		}
	}
}

template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32)
void renderCUDA16x16TopKSmemV2(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks, int total_tiles,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;

	// Per-warp invariants
	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	// First tile fetch
	int tile_id = 0;
	if (lane == 0)
		tile_id = (int)atomicAdd(&g_tile_counter, 1);
	tile_id = __shfl_sync(~0, tile_id, 0);

	int2 range = make_int2(0, 0);
	if (tile_id < total_tiles && lane == 0)
		range = find_tile_range(sorted_keys, num_rendered, (uint32_t)tile_id);
	range.x = __shfl_sync(~0, range.x, 0);
	range.y = __shfl_sync(~0, range.y, 0);

	// Persistent tile loop
	while (tile_id < total_tiles)
	{
		renderTileV2<THREAD_X, THREAD_Y>(
			tile_id, x_blocks,
			sorted_keys, num_rendered, point_list,
			width, height,
			data, lg2_scale,
			thread_col_base, thread_row_base,
			bg_color, out_color, range);

		// Next tile
		if (lane == 0)
			tile_id = (int)atomicAdd(&g_tile_counter, 1);
		tile_id = __shfl_sync(~0, tile_id, 0);

		range = make_int2(0, 0);
		if (tile_id < total_tiles && lane == 0)
			range = find_tile_range(sorted_keys, num_rendered, (uint32_t)tile_id);
		range.x = __shfl_sync(~0, range.x, 0);
		range.y = __shfl_sync(~0, range.y, 0);
	}
}

// V3 tile function: same logic as renderTileV2 but in a dedicated copy so the
// compiler can optimize register allocation under the V3 kernel's tighter
// __launch_bounds__(32, 32) constraint independently.
template<int THREAD_X, int THREAD_Y>
__noinline__ __device__ void renderTileV3(
	int tile_id, int x_blocks,
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	const void* data, int lg2_scale,
	int thread_col_base, int thread_row_base,
	float3 bg_color,
	uchar3* __restrict__ out_color,
	int2 range)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;
	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;

	int tile_col = tile_id % x_blocks;
	int tile_row = tile_id / x_blocks;
	int tile_origin_x = tile_col * BLOCK_X;
	int tile_origin_y = tile_row * BLOCK_Y;

	int pix_x[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
		pix_x[j] = tile_origin_x + thread_col_base + j;

	int pix_y[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
		pix_y[i] = tile_origin_y + thread_row_base + i;

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			T[i][j] = 1.0f;
	}

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			C[i][j] = { 0.0f, 0.0f, 0.0f };
	}

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
			offset = point_id + 2;
		else if (lane == 4)
			offset = point_id + 3;
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
			offset = point_list[point_id + 0];
		else if (point_id + 1 < range.y)
			offset = point_list[point_id + 1];

		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
			load_enable = load_enable && point_id + 2 < range.y;
		else if (lane == 4)
			load_enable = load_enable && point_id + 3 < range.y;
		else if ((lane & 4) == 0)
			load_enable = load_enable && point_id + 0 < range.y;
		else
			load_enable = load_enable && point_id + 1 < range.y;

		if (load_enable)
			buf = load_lane_value(data, lg2_scale, offset);

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
				offset = point_id + 4;
			if (lane == 4)
				offset = point_id + 5;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			}

			point_id += 2;
			buf = ldg_buf;
		}
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
				offset = point_id + 4;
			if (lane == 4)
				offset = point_id + 5;

			if (lane == 0)
				load_enable = load_enable && point_id + 4 < range.y;
			else if (lane == 4)
				load_enable = load_enable && point_id + 5 < range.y;
			else if ((lane & 4) == 0)
				load_enable = load_enable && point_id + 2 < range.y;
			else
				load_enable = load_enable && point_id + 3 < range.y;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			if (point_id + 1 >= range.y)
				break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			}
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
		}
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix_x[j] < width && pix_y[i] < height)
				{
					int pix_id = width * pix_y[i] + pix_x[j];
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
		}
	}
}

// V3 kernel: persistent work-stealing with precomputed tile ranges.
// Uses __launch_bounds__(32, 28) + smem wrapper + precomputed ranges to minimize
// wrapper overhead and maximize occupancy.
template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32, 28)
void renderCUDA16x16PersistentV3(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks, int total_tiles,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int2* __restrict__ tile_ranges,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;

	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	// First tile fetch
	int tile_id = 0;
	if (lane == 0)
		tile_id = (int)atomicAdd(&g_tile_counter, 1);
	tile_id = __shfl_sync(~0, tile_id, 0);

	int2 range = make_int2(0, 0);
	if (tile_id < total_tiles && lane == 0)
		range = __ldg(&tile_ranges[tile_id]);
	range.x = __shfl_sync(~0, range.x, 0);
	range.y = __shfl_sync(~0, range.y, 0);

	// Persistent tile loop
	while (tile_id < total_tiles)
	{
		renderTileV3<THREAD_X, THREAD_Y>(
			tile_id, x_blocks,
			sorted_keys, num_rendered, point_list,
			width, height,
			data, lg2_scale,
			thread_col_base, thread_row_base,
			bg_color, out_color, range);

		// Claim next tile
		if (lane == 0)
			tile_id = (int)atomicAdd(&g_tile_counter, 1);
		tile_id = __shfl_sync(~0, tile_id, 0);

		range = make_int2(0, 0);
		if (tile_id < total_tiles && lane == 0)
			range = __ldg(&tile_ranges[tile_id]);
		range.x = __shfl_sync(~0, range.x, 0);
		range.y = __shfl_sync(~0, range.y, 0);
	}
}

// Kernel to precompute tile ranges from sorted keys
__global__ void computeTileRanges(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	int total_tiles,
	int2* __restrict__ tile_ranges)
{
	int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (tile_id < total_tiles)
	{
		tile_ranges[tile_id] = find_tile_range(sorted_keys, num_rendered, (uint32_t)tile_id);
	}
}

// Boundary-scan tile range computation: O(num_rendered) with sequential L2 access.
// Each thread checks if sorted_keys[idx] crosses a tile boundary.
// Requires tile_ranges to be pre-zeroed (tiles with no gaussians get {0,0}).
__global__ void computeTileRangesScan(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	int2* __restrict__ tile_ranges)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_rendered)
		return;

	uint32_t tile_cur = (uint32_t)(__ldg(&sorted_keys[idx]) >> 32);

	// First element: start of its tile
	if (idx == 0)
	{
		tile_ranges[tile_cur].x = 0;
	}
	else
	{
		uint32_t tile_prev = (uint32_t)(__ldg(&sorted_keys[idx - 1]) >> 32);
		if (tile_cur != tile_prev)
		{
			tile_ranges[tile_prev].y = idx;
			tile_ranges[tile_cur].x = idx;
		}
	}

	// Last element: end of its tile
	if (idx == num_rendered - 1)
	{
		tile_ranges[tile_cur].y = num_rendered;
	}
}

// Non-persistent topk smem kernel: standard 1-warp-per-tile launch (HW scheduling)
// with topk gaussians cached in shared memory. Avoids the persistent kernel's
// register pressure (1024 threads → 64 regs/thread) by using __launch_bounds__(32)
// which gives each thread up to 256 registers.
template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA16x16TopKSmemNonPersistent(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const float2* __restrict__ topk_xy,
	const float4* __restrict__ topk_rgb_depth,
	const float4* __restrict__ topk_conic,
	int num_topk,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;

	extern __shared__ float topk_smem[];
	topk_smem_buffers smem = make_topk_smem_buffers(topk_smem, num_topk);

	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x; // 0-31

	// Cooperatively load topk gaussians into smem using the single warp.
	for (int i = lane; i < num_topk; i += POLYSPLAT_WARP_SIZE)
	{
		float2 xy = __ldg(&topk_xy[i]);
		smem.xy_x[i] = xy.x;
		smem.xy_y[i] = xy.y;

		float4 rd = __ldg(&topk_rgb_depth[i]);
		smem.rgb_r[i] = rd.x;
		smem.rgb_g[i] = rd.y;
		smem.rgb_b[i] = rd.z;

		float4 cn = __ldg(&topk_conic[i]);
		smem.con_x[i] = cn.x;
		smem.con_y[i] = cn.y;
		smem.con_z[i] = cn.z;
		smem.con_w[i] = cn.w;
	}
	__syncwarp(); // single warp, syncwarp is sufficient

	uint32_t tile_id = blockIdx.y * x_blocks + blockIdx.x;
	int2 range;
	if (lane == 0)
	{
		range = find_tile_range(sorted_keys, num_rendered, tile_id);
	}
	range.x = __shfl_sync(~0, range.x, 0);
	range.y = __shfl_sync(~0, range.y, 0);
	int tile_origin_x = (int)blockIdx.x * BLOCK_X;
	int tile_origin_y = (int)blockIdx.y * BLOCK_Y;
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];
	bool is_feature_lane = lane_uses_gaussian_features(lane);
	const float* smem_lane_ptr = get_topk_smem_lane_ptr(lane, smem);

	int pix_x[THREAD_X];
	float pixf_x[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
	{
		pix_x[j] = tile_origin_x + thread_col_base + j;
		pixf_x[j] = (float)pix_x[j];
	}

	int pix_y[THREAD_Y];
	float pixf_y[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
		pix_y[i] = tile_origin_y + thread_row_base + i;
		pixf_y[i] = (float)pix_y[i];
	}

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			T[i][j] = 1.0f;
		}
	}

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			C[i][j] = { 0.0f, 0.0f, 0.0f };
		}
	}

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
		{
			offset = point_id + 2;
		}
		else if (lane == 4)
		{
			offset = point_id + 3;
		}
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
		{
			offset = point_list[point_id + 0];
		}
		else if (point_id + 1 < range.y)
		{
			offset = point_list[point_id + 1];
		}
		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
		{
			load_enable = load_enable && point_id + 2 < range.y;
		}
		else if (lane == 4)
		{
			load_enable = load_enable && point_id + 3 < range.y;
		}
		else if ((lane & 4) == 0)
		{
			load_enable = load_enable && point_id + 0 < range.y;
		}
		else
		{
			load_enable = load_enable && point_id + 1 < range.y;
		}
		if (load_enable)
		{
			buf = load_lane_value_topk_smem_fast(
				data, lg2_scale, offset, is_feature_lane, num_topk, smem_lane_ptr);
		}

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

			float ldg_buf;
			if (load_enable)
			{
				ldg_buf = load_lane_value_topk_smem_fast(
					data, lg2_scale, offset, is_feature_lane, num_topk, smem_lane_ptr);
			}

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}

			point_id += 2;
			buf = ldg_buf;
		}
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

			if (lane == 0)
			{
				load_enable = load_enable && point_id + 4 < range.y;
			}
			else if (lane == 4)
			{
				load_enable = load_enable && point_id + 5 < range.y;
			}
			else if ((lane & 4) == 0)
			{
				load_enable = load_enable && point_id + 2 < range.y;
			}
			else
			{
				load_enable = load_enable && point_id + 3 < range.y;
			}

			float ldg_buf;
			if (load_enable)
			{
				ldg_buf = load_lane_value_topk_smem_fast(
					data, lg2_scale, offset, is_feature_lane, num_topk, smem_lane_ptr);
			}

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			if (point_id + 1 >= range.y)
				break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
			}
		}
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix_x[j] < width && pix_y[i] < height)
				{
					int pix_id = width * pix_y[i] + pix_x[j];
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
		}
	}
}

__global__ __launch_bounds__(64) void renderCUDA32x16Split(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 32;
	constexpr int BLOCK_Y = 16;
	constexpr int THREAD_X = 2;
	constexpr int THREAD_Y = 4;

	uint32_t tile_id = blockIdx.y * x_blocks + blockIdx.x;
	int2 range = find_tile_range(sorted_keys, num_rendered, tile_id);
	int warp = threadIdx.y >> 1;
	int lane = ((int)threadIdx.y & 1) * blockDim.x + threadIdx.x;
	int tile_origin_x = (int)blockIdx.x * BLOCK_X;
	int tile_origin_y = (int)blockIdx.y * BLOCK_Y;
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = warp * 8 + (((int)threadIdx.y & 1) * THREAD_Y);

	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	int pix_x[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
	{
		pix_x[j] = tile_origin_x + thread_col_base + j;
	}

	int pix_y[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
		pix_y[i] = tile_origin_y + thread_row_base + i;
	}

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			T[i][j] = 1.0f;
		}
	}

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			C[i][j] = { 0.0f, 0.0f, 0.0f };
		}
	}

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
		{
			offset = point_id + 2;
		}
		else if (lane == 4)
		{
			offset = point_id + 3;
		}
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
		{
			offset = point_list[point_id + 0];
		}
		else if (point_id + 1 < range.y)
		{
			offset = point_list[point_id + 1];
		}
		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
		{
			load_enable = load_enable && point_id + 2 < range.y;
		}
		else if (lane == 4)
		{
			load_enable = load_enable && point_id + 3 < range.y;
		}
		else if ((lane & 4) == 0)
		{
			load_enable = load_enable && point_id + 0 < range.y;
		}
		else
		{
			load_enable = load_enable && point_id + 1 < range.y;
		}
		if (load_enable)
		{
			buf = load_lane_value(data, lg2_scale, offset);
		}

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0u, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0u, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

			float ldg_buf;
			if (load_enable)
			{
				ldg_buf = load_lane_value(data, lg2_scale, offset);
			}

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}

			point_id += 2;
			buf = ldg_buf;
		}
		while (__any_sync(~0u, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0u, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

			if (lane == 0)
			{
				load_enable = load_enable && point_id + 4 < range.y;
			}
			else if (lane == 4)
			{
				load_enable = load_enable && point_id + 5 < range.y;
			}
			else if ((lane & 4) == 0)
			{
				load_enable = load_enable && point_id + 2 < range.y;
			}
			else
			{
				load_enable = load_enable && point_id + 3 < range.y;
			}

			float ldg_buf;
			if (load_enable)
			{
				ldg_buf = load_lane_value(data, lg2_scale, offset);
			}

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			if (point_id + 1 >= range.y)
				break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
			}
		}
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix_x[j] < width && pix_y[i] < height)
				{
					int pix_id = width * pix_y[i] + pix_x[j];
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
		}
	}
}

template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y, int WARPS_Y>
__global__ __launch_bounds__(32 * WARPS_Y) void renderCUDAWarpSplit(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	uint32_t tile_id = blockIdx.y * x_blocks + blockIdx.x;
	int2 range = find_tile_range(sorted_keys, num_rendered, tile_id);
	int warp = (int)threadIdx.y >> 2;
	int lane = ((int)threadIdx.y & 3) * blockDim.x + threadIdx.x;
	int tile_origin_x = (int)blockIdx.x * BLOCK_X;
	int tile_origin_y = (int)blockIdx.y * BLOCK_Y;
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = warp * (THREAD_Y * 4) + (((int)threadIdx.y & 3) * THREAD_Y);

	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	int pix_x[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
	{
		pix_x[j] = tile_origin_x + thread_col_base + j;
	}

	int pix_y[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
		pix_y[i] = tile_origin_y + thread_row_base + i;
	}

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			T[i][j] = 1.0f;
		}
	}

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			C[i][j] = { 0.0f, 0.0f, 0.0f };
		}
	}

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
		{
			offset = point_id + 2;
		}
		else if (lane == 4)
		{
			offset = point_id + 3;
		}
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
		{
			offset = point_list[point_id + 0];
		}
		else if (point_id + 1 < range.y)
		{
			offset = point_list[point_id + 1];
		}
		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
		{
			load_enable = load_enable && point_id + 2 < range.y;
		}
		else if (lane == 4)
		{
			load_enable = load_enable && point_id + 3 < range.y;
		}
		else if ((lane & 4) == 0)
		{
			load_enable = load_enable && point_id + 0 < range.y;
		}
		else
		{
			load_enable = load_enable && point_id + 1 < range.y;
		}
		if (load_enable)
		{
			buf = load_lane_value(data, lg2_scale, offset);
		}

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0u, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0u, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

			float ldg_buf;
			if (load_enable)
			{
				ldg_buf = load_lane_value(data, lg2_scale, offset);
			}

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}

			point_id += 2;
			buf = ldg_buf;
		}
		while (__any_sync(~0u, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0u, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

			if (lane == 0)
			{
				load_enable = load_enable && point_id + 4 < range.y;
			}
			else if (lane == 4)
			{
				load_enable = load_enable && point_id + 5 < range.y;
			}
			else if ((lane & 4) == 0)
			{
				load_enable = load_enable && point_id + 2 < range.y;
			}
			else
			{
				load_enable = load_enable && point_id + 3 < range.y;
			}

			float ldg_buf;
			if (load_enable)
			{
				ldg_buf = load_lane_value(data, lg2_scale, offset);
			}

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			if (point_id + 1 >= range.y)
				break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T,
				tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base,
				xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
			}
		}
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix_x[j] < width && pix_y[i] < height)
				{
					int pix_id = width * pix_y[i] + pix_x[j];
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
		}
	}
}

template<int BLOCK_X, int BLOCK_Y, bool USE_UNROLL2 = false>
void render(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

    renderCUDA<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4, USE_UNROLL2><<<grid, dim3(8, 4, 1), 0, stream>>>(
        gaussian_keys_sorted,
        num_rendered,
        gaussian_values_sorted,
        width, height, grid.x,
        points_xy,
        rgb_depth,
        conic_opacity,
        render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
        bg_color,
        out_color);
}

template<int BLOCK_X, int BLOCK_Y, int WARPS_Y>
void render_split(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int THREAD_X = BLOCK_X / 8;
	constexpr int THREAD_Y = BLOCK_Y / (WARPS_Y * 4);
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	renderCUDAWarpSplit<BLOCK_X, BLOCK_Y, THREAD_X, THREAD_Y, WARPS_Y><<<grid, dim3(8, 4 * WARPS_Y, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, grid.x,
		points_xy,
		rgb_depth,
		conic_opacity,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

// ============================================================================
// Direction A: extract tile counts from precomputed tile_ranges
// ============================================================================
__global__ void extractTileCounts(
	const int2* __restrict__ tile_ranges,
	int total_tiles,
	uint32_t* __restrict__ tile_counts,
	uint32_t* __restrict__ tile_ids)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < total_tiles)
	{
		int2 range = tile_ranges[tid];
		tile_counts[tid] = (uint32_t)(range.y - range.x);
		tile_ids[tid] = (uint32_t)tid;
	}
}

// Morton (Z-order) code for 2D coordinates — interleaves bits of x and y
__device__ __forceinline__ uint32_t morton_2d(uint32_t x, uint32_t y)
{
	x = (x | (x << 8)) & 0x00FF00FFu;
	x = (x | (x << 4)) & 0x0F0F0F0Fu;
	x = (x | (x << 2)) & 0x33333333u;
	x = (x | (x << 1)) & 0x55555555u;
	y = (y | (y << 8)) & 0x00FF00FFu;
	y = (y | (y << 4)) & 0x0F0F0F0Fu;
	y = (y | (y << 2)) & 0x33333333u;
	y = (y | (y << 1)) & 0x55555555u;
	return x | (y << 1);
}

// Inverse morton: extract even bits (x) and odd bits (y) from a morton code
__device__ __forceinline__ uint32_t morton_extract_even(uint32_t z)
{
	z = z & 0x55555555u;
	z = (z | (z >> 1)) & 0x33333333u;
	z = (z | (z >> 2)) & 0x0F0F0F0Fu;
	z = (z | (z >> 4)) & 0x00FF00FFu;
	z = (z | (z >> 8)) & 0x0000FFFFu;
	return z;
}

// Convert 1D morton index to 2D (x, y), clamped to grid bounds.
// Returns linear tile_id = y * x_blocks + x, or -1 if out of bounds.
__device__ __forceinline__ int morton_to_tile_id(int morton_idx, int x_blocks, int y_blocks)
{
	uint32_t x = morton_extract_even((uint32_t)morton_idx);
	uint32_t y = morton_extract_even((uint32_t)morton_idx >> 1);
	if ((int)x >= x_blocks || (int)y >= y_blocks)
		return -1;
	return (int)y * x_blocks + (int)x;
}

// Hilbert curve: map (x, y) in [0, n) × [0, n) to 1D Hilbert index.
// n must be a power of 2. Better spatial locality than Morton — consecutive
// indices are always edge-adjacent (no diagonal jumps).
__device__ __forceinline__ uint32_t hilbert_xy_to_d(uint32_t n, uint32_t x, uint32_t y)
{
	uint32_t d = 0;
	for (uint32_t s = n >> 1; s > 0; s >>= 1)
	{
		uint32_t rx = (x & s) > 0 ? 1u : 0u;
		uint32_t ry = (y & s) > 0 ? 1u : 0u;
		d += s * s * ((3u * rx) ^ ry);
		// Rotate quadrant
		if (ry == 0)
		{
			if (rx == 1)
			{
				x = s - 1 - x;
				y = s - 1 - y;
			}
			uint32_t t = x;
			x = y;
			y = t;
		}
	}
	return d;
}

// Composite sort key: (count_bucket << 16) | (0xFFFF - morton_code)
// For descending sort:
//   - Higher count_bucket → comes first (heavy tiles first)
//   - Within same bucket, inverted morton → ascending Morton order (spatial locality)
__global__ void extractTileCountsMorton(
	const int2* __restrict__ tile_ranges,
	int total_tiles,
	int x_blocks,
	int bucket_size,
	uint32_t* __restrict__ tile_sort_keys,
	uint32_t* __restrict__ tile_ids)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < total_tiles)
	{
		int2 range = tile_ranges[tid];
		uint32_t count = (uint32_t)(range.y - range.x);
		uint32_t bucket = count / (uint32_t)bucket_size;
		if (bucket > 0xFFFFu) bucket = 0xFFFFu;

		uint32_t tile_x = (uint32_t)(tid % x_blocks);
		uint32_t tile_y = (uint32_t)(tid / x_blocks);
		uint32_t morton = morton_2d(tile_x, tile_y) & 0xFFFFu;

		tile_sort_keys[tid] = (bucket << 16) | (0xFFFFu - morton);
		tile_ids[tid] = (uint32_t)tid;
	}
}

// Composite sort key using Hilbert curve for better spatial locality than Morton.
// Hilbert curves guarantee consecutive indices are always edge-adjacent (no diagonal jumps).
__global__ void extractTileCountsHilbert(
	const int2* __restrict__ tile_ranges,
	int total_tiles,
	int x_blocks,
	int y_blocks,
	int bucket_size,
	uint32_t* __restrict__ tile_sort_keys,
	uint32_t* __restrict__ tile_ids)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < total_tiles)
	{
		int2 range = tile_ranges[tid];
		uint32_t count = (uint32_t)(range.y - range.x);
		uint32_t bucket = count / (uint32_t)bucket_size;
		if (bucket > 0xFFFFu) bucket = 0xFFFFu;

		uint32_t tile_x = (uint32_t)(tid % x_blocks);
		uint32_t tile_y = (uint32_t)(tid / x_blocks);

		// Round up to next power of 2 for Hilbert curve
		uint32_t n = 1;
		uint32_t max_dim = (uint32_t)x_blocks > (uint32_t)y_blocks ? (uint32_t)x_blocks : (uint32_t)y_blocks;
		while (n < max_dim) n <<= 1;

		uint32_t hilbert = hilbert_xy_to_d(n, tile_x, tile_y) & 0xFFFFu;

		tile_sort_keys[tid] = (bucket << 16) | (0xFFFFu - hilbert);
		tile_ids[tid] = (uint32_t)tid;
	}
}

// Regather gaussian data to match new tile processing order.
// After tile reordering, tiles are processed in a different order than the
// original tile_id sort. This kernel copies gaussian feature data so that
// the render kernel sees sequential memory access.
// tile_desc[i] = {col, row, old_range_start, old_range_end}
// new_ranges[i] = {new_start, new_end} in the regathered arrays
__global__ void regatherGaussianData(
	const int4* __restrict__ tile_desc,
	const uint32_t* __restrict__ new_offsets,  // prefix sum of tile counts in new order
	int total_tiles,
	const uint32_t* __restrict__ old_point_list,
	const float2* __restrict__ old_xy,
	const float4* __restrict__ old_rgb,
	const float4* __restrict__ old_conic,
	uint32_t* __restrict__ new_point_list,
	float2* __restrict__ new_xy,
	float4* __restrict__ new_rgb,
	float4* __restrict__ new_conic)
{
	// Each block handles one tile
	int tile_idx = blockIdx.x;
	if (tile_idx >= total_tiles) return;

	int4 desc = __ldg(&tile_desc[tile_idx]);
	int old_start = desc.z;
	int old_end = desc.w;
	int count = old_end - old_start;
	int new_start = __ldg(&new_offsets[tile_idx]);

	// Each thread handles multiple gaussians
	for (int i = threadIdx.x; i < count; i += blockDim.x)
	{
		int old_idx = old_start + i;
		int new_idx = new_start + i;

		uint32_t gid = old_point_list[old_idx];
		new_point_list[new_idx] = gid;

		// Gather features
		new_xy[new_idx] = __ldg(&old_xy[gid]);
		new_rgb[new_idx] = __ldg(&old_rgb[gid]);
		new_conic[new_idx] = __ldg(&old_conic[gid]);
	}
}

// Compute prefix sum of tile counts in new tile order.
// Sets new_offsets[i] = sum of counts for tiles 0..i-1 in new order.
// Also updates tile_desc[i].z/w to point to new ranges.
__global__ void computeNewTileOffsets(
	int4* __restrict__ tile_desc,
	const uint32_t* __restrict__ prefix_sum,
	int total_tiles)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < total_tiles)
	{
		int4 desc = tile_desc[tid];
		int count = desc.w - desc.z;
		int new_start = prefix_sum[tid];
		desc.z = new_start;
		desc.w = new_start + count;
		tile_desc[tid] = desc;
	}
}

// ============================================================================
// Direction A: reordered render — same as default 16x16 but uses
// tile_order[] to map linear block index -> actual tile_id
// ============================================================================
template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA16x16Reordered(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int2* __restrict__ tile_ranges,
	const uint32_t* __restrict__ tile_order,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;

	uint32_t linear_idx = blockIdx.y * (uint32_t)gridDim.x + blockIdx.x;
	uint32_t tile_id = __ldg(&tile_order[linear_idx]);
	int2 range = __ldg(&tile_ranges[tile_id]);

	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
	int tile_col = tile_id % x_blocks;
	int tile_row = tile_id / x_blocks;
	int tile_origin_x = tile_col * BLOCK_X;
	int tile_origin_y = tile_row * BLOCK_Y;
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	int pix_x[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
		pix_x[j] = tile_origin_x + thread_col_base + j;

	int pix_y[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
		pix_y[i] = tile_origin_y + thread_row_base + i;

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			T[i][j] = 1.0f;

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			C[i][j] = { 0.0f, 0.0f, 0.0f };

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
			offset = point_id + 2;
		else if (lane == 4)
			offset = point_id + 3;
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
			offset = point_list[point_id + 0];
		else if (point_id + 1 < range.y)
			offset = point_list[point_id + 1];

		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
			load_enable = load_enable && point_id + 2 < range.y;
		else if (lane == 4)
			load_enable = load_enable && point_id + 3 < range.y;
		else if ((lane & 4) == 0)
			load_enable = load_enable && point_id + 0 < range.y;
		else
			load_enable = load_enable && point_id + 1 < range.y;

		if (load_enable)
			buf = load_lane_value(data, lg2_scale, offset);

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;

			point_id += 2;
			buf = ldg_buf;
		}
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			if (lane == 0)
				load_enable = load_enable && point_id + 4 < range.y;
			else if (lane == 4)
				load_enable = load_enable && point_id + 5 < range.y;
			else if ((lane & 4) == 0)
				load_enable = load_enable && point_id + 2 < range.y;
			else
				load_enable = load_enable && point_id + 3 < range.y;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			if (point_id + 1 >= range.y) break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix_x[j] < width && pix_y[i] < height)
				{
					int pix_id = width * pix_y[i] + pix_x[j];
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
	}
}

// ============================================================================
// Direction A v2: gather packed int4 tile descriptor
// tile_desc[new_pos] = {tile_col, tile_row, range_start, range_end}
// where new_pos follows tile_order (descending count).
// ============================================================================
__global__ void gatherTileDesc(
	const uint32_t* __restrict__ tile_order,
	const int2* __restrict__ tile_ranges,
	int total_tiles,
	int x_blocks,
	int4* __restrict__ tile_desc)
{
	int new_pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (new_pos < total_tiles)
	{
		uint32_t old_id = tile_order[new_pos];
		int2 range = __ldg(&tile_ranges[old_id]);
		int4 desc;
		desc.x = (int)(old_id) % x_blocks;
		desc.y = (int)(old_id) / x_blocks;
		desc.z = range.x;
		desc.w = range.y;
		tile_desc[new_pos] = desc;
	}
}

// Zig-zag gather: remap descending-sorted tile_order via zigzag across groups of
// `group_size`. Under HW round-robin, SM i receives tile at slot i in each group.
// Alternating reverse-direction groups ensures SM i sees heaviest/lightest on
// alternating groups, cancelling the per-group imbalance.
__global__ void gatherTileDescZigzag(
	const uint32_t* __restrict__ tile_order,
	const int2* __restrict__ tile_ranges,
	int total_tiles,
	int x_blocks,
	int group_size,
	int4* __restrict__ tile_desc)
{
	int new_pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (new_pos < total_tiles)
	{
		int group = new_pos / group_size;
		int slot = new_pos % group_size;
		// Reverse direction on odd groups
		int base = group * group_size;
		int group_end = base + group_size;
		if (group_end > total_tiles) group_end = total_tiles;
		int group_len = group_end - base;
		int src;
		if (group & 1)
			src = base + (group_len - 1 - slot);
		else
			src = base + slot;
		if (src >= total_tiles) src = total_tiles - 1;
		uint32_t old_id = tile_order[src];
		int2 range = __ldg(&tile_ranges[old_id]);
		int4 desc;
		desc.x = (int)(old_id) % x_blocks;
		desc.y = (int)(old_id) / x_blocks;
		desc.z = range.x;
		desc.w = range.y;
		tile_desc[new_pos] = desc;
	}
}

// Interleaved (round-robin) tile reordering: assigns tiles from the count-
// descending sorted order to SM groups in round-robin fashion.
// With N SMs and 28 blocks/SM, the first N×28 tiles get distributed as:
//   SM 0: tiles 0, N, 2N, ...
//   SM 1: tiles 1, N+1, 2N+1, ...
// Then zigzag is applied on top: odd "rounds" (groups of group_size) are
// reversed.  This ensures each SM gets an even mix of heavy and light tiles.
__global__ void gatherTileDescInterleaved(
	const uint32_t* __restrict__ tile_order,
	const int2* __restrict__ tile_ranges,
	int total_tiles,
	int x_blocks,
	int num_sms,
	int blocks_per_sm,
	int group_size,
	int4* __restrict__ tile_desc)
{
	int new_pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (new_pos < total_tiles)
	{
		// Round-robin distribute: position new_pos maps to sorted index src
		// such that SM s gets tiles s, s+N, s+2N, ... (stride = num_sms * blocks_per_sm)
		int wave_size = num_sms * blocks_per_sm;
		int wave = new_pos / wave_size;
		int pos_in_wave = new_pos % wave_size;

		// Within a wave, apply zigzag if group_size > 0
		int src_in_wave = pos_in_wave;
		if (group_size > 1)
		{
			int group = pos_in_wave / group_size;
			int slot = pos_in_wave % group_size;
			int base = group * group_size;
			int group_end_in_wave = base + group_size;
			int wave_len = wave_size;
			if (wave * wave_size + wave_len > total_tiles)
				wave_len = total_tiles - wave * wave_size;
			if (group_end_in_wave > wave_len)
				group_end_in_wave = wave_len;
			int group_len = group_end_in_wave - base;
			if (group & 1)
				src_in_wave = base + (group_len - 1 - min(slot, group_len - 1));
			else
				src_in_wave = base + min(slot, group_len - 1);
		}

		int src = wave * wave_size + src_in_wave;
		if (src >= total_tiles) src = total_tiles - 1;

		uint32_t old_id = tile_order[src];
		int2 range = __ldg(&tile_ranges[old_id]);
		int4 desc;
		desc.x = (int)(old_id) % x_blocks;
		desc.y = (int)(old_id) / x_blocks;
		desc.z = range.x;
		desc.w = range.y;
		tile_desc[new_pos] = desc;
	}
}

// ============================================================================
// Direction A v2: reordered render with packed int4 tile descriptor.
// A single __ldg fetches {col, row, range_start, range_end}, removing:
//   - second __ldg for tile_ranges
//   - tile_id % x_blocks / tile_id / x_blocks integer div/mod
//   - the long dependency chain through tile_id
// Target: REG=64 (vs v1 reordered REG=72).
// ============================================================================
template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA16x16ReorderedV2(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int4* __restrict__ tile_desc,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;

	uint32_t linear_idx = blockIdx.y * (uint32_t)gridDim.x + blockIdx.x;
	int4 desc = __ldg(&tile_desc[linear_idx]);
	int tile_origin_x = desc.x * BLOCK_X;
	int tile_origin_y = desc.y * BLOCK_Y;
	int2 range = { desc.z, desc.w };

	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	int pix_x[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
		pix_x[j] = tile_origin_x + thread_col_base + j;

	int pix_y[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
		pix_y[i] = tile_origin_y + thread_row_base + i;

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			T[i][j] = 1.0f;

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			C[i][j] = { 0.0f, 0.0f, 0.0f };

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
			offset = point_id + 2;
		else if (lane == 4)
			offset = point_id + 3;
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
			offset = point_list[point_id + 0];
		else if (point_id + 1 < range.y)
			offset = point_list[point_id + 1];

		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
			load_enable = load_enable && point_id + 2 < range.y;
		else if (lane == 4)
			load_enable = load_enable && point_id + 3 < range.y;
		else if ((lane & 4) == 0)
			load_enable = load_enable && point_id + 0 < range.y;
		else
			load_enable = load_enable && point_id + 1 < range.y;

		if (load_enable)
			buf = load_lane_value(data, lg2_scale, offset);

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;

			point_id += 2;
			buf = ldg_buf;
		}
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			if (lane == 0)
				load_enable = load_enable && point_id + 4 < range.y;
			else if (lane == 4)
				load_enable = load_enable && point_id + 5 < range.y;
			else if ((lane & 4) == 0)
				load_enable = load_enable && point_id + 2 < range.y;
			else
				load_enable = load_enable && point_id + 3 < range.y;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			if (point_id + 1 >= range.y) break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix_x[j] < width && pix_y[i] < height)
				{
					int pix_id = width * pix_y[i] + pix_x[j];
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
	}
}

// ============================================================================
// Reordered V3: same as V2 but eliminates pix_x[]/pix_y[] arrays to reduce
// register pressure.  Those arrays only serve the write_color phase and can be
// recomputed cheaply from tile_origin_{x,y} + thread_{col,row}_base + loop idx.
// Target: 64 regs / 0 stack → 32 blocks/SM → +14% occupancy vs V2's 72/28.
// ============================================================================
template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA16x16ReorderedV3(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int4* __restrict__ tile_desc,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;

	uint32_t linear_idx = blockIdx.y * (uint32_t)gridDim.x + blockIdx.x;
	int4 desc = __ldg(&tile_desc[linear_idx]);
	int tile_origin_x = desc.x * BLOCK_X;
	int tile_origin_y = desc.y * BLOCK_Y;
	int2 range = { desc.z, desc.w };

	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	// NO pix_x[]/pix_y[] arrays — recompute at write time to save ~6 registers.

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			T[i][j] = 1.0f;

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			C[i][j] = { 0.0f, 0.0f, 0.0f };

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
			offset = point_id + 2;
		else if (lane == 4)
			offset = point_id + 3;
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
			offset = point_list[point_id + 0];
		else if (point_id + 1 < range.y)
			offset = point_list[point_id + 1];

		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
			load_enable = load_enable && point_id + 2 < range.y;
		else if (lane == 4)
			load_enable = load_enable && point_id + 3 < range.y;
		else if ((lane & 4) == 0)
			load_enable = load_enable && point_id + 0 < range.y;
		else
			load_enable = load_enable && point_id + 1 < range.y;

		if (load_enable)
			buf = load_lane_value(data, lg2_scale, offset);

		load_enable = data != nullptr;

		bool done = false;

		// ---- 4-gaussian fast loop: check done every 4 gaussians ----
		// Manually unrolled 2 pairs to reduce __any_sync overhead by ~50%.
		while (__any_sync(~0, point_id + 9 < range.y && !done))
		{
			// Pair A
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;
			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);
			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);
			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);
			point_id += 2;
			buf = ldg_buf;

			// Pair B
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);
			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);
			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;

			point_id += 2;
			buf = ldg_buf;
		}

		// ---- 2-gaussian loop for remainder ----
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;

			point_id += 2;
			buf = ldg_buf;
		}

		// ---- Tail loop with bounds checking ----
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			if (lane == 0)
				load_enable = load_enable && point_id + 4 < range.y;
			else if (lane == 4)
				load_enable = load_enable && point_id + 5 < range.y;
			else if ((lane & 4) == 0)
				load_enable = load_enable && point_id + 2 < range.y;
			else
				load_enable = load_enable && point_id + 3 < range.y;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			if (point_id + 1 >= range.y) break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			point_id += 2;
			buf = ldg_buf;
		}
		// Write phase: recompute pixel positions from tile_origin + thread base
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color,
					{ tile_origin_x + thread_col_base + j,
					  tile_origin_y + thread_row_base + i },
					width, height, C[i][j], T[i][j]);
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int px = tile_origin_x + thread_col_base + j;
				int py = tile_origin_y + thread_row_base + i;
				if (px < width && py < height)
				{
					int pix_id = width * py + px;
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
	}
}

// ============================================================================
// Direction A v2 (persistent combo): reordered persistent with work-stealing.
// Each warp claims the next packed descriptor via atomicAdd(&g_tile_counter).
// Combines heavy-first ordering with dynamic work-stealing for tail balance.
// Uses __noinline__ tile function (same as persistent_v3) to isolate T/C
// register lifetimes.
// ============================================================================
template<int THREAD_X, int THREAD_Y>
__noinline__ __device__ void renderTileReorderedV2(
	int tile_origin_x, int tile_origin_y, int2 range,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	const void* data, int lg2_scale,
	int thread_col_base, int thread_row_base,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;
	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;

	int pix_x[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
		pix_x[j] = tile_origin_x + thread_col_base + j;

	int pix_y[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
		pix_y[i] = tile_origin_y + thread_row_base + i;

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			T[i][j] = 1.0f;

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			C[i][j] = { 0.0f, 0.0f, 0.0f };

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
			offset = point_id + 2;
		else if (lane == 4)
			offset = point_id + 3;
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
			offset = point_list[point_id + 0];
		else if (point_id + 1 < range.y)
			offset = point_list[point_id + 1];

		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
			load_enable = load_enable && point_id + 2 < range.y;
		else if (lane == 4)
			load_enable = load_enable && point_id + 3 < range.y;
		else if ((lane & 4) == 0)
			load_enable = load_enable && point_id + 0 < range.y;
		else
			load_enable = load_enable && point_id + 1 < range.y;

		if (load_enable)
			buf = load_lane_value(data, lg2_scale, offset);

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;

			point_id += 2;
			buf = ldg_buf;
		}
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			if (lane == 0)
				load_enable = load_enable && point_id + 4 < range.y;
			else if (lane == 4)
				load_enable = load_enable && point_id + 5 < range.y;
			else if ((lane & 4) == 0)
				load_enable = load_enable && point_id + 2 < range.y;
			else
				load_enable = load_enable && point_id + 3 < range.y;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			if (point_id + 1 >= range.y) break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix_x[j] < width && pix_y[i] < height)
				{
					int pix_id = width * pix_y[i] + pix_x[j];
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
	}
}

template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32, 28)
void renderCUDA16x16ReorderedPersistent(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	int total_tiles,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int4* __restrict__ tile_desc,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;

	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	// First-tile claim: use blockIdx.x (static map, avoids atomic contention storm
	// at kernel launch where all 3696 warps would otherwise hammer the same counter).
	int idx = (int)blockIdx.x;

	int4 desc = make_int4(0, 0, 0, 0);
	if (idx < total_tiles && lane == 0)
		desc = __ldg(&tile_desc[idx]);
	desc.x = __shfl_sync(~0, desc.x, 0);
	desc.y = __shfl_sync(~0, desc.y, 0);
	desc.z = __shfl_sync(~0, desc.z, 0);
	desc.w = __shfl_sync(~0, desc.w, 0);

	while (idx < total_tiles)
	{
		int tile_origin_x = desc.x * BLOCK_X;
		int tile_origin_y = desc.y * BLOCK_Y;
		int2 range = { desc.z, desc.w };

		renderTileReorderedV2<THREAD_X, THREAD_Y>(
			tile_origin_x, tile_origin_y, range,
			point_list, width, height,
			data, lg2_scale,
			thread_col_base, thread_row_base,
			bg_color, out_color);

		// Claim next tile: tail wave via work-stealing. We offset by gridDim.x
		// (== num_sms * blocks_per_sm) so the atomic range covers tiles beyond
		// the static first wave.
		if (lane == 0)
			idx = (int)atomicAdd(&g_tile_counter, 1) + (int)gridDim.x;
		idx = __shfl_sync(~0, idx, 0);

		desc = make_int4(0, 0, 0, 0);
		if (idx < total_tiles && lane == 0)
			desc = __ldg(&tile_desc[idx]);
		desc.x = __shfl_sync(~0, desc.x, 0);
		desc.y = __shfl_sync(~0, desc.y, 0);
		desc.z = __shfl_sync(~0, desc.z, 0);
		desc.w = __shfl_sync(~0, desc.w, 0);
	}
}

// ============================================================================
// Reordered Persistent NoSort: Same V2 inner loop as ReorderedPersistent, but
// reads ranges directly from tile_ranges_buf (no CUB sort / tile_desc). Claimed
// tile indices are transformed on-the-fly to get spatial locality without the
// 50-60us sort overhead.
//
// tile_order_mode:
//   0 → natural row-major order
//   1 → zigzag: groups of zigzag_group_size, reverse every other group
//   2 → morton: traverse tiles in Z-order curve (spatial locality)
//   3 → morton + zigzag: morton order within zigzag groups
// ============================================================================
template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32, 28)
void renderCUDA16x16ReorderedPersistentNoSort(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	int x_blocks, int y_blocks, int total_tiles,
	int tile_order_mode,
	int zigzag_group_size,
	int morton_total,  // total cells in morton grid (next_pow2(max(x,y))^2)
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int2* __restrict__ tile_ranges,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;

	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	// First-tile claim via static map (avoids atomic contention storm).
	int idx = (int)blockIdx.x;
	// For morton modes, idx is a morton counter. We need a separate mapping
	// to handle out-of-bounds morton cells (where x >= x_blocks or y >= y_blocks).
	// morton_idx tracks how many morton cells we've claimed (may exceed total_tiles
	// due to skipped cells).
	int morton_idx = idx;  // only used in morton modes

	while (idx < total_tiles)
	{
		int tile_id;

		if (tile_order_mode == 2 || tile_order_mode == 3)
		{
			// Morton traversal: convert claimed morton_idx to tile_id.
			// Skip out-of-bounds morton cells (where x >= x_blocks or y >= y_blocks).
			tile_id = -1;
			while (tile_id < 0 && morton_idx < morton_total)
			{
				int candidate;
				if (tile_order_mode == 3 && zigzag_group_size > 1)
				{
					// Morton + zigzag: zigzag within groups of the morton sequence
					int group = morton_idx / zigzag_group_size;
					int slot = morton_idx - group * zigzag_group_size;
					int base_idx = group * zigzag_group_size;
					int group_end = base_idx + zigzag_group_size;
					if (group_end > morton_total) group_end = morton_total;
					int group_len = group_end - base_idx;
					if (group & 1)
						candidate = base_idx + (group_len - 1 - slot);
					else
						candidate = base_idx + slot;
				}
				else
				{
					candidate = morton_idx;
				}
				tile_id = morton_to_tile_id(candidate, x_blocks, y_blocks);
				if (tile_id < 0)
				{
					// This morton cell is out of bounds, claim next
					if (lane == 0)
						morton_idx = (int)atomicAdd(&g_tile_counter, 1) + (int)gridDim.x;
					morton_idx = __shfl_sync(~0, morton_idx, 0);
				}
			}
			if (tile_id < 0) break;  // no more tiles
		}
		else if (tile_order_mode == 1 && zigzag_group_size > 1)
		{
			// Zigzag grouping on row-major order
			int group = idx / zigzag_group_size;
			int slot = idx - group * zigzag_group_size;
			int base = group * zigzag_group_size;
			int group_end = base + zigzag_group_size;
			if (group_end > total_tiles) group_end = total_tiles;
			int group_len = group_end - base;
			if (group & 1)
				tile_id = base + (group_len - 1 - slot);
			else
				tile_id = base + slot;
			if (tile_id >= total_tiles) tile_id = total_tiles - 1;
		}
		else
		{
			// Natural row-major
			tile_id = idx;
		}

		int tile_col = tile_id % x_blocks;
		int tile_row = tile_id / x_blocks;
		int tile_origin_x = tile_col * BLOCK_X;
		int tile_origin_y = tile_row * BLOCK_Y;

		int2 range = make_int2(0, 0);
		if (lane == 0)
			range = __ldg(&tile_ranges[tile_id]);
		range.x = __shfl_sync(~0, range.x, 0);
		range.y = __shfl_sync(~0, range.y, 0);

		renderTileReorderedV2<THREAD_X, THREAD_Y>(
			tile_origin_x, tile_origin_y, range,
			point_list, width, height,
			data, lg2_scale,
			thread_col_base, thread_row_base,
			bg_color, out_color);

		// Claim next tile via work-stealing.
		if (tile_order_mode >= 2)
		{
			// Morton modes: advance morton counter (may skip OOB cells)
			if (lane == 0)
				morton_idx = (int)atomicAdd(&g_tile_counter, 1) + (int)gridDim.x;
			morton_idx = __shfl_sync(~0, morton_idx, 0);
			// idx tracks valid tiles processed (for loop termination)
			idx++;
		}
		else
		{
			if (lane == 0)
				idx = (int)atomicAdd(&g_tile_counter, 1) + (int)gridDim.x;
			idx = __shfl_sync(~0, idx, 0);
		}
	}
}

// ============================================================================
// Reordered Persistent V3: V3's 4-gaussian fast loop inside work-stealing
// persistent wrapper. Uses __noinline__ tile function to isolate T/C register
// lifetimes. Benefits scenes with large tiles (truck) more than small tiles.
// ============================================================================
template<int THREAD_X, int THREAD_Y>
__noinline__ __device__ void renderTileReorderedV3(
	int tile_origin_x, int tile_origin_y, int2 range,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	const void* data, int lg2_scale,
	int thread_col_base, int thread_row_base,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;
	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			T[i][j] = 1.0f;

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			C[i][j] = { 0.0f, 0.0f, 0.0f };

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
			offset = point_id + 2;
		else if (lane == 4)
			offset = point_id + 3;
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
			offset = point_list[point_id + 0];
		else if (point_id + 1 < range.y)
			offset = point_list[point_id + 1];

		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
			load_enable = load_enable && point_id + 2 < range.y;
		else if (lane == 4)
			load_enable = load_enable && point_id + 3 < range.y;
		else if ((lane & 4) == 0)
			load_enable = load_enable && point_id + 0 < range.y;
		else
			load_enable = load_enable && point_id + 1 < range.y;

		if (load_enable)
			buf = load_lane_value(data, lg2_scale, offset);

		load_enable = data != nullptr;
		bool done = false;

		// 4-gaussian fast loop
		while (__any_sync(~0, point_id + 9 < range.y && !done))
		{
			// Pair A
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;
			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);
			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);
			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);
			point_id += 2;
			buf = ldg_buf;

			// Pair B
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);
			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);
			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;

			point_id += 2;
			buf = ldg_buf;
		}

		// 2-gaussian remainder loop
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;

			point_id += 2;
			buf = ldg_buf;
		}

		// Tail loop with bounds checking
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			if (lane == 0)
				load_enable = load_enable && point_id + 4 < range.y;
			else if (lane == 4)
				load_enable = load_enable && point_id + 5 < range.y;
			else if ((lane & 4) == 0)
				load_enable = load_enable && point_id + 2 < range.y;
			else
				load_enable = load_enable && point_id + 3 < range.y;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			if (point_id + 1 >= range.y) break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			point_id += 2;
			buf = ldg_buf;
		}

		// Write phase: recompute pixel positions from tile_origin + thread base
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color,
					{ tile_origin_x + thread_col_base + j,
					  tile_origin_y + thread_row_base + i },
					width, height, C[i][j], T[i][j]);
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int px = tile_origin_x + thread_col_base + j;
				int py = tile_origin_y + thread_row_base + i;
				if (px < width && py < height)
				{
					int pix_id = width * py + px;
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
	}
}

template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32, 28)
void renderCUDA16x16ReorderedPersistentV3(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	int total_tiles,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int4* __restrict__ tile_desc,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;

	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	int idx = (int)blockIdx.x;

	int4 desc = make_int4(0, 0, 0, 0);
	if (idx < total_tiles && lane == 0)
		desc = __ldg(&tile_desc[idx]);
	desc.x = __shfl_sync(~0, desc.x, 0);
	desc.y = __shfl_sync(~0, desc.y, 0);
	desc.z = __shfl_sync(~0, desc.z, 0);
	desc.w = __shfl_sync(~0, desc.w, 0);

	while (idx < total_tiles)
	{
		int tile_origin_x = desc.x * BLOCK_X;
		int tile_origin_y = desc.y * BLOCK_Y;
		int2 range = { desc.z, desc.w };

		renderTileReorderedV3<THREAD_X, THREAD_Y>(
			tile_origin_x, tile_origin_y, range,
			point_list, width, height,
			data, lg2_scale,
			thread_col_base, thread_row_base,
			bg_color, out_color);

		if (lane == 0)
			idx = (int)atomicAdd(&g_tile_counter, 1) + (int)gridDim.x;
		idx = __shfl_sync(~0, idx, 0);

		desc = make_int4(0, 0, 0, 0);
		if (idx < total_tiles && lane == 0)
			desc = __ldg(&tile_desc[idx]);
		desc.x = __shfl_sync(~0, desc.x, 0);
		desc.y = __shfl_sync(~0, desc.y, 0);
		desc.z = __shfl_sync(~0, desc.z, 0);
		desc.w = __shfl_sync(~0, desc.w, 0);
	}
}

// ============================================================================
// 3-Gaussian Warp Layout (V5): Process 3 gaussians per warp iteration
// instead of 2, using previously-idle lanes for the 3rd gaussian's data.
//
// Lane mapping:
//   ID prefetch: lane 0 → point_list[+3], lane 3 → point_list[+4], lane 19 → point_list[+5]
//   G0 (offset=0): xy → 8,9; rgb → 16,17,18; conic → 24,25,26,27
//   G1 (offset=4): xy → 12,13; rgb → 20,21,22; conic → 28,29,30,31
//   G2 (3rd gauss): xy → 1,2; rgb → 5,6,7; conic → 10,11,14,15
//   Unused: lane 23 (nullptr, 1 idle lane — down from 13)
//
// Benefits:
//   - 33% fewer loop iterations → 33% fewer __any_sync checks
//   - Better instruction-level parallelism from 3 shaders per iteration
//   - Same register count target (72) since features are consumed immediately
// ============================================================================

struct render_load_info_3g
{
	const void* data[POLYSPLAT_WARP_SIZE] = { nullptr };
	int lg2_scale[POLYSPLAT_WARP_SIZE] = { 0 };

	render_load_info_3g(const uint32_t* point_list, const float2* points_xy,
		const float4* rgb_depth, const float4* conic_opacity)
	{
		for (int lane = 0; lane < 32; lane++)
		{
			switch (lane)
			{
			// ID prefetch lanes
			case 0:  data[lane] = point_list; lg2_scale[lane] = 2; break;
			case 3:  data[lane] = point_list; lg2_scale[lane] = 2; break;
			case 19: data[lane] = point_list; lg2_scale[lane] = 2; break;
			// G0: xy(8,9), rgb(16,17,18), conic(24,25,26,27)
			case 8:  data[lane] = &points_xy->x; lg2_scale[lane] = 3; break;
			case 9:  data[lane] = &points_xy->y; lg2_scale[lane] = 3; break;
			case 16: data[lane] = &rgb_depth->x; lg2_scale[lane] = 4; break;
			case 17: data[lane] = &rgb_depth->y; lg2_scale[lane] = 4; break;
			case 18: data[lane] = &rgb_depth->z; lg2_scale[lane] = 4; break;
			case 24: data[lane] = &conic_opacity->x; lg2_scale[lane] = 4; break;
			case 25: data[lane] = &conic_opacity->y; lg2_scale[lane] = 4; break;
			case 26: data[lane] = &conic_opacity->z; lg2_scale[lane] = 4; break;
			case 27: data[lane] = &conic_opacity->w; lg2_scale[lane] = 4; break;
			// G1: xy(12,13), rgb(20,21,22), conic(28,29,30,31)
			case 12: data[lane] = &points_xy->x; lg2_scale[lane] = 3; break;
			case 13: data[lane] = &points_xy->y; lg2_scale[lane] = 3; break;
			case 20: data[lane] = &rgb_depth->x; lg2_scale[lane] = 4; break;
			case 21: data[lane] = &rgb_depth->y; lg2_scale[lane] = 4; break;
			case 22: data[lane] = &rgb_depth->z; lg2_scale[lane] = 4; break;
			case 28: data[lane] = &conic_opacity->x; lg2_scale[lane] = 4; break;
			case 29: data[lane] = &conic_opacity->y; lg2_scale[lane] = 4; break;
			case 30: data[lane] = &conic_opacity->z; lg2_scale[lane] = 4; break;
			case 31: data[lane] = &conic_opacity->w; lg2_scale[lane] = 4; break;
			// G2: xy(1,2), rgb(5,6,7), conic(10,11,14,15)
			case 1:  data[lane] = &points_xy->x; lg2_scale[lane] = 3; break;
			case 2:  data[lane] = &points_xy->y; lg2_scale[lane] = 3; break;
			case 5:  data[lane] = &rgb_depth->x; lg2_scale[lane] = 4; break;
			case 6:  data[lane] = &rgb_depth->y; lg2_scale[lane] = 4; break;
			case 7:  data[lane] = &rgb_depth->z; lg2_scale[lane] = 4; break;
			case 10: data[lane] = &conic_opacity->x; lg2_scale[lane] = 4; break;
			case 11: data[lane] = &conic_opacity->y; lg2_scale[lane] = 4; break;
			case 14: data[lane] = &conic_opacity->z; lg2_scale[lane] = 4; break;
			case 15: data[lane] = &conic_opacity->w; lg2_scale[lane] = 4; break;
			// lane 4, 23: unused
			default: data[lane] = nullptr; lg2_scale[lane] = 0; break;
			}
		}
	}
};

// Extract G2's features from the 3-gaussian lane layout
__forceinline__ __device__ void get_gaussian_features_g2(float2& xy, float3& rgb, float4& con_o, float buf)
{
	xy = {
		__shfl_sync(~0, buf, 1),
		__shfl_sync(~0, buf, 2)
	};
	rgb = {
		__shfl_sync(~0, buf, 5),
		__shfl_sync(~0, buf, 6),
		__shfl_sync(~0, buf, 7)
	};
	con_o = {
		__shfl_sync(~0, buf, 10),
		__shfl_sync(~0, buf, 11),
		__shfl_sync(~0, buf, 14),
		__shfl_sync(~0, buf, 15)
	};
}

// Determine which gaussian's ID this lane loads for the 3-gaussian layout.
// Returns 0 for G0 feature lanes, 1 for G1 feature lanes, 2 for G2 feature lanes,
// -1 for ID prefetch lanes, -2 for unused lanes.
__forceinline__ __device__ int lane_gauss_index_3g(int lane)
{
	switch (lane)
	{
	case 0: case 3: case 19: return -1; // ID prefetch
	case 8: case 9: case 16: case 17: case 18:
	case 24: case 25: case 26: case 27: return 0; // G0
	case 12: case 13: case 20: case 21: case 22:
	case 28: case 29: case 30: case 31: return 1; // G1
	case 1: case 2: case 5: case 6: case 7:
	case 10: case 11: case 14: case 15: return 2; // G2
	default: return -2; // unused (lane 4, 23)
	}
}

template<int THREAD_X, int THREAD_Y>
__noinline__ __device__ void renderTileReorderedV5(
	int tile_origin_x, int tile_origin_y, int2 range,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	const void* data, int lg2_scale,
	int thread_col_base, int thread_row_base,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;
	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;

	int pix_x[THREAD_X];
#pragma unroll
	for (int j = 0; j < THREAD_X; j++)
		pix_x[j] = tile_origin_x + thread_col_base + j;

	int pix_y[THREAD_Y];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
		pix_y[i] = tile_origin_y + thread_row_base + i;

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			T[i][j] = 1.0f;

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			C[i][j] = { 0.0f, 0.0f, 0.0f };

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;

		// Determine which gaussian this lane loads features for
		int gi = lane_gauss_index_3g(lane);

		// Initial load: feature lanes load from point_list[point_id + 0/1/2],
		// ID lanes prefetch point_list indices [point_id + 3/4/5]
		if (lane == 0)
			offset = point_id + 3;
		else if (lane == 3)
			offset = point_id + 4;
		else if (lane == 19)
			offset = point_id + 5;
		else if (gi == 0 && point_id + 0 < range.y)
			offset = point_list[point_id + 0];
		else if (gi == 1 && point_id + 1 < range.y)
			offset = point_list[point_id + 1];
		else if (gi == 2 && point_id + 2 < range.y)
			offset = point_list[point_id + 2];

		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
			load_enable = load_enable && point_id + 3 < range.y;
		else if (lane == 3)
			load_enable = load_enable && point_id + 4 < range.y;
		else if (lane == 19)
			load_enable = load_enable && point_id + 5 < range.y;
		else if (gi == 0)
			load_enable = load_enable && point_id + 0 < range.y;
		else if (gi == 1)
			load_enable = load_enable && point_id + 1 < range.y;
		else if (gi == 2)
			load_enable = load_enable && point_id + 2 < range.y;
		else
			load_enable = false;

		if (load_enable)
			buf = load_lane_value(data, lg2_scale, offset);

		load_enable = data != nullptr;
		bool done = false;

		// Fast loop: process 3 gaussians per iteration (no bounds check)
		// Needs point_id + 8 < range.y to guarantee 3 current + 3 prefetch + 3 next prefetch
		while (__any_sync(~0, point_id + 8 < range.y && !done))
		{
			// Get gaussian IDs from prefetched lanes
			// Lanes 0,3,19 have the IDs for point_id+3, +4, +5
			// Feature lanes need IDs for point_id+0, +1, +2 (already loaded)
			// After processing, we prefetch point_id+6, +7, +8

			// Resolve IDs: feature lanes get IDs from their respective prefetch lane
			int id0 = __shfl_sync(~0, __float_as_uint(buf), 0);   // point_list idx +3
			int id1 = __shfl_sync(~0, __float_as_uint(buf), 3);   // point_list idx +4
			int id2 = __shfl_sync(~0, __float_as_uint(buf), 19);  // point_list idx +5

			// Prepare next prefetch: ID lanes load point_id+6,+7,+8
			if (lane == 0) offset = point_id + 6;
			else if (lane == 3) offset = point_id + 7;
			else if (lane == 19) offset = point_id + 8;
			else if (gi == 0) offset = id0;
			else if (gi == 1) offset = id1;
			else if (gi == 2) offset = id2;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			// Process G0
			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			// Process G1
			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			// Process G2
			get_gaussian_features_g2(xy, rgb, con_o, buf);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;

			point_id += 3;
			buf = ldg_buf;
		}

		// Tail: process remaining gaussians (at most 8).
		// After the fast loop, buf contains valid features for point_id+0/1/2
		// and prefetched IDs for point_id+3/4/5.
		// We process in 3-gauss batches with bounds checking.
		while (__any_sync(~0, point_id < range.y && !done))
		{
			// Process G0 if available
			if (point_id < range.y)
			{
				get_gaussian_features(xy, rgb, con_o, buf, 0);
				pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
					C, T, tile_origin_x, tile_origin_y,
					thread_col_base, thread_row_base, xy, con_o, rgb);
			}

			// Process G1 if available
			if (point_id + 1 < range.y)
			{
				get_gaussian_features(xy, rgb, con_o, buf, 4);
				pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
					C, T, tile_origin_x, tile_origin_y,
					thread_col_base, thread_row_base, xy, con_o, rgb);
			}

			// Process G2 if available
			if (point_id + 2 < range.y)
			{
				get_gaussian_features_g2(xy, rgb, con_o, buf);
				pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
					C, T, tile_origin_x, tile_origin_y,
					thread_col_base, thread_row_base, xy, con_o, rgb);
			}

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;

			point_id += 3;

			// Load next batch from prefetched IDs if needed
			if (__any_sync(~0, point_id < range.y && !done))
			{
				int id0 = __shfl_sync(~0, __float_as_uint(buf), 0);
				int id1 = __shfl_sync(~0, __float_as_uint(buf), 3);
				int id2 = __shfl_sync(~0, __float_as_uint(buf), 19);

				// For ID lanes: prefetch next batch IDs
				if (lane == 0)
					offset = point_id + 3;
				else if (lane == 3)
					offset = point_id + 4;
				else if (lane == 19)
					offset = point_id + 5;
				else if (gi == 0)
					offset = id0;
				else if (gi == 1)
					offset = id1;
				else if (gi == 2)
					offset = id2;

				load_enable = data != nullptr;
				if (lane == 0)
					load_enable = load_enable && point_id + 3 < range.y;
				else if (lane == 3)
					load_enable = load_enable && point_id + 4 < range.y;
				else if (lane == 19)
					load_enable = load_enable && point_id + 5 < range.y;
				else if (gi == 0)
					load_enable = load_enable && point_id < range.y;
				else if (gi == 1)
					load_enable = load_enable && point_id + 1 < range.y;
				else if (gi == 2)
					load_enable = load_enable && point_id + 2 < range.y;
				else
					load_enable = false;

				if (load_enable)
					buf = load_lane_value(data, lg2_scale, offset);
			}
		}

#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color, { pix_x[j], pix_y[i] }, width, height, C[i][j], T[i][j]);
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix_x[j] < width && pix_y[i] < height)
				{
					int pix_id = width * pix_y[i] + pix_x[j];
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
	}
}

template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32, 28)
void renderCUDA16x16ReorderedPersistentV5(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	int total_tiles,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int4* __restrict__ tile_desc,
	render_load_info_3g info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;

	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	int idx = (int)blockIdx.x;

	int4 desc = make_int4(0, 0, 0, 0);
	if (idx < total_tiles && lane == 0)
		desc = __ldg(&tile_desc[idx]);
	desc.x = __shfl_sync(~0, desc.x, 0);
	desc.y = __shfl_sync(~0, desc.y, 0);
	desc.z = __shfl_sync(~0, desc.z, 0);
	desc.w = __shfl_sync(~0, desc.w, 0);

	while (idx < total_tiles)
	{
		int tile_origin_x = desc.x * BLOCK_X;
		int tile_origin_y = desc.y * BLOCK_Y;
		int2 range = { desc.z, desc.w };

		renderTileReorderedV5<THREAD_X, THREAD_Y>(
			tile_origin_x, tile_origin_y, range,
			point_list, width, height,
			data, lg2_scale,
			thread_col_base, thread_row_base,
			bg_color, out_color);

		if (lane == 0)
			idx = (int)atomicAdd(&g_tile_counter, 1) + (int)gridDim.x;
		idx = __shfl_sync(~0, idx, 0);

		desc = make_int4(0, 0, 0, 0);
		if (idx < total_tiles && lane == 0)
			desc = __ldg(&tile_desc[idx]);
		desc.x = __shfl_sync(~0, desc.x, 0);
		desc.y = __shfl_sync(~0, desc.y, 0);
		desc.z = __shfl_sync(~0, desc.z, 0);
		desc.w = __shfl_sync(~0, desc.w, 0);
	}
}

} // namespace
void render_16x16(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
    render<16, 16>(num_rendered, width, height, points_xy, rgb_depth, conic_opacity,
	    gaussian_keys_sorted, gaussian_values_sorted, bg_color, out_color, stream);
}

void render_16x16_preranges(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	renderCUDA_preranges<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid, dim3(8, 4, 1), 0, stream>>>(
		gaussian_values_sorted,
		width, height, grid.x,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_ranges_buf,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_16x16_preranges_smem(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	renderCUDA_preranges_smem<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid, dim3(8, 4, 1), 0, stream>>>(
		gaussian_values_sorted,
		width, height, grid.x,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_ranges_buf,
		bg_color,
		out_color);
}

void render_16x16_preranges_smem_persistent(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
		renderCUDA_preranges_smem_persistent<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>,
		32, sizeof(float2) * 64 + sizeof(float4) * 128);
	int grid_size = min(num_sms * max_active_blocks, total_tiles);

	// Reset tile counter
	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	renderCUDA_preranges_smem_persistent<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid_size, dim3(8, 4, 1), 0, stream>>>(
		gaussian_values_sorted,
		width, height, total_tiles,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_desc_buf,
		bg_color,
		out_color);
}

void render_16x16_preranges_smem_persistent_lite(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
		renderCUDA_preranges_smem_persistent_lite<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>,
		32, sizeof(float2) * 64 + sizeof(float4) * 128);
	int grid_size = min(num_sms * max_active_blocks, total_tiles);

	// Reset tile counter
	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	renderCUDA_preranges_smem_persistent_lite<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid_size, dim3(8, 4, 1), 0, stream>>>(
		gaussian_values_sorted,
		width, height,
		x_blocks, total_tiles,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_ranges_buf,
		bg_color,
		out_color);
}

// Dynamic Thresholding variant: skips ex2 for Gaussians with power < log2(1/255).
void render_16x16_preranges_smem_persistent_lite_dt(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
		renderCUDA_preranges_smem_persistent_lite_dt<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>,
		32, sizeof(float2) * 64 + sizeof(float4) * 128);
	int grid_size = min(num_sms * max_active_blocks, total_tiles);

	// Reset tile counter
	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	renderCUDA_preranges_smem_persistent_lite_dt<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid_size, dim3(8, 4, 1), 0, stream>>>(
		gaussian_values_sorted,
		width, height,
		x_blocks, total_tiles,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_ranges_buf,
		bg_color,
		out_color);
}

// E2E fused variant: computes tile_ranges inline via binary search (no separate
// precompute_tile_ranges kernel). Eliminates ~26us of pipeline overhead.
void render_16x16_preranges_smem_persistent_lite_fused(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
		renderCUDA_preranges_smem_persistent_lite_fused<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>,
		32, sizeof(float2) * 64 + sizeof(float4) * 128);
	int grid_size = min(num_sms * max_active_blocks, total_tiles);

	// Reset tile counter
	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	renderCUDA_preranges_smem_persistent_lite_fused<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid_size, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height,
		x_blocks, total_tiles,
		points_xy,
		rgb_depth,
		conic_opacity,
		bg_color,
		out_color);
}

void render_16x16_preranges_naive(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	renderCUDA_preranges_naive<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid, dim3(8, 4, 1), 0, stream>>>(
		gaussian_values_sorted,
		width, height, grid.x,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_ranges_buf,
		bg_color,
		out_color);
}

void render_16x16_preranges_smem_v2(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	renderCUDA_preranges_smem_v2<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, grid.x,
		points_xy,
		rgb_depth,
		conic_opacity,
		bg_color,
		out_color);
}

void render_16x16_topk_smem(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	const float2* topk_xy, const float4* topk_rgb_depth, const float4* topk_conic,
	int num_topk,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	// Non-persistent kernel with packed topk: standard 1-warp-per-tile launch,
	// using topk compact arrays via L2 cache instead of shared memory.
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	renderCUDA<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4, false, true><<<grid, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, grid.x,
		points_xy,
		rgb_depth,
		conic_opacity,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity,
			topk_xy, topk_rgb_depth, topk_conic),
		bg_color,
		out_color);
}

void render_16x16_topk_smem_persistent(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	const float2* topk_xy, const float4* topk_rgb_depth, const float4* topk_conic,
	int num_topk,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;
	constexpr int WPB = POLYSPLAT_TOPK_SMEM_PERSISTENT_WPB;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int grid_size = min(num_sms, (total_tiles + WPB - 1) / WPB);

	// Reset tile counter
	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	int topk_smem_bytes = topk_async_smem_bytes_required(num_topk);
	cudaFuncSetAttribute(
		renderCUDA16x16TopKSmem<BLOCK_X / 8, BLOCK_Y / 4, WPB>,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		topk_smem_bytes);

	renderCUDA16x16TopKSmem<BLOCK_X / 8, BLOCK_Y / 4, WPB>
		<<<grid_size, dim3(8, 4, WPB), topk_smem_bytes, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, x_blocks, total_tiles,
		points_xy,
		rgb_depth,
		conic_opacity,
		topk_xy, topk_rgb_depth, topk_conic,
		num_topk,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_16x16_topk_smem_persistent_v2(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	const float2* topk_xy, const float4* topk_rgb_depth, const float4* topk_conic,
	int num_topk,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	// With 1 warp per block and __launch_bounds__(32), the HW can schedule
	// up to 32 blocks per SM. We launch num_sms * max_blocks_per_sm blocks.
	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
		renderCUDA16x16TopKSmemV2<BLOCK_X / 8, BLOCK_Y / 4>,
		32, 0);
	int grid_size = min(num_sms * max_active_blocks, total_tiles);

	// Reset tile counter
	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	renderCUDA16x16TopKSmemV2<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid_size, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, x_blocks, total_tiles,
		points_xy,
		rgb_depth,
		conic_opacity,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void precompute_tile_ranges(int num_rendered,
	int width, int height, int block_x, int block_y,
	uint64_t* gaussian_keys_sorted,
	int2* tile_ranges_buf, cudaStream_t stream)
{
	int x_blocks = (width + block_x - 1) / block_x;
	int y_blocks = (height + block_y - 1) / block_y;
	int total_tiles = x_blocks * y_blocks;
	int threads = 256;
	int blocks = (total_tiles + threads - 1) / threads;
	computeTileRanges<<<blocks, threads, 0, stream>>>(
		gaussian_keys_sorted, num_rendered, total_tiles, tile_ranges_buf);
}

void precompute_tile_ranges_scan(int num_rendered,
	int width, int height, int block_x, int block_y,
	uint64_t* gaussian_keys_sorted,
	int2* tile_ranges_buf, cudaStream_t stream)
{
	int x_blocks = (width + block_x - 1) / block_x;
	int y_blocks = (height + block_y - 1) / block_y;
	int total_tiles = x_blocks * y_blocks;

	// Zero-initialize tile_ranges (tiles with no gaussians stay {0,0})
	cudaMemsetAsync(tile_ranges_buf, 0, total_tiles * sizeof(int2), stream);

	if (num_rendered > 0)
	{
		int threads = 256;
		int blocks = (num_rendered + threads - 1) / threads;
		computeTileRangesScan<<<blocks, threads, 0, stream>>>(
			gaussian_keys_sorted, num_rendered, tile_ranges_buf);
	}
}

void render_16x16_persistent_v3(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
		renderCUDA16x16PersistentV3<BLOCK_X / 8, BLOCK_Y / 4>,
		32, 0);
	int grid_size = min(num_sms * max_active_blocks, total_tiles);

	// Reset tile counter
	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	renderCUDA16x16PersistentV3<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid_size, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, x_blocks, total_tiles,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_ranges_buf,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_16x16_unroll2(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
    render<16, 16, true>(num_rendered, width, height, points_xy, rgb_depth, conic_opacity,
	    gaussian_keys_sorted, gaussian_values_sorted, bg_color, out_color, stream);
}

void render_16x16_split(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	render_split<16, 16, 2>(num_rendered, width, height, points_xy, rgb_depth, conic_opacity,
		gaussian_keys_sorted, gaussian_values_sorted, bg_color, out_color, stream);
}

void render_24x16_split(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	render_split<24, 16, 2>(num_rendered, width, height, points_xy, rgb_depth, conic_opacity,
		gaussian_keys_sorted, gaussian_values_sorted, bg_color, out_color, stream);
}

void render_32x16(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
    render<32, 16>(num_rendered, width, height, points_xy, rgb_depth, conic_opacity,
	    gaussian_keys_sorted, gaussian_values_sorted, bg_color, out_color, stream);
}

void render_32x16_split(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	dim3 grid((width + 31) / 32, (height + 15) / 16, 1);

	renderCUDA32x16Split<<<grid, dim3(16, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, grid.x,
		points_xy,
		rgb_depth,
		conic_opacity,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_32x32(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
    render<32, 32>(num_rendered, width, height, points_xy, rgb_depth, conic_opacity,
	    gaussian_keys_sorted, gaussian_values_sorted, bg_color, out_color, stream);
}


void compute_tile_order(
	int2* tile_ranges_buf,
	int total_tiles,
	uint32_t* tile_order,
	uint32_t* tile_counts_buf,
	uint32_t* tile_ids_buf,
	char* sort_temp,
	size_t sort_temp_bytes,
	cudaStream_t stream)
{
	int threads = 256;
	int blocks = (total_tiles + threads - 1) / threads;
	extractTileCounts<<<blocks, threads, 0, stream>>>(
		tile_ranges_buf, total_tiles, tile_counts_buf, tile_ids_buf);

	cub::DeviceRadixSort::SortPairsDescending(
		sort_temp, sort_temp_bytes,
		tile_counts_buf, tile_counts_buf,
		tile_ids_buf, tile_order,
		total_tiles, 0, 32, stream);
}

void compute_tile_order_packed(
	int2* tile_ranges_buf,
	int total_tiles,
	int x_blocks,
	int4* tile_desc_buf,
	uint32_t* tile_order,
	uint32_t* tile_counts_buf,
	uint32_t* tile_ids_buf,
	char* sort_temp,
	size_t sort_temp_bytes,
	int zigzag_group_size,  // 0 disables zigzag (pure descending)
	cudaStream_t stream)
{
	int threads = 256;
	int blocks = (total_tiles + threads - 1) / threads;
	extractTileCounts<<<blocks, threads, 0, stream>>>(
		tile_ranges_buf, total_tiles, tile_counts_buf, tile_ids_buf);

	cub::DeviceRadixSort::SortPairsDescending(
		sort_temp, sort_temp_bytes,
		tile_counts_buf, tile_counts_buf,
		tile_ids_buf, tile_order,
		total_tiles, 0, 32, stream);

	if (zigzag_group_size > 1)
	{
		gatherTileDescZigzag<<<blocks, threads, 0, stream>>>(
			tile_order, tile_ranges_buf, total_tiles, x_blocks,
			zigzag_group_size, tile_desc_buf);
	}
	else
	{
		gatherTileDesc<<<blocks, threads, 0, stream>>>(
			tile_order, tile_ranges_buf, total_tiles, x_blocks, tile_desc_buf);
	}
}

void compute_tile_order_packed_interleaved(
	int2* tile_ranges_buf,
	int total_tiles,
	int x_blocks,
	int4* tile_desc_buf,
	uint32_t* tile_order,
	uint32_t* tile_counts_buf,
	uint32_t* tile_ids_buf,
	char* sort_temp,
	size_t sort_temp_bytes,
	int zigzag_group_size,
	int num_sms,
	int blocks_per_sm,
	cudaStream_t stream)
{
	int threads = 256;
	int blocks = (total_tiles + threads - 1) / threads;
	extractTileCounts<<<blocks, threads, 0, stream>>>(
		tile_ranges_buf, total_tiles, tile_counts_buf, tile_ids_buf);

	cub::DeviceRadixSort::SortPairsDescending(
		sort_temp, sort_temp_bytes,
		tile_counts_buf, tile_counts_buf,
		tile_ids_buf, tile_order,
		total_tiles, 0, 32, stream);

	gatherTileDescInterleaved<<<blocks, threads, 0, stream>>>(
		tile_order, tile_ranges_buf, total_tiles, x_blocks,
		num_sms, blocks_per_sm, zigzag_group_size,
		tile_desc_buf);
}

void compute_tile_order_packed_morton(
	int2* tile_ranges_buf,
	int total_tiles,
	int x_blocks,
	int4* tile_desc_buf,
	uint32_t* tile_order,
	uint32_t* tile_counts_buf,
	uint32_t* tile_ids_buf,
	char* sort_temp,
	size_t sort_temp_bytes,
	int zigzag_group_size,
	int morton_bucket_size,
	cudaStream_t stream)
{
	int threads = 256;
	int blocks = (total_tiles + threads - 1) / threads;
	extractTileCountsMorton<<<blocks, threads, 0, stream>>>(
		tile_ranges_buf, total_tiles, x_blocks,
		morton_bucket_size,
		tile_counts_buf, tile_ids_buf);

	cub::DeviceRadixSort::SortPairsDescending(
		sort_temp, sort_temp_bytes,
		tile_counts_buf, tile_counts_buf,
		tile_ids_buf, tile_order,
		total_tiles, 0, 32, stream);

	if (zigzag_group_size > 1)
	{
		gatherTileDescZigzag<<<blocks, threads, 0, stream>>>(
			tile_order, tile_ranges_buf, total_tiles, x_blocks,
			zigzag_group_size, tile_desc_buf);
	}
	else
	{
		gatherTileDesc<<<blocks, threads, 0, stream>>>(
			tile_order, tile_ranges_buf, total_tiles, x_blocks, tile_desc_buf);
	}
}

size_t get_tile_order_sort_temp_size(int total_tiles)
{
	size_t temp_bytes = 0;
	cub::DeviceRadixSort::SortPairsDescending(
		nullptr, temp_bytes,
		(uint32_t*)nullptr, (uint32_t*)nullptr,
		(uint32_t*)nullptr, (uint32_t*)nullptr,
		total_tiles, 0, 32, (cudaStream_t)0);
	return temp_bytes;
}

void render_16x16_reordered(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf, uint32_t* tile_order,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	renderCUDA16x16Reordered<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, grid.x,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_ranges_buf,
		tile_order,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_16x16_reordered_v2(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	renderCUDA16x16ReorderedV2<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_desc_buf,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_16x16_reordered_v3(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	renderCUDA16x16ReorderedV3<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_desc_buf,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_16x16_reordered_persistent(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
		renderCUDA16x16ReorderedPersistent<BLOCK_X / 8, BLOCK_Y / 4>,
		32, 0);
	int grid_size = min(num_sms * max_active_blocks, total_tiles);

	// Reset tile counter
	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	renderCUDA16x16ReorderedPersistent<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid_size, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, total_tiles,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_desc_buf,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_16x16_reordered_persistent_nosort(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	int tile_order_mode,
	int zigzag_group_size,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	// For morton modes, compute the total number of morton cells
	int morton_total = 0;
	if (tile_order_mode >= 2)
	{
		int max_dim = x_blocks > y_blocks ? x_blocks : y_blocks;
		morton_total = 1;
		while (morton_total < max_dim) morton_total <<= 1;
		morton_total = morton_total * morton_total;
	}

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
		renderCUDA16x16ReorderedPersistentNoSort<BLOCK_X / 8, BLOCK_Y / 4>,
		32, 0);
	// For morton modes, grid_size uses morton_total (need to cover all morton cells)
	int counter_total = (tile_order_mode >= 2) ? morton_total : total_tiles;
	int grid_size = min(num_sms * max_active_blocks, counter_total);

	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	renderCUDA16x16ReorderedPersistentNoSort<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid_size, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, x_blocks, y_blocks, total_tiles,
		tile_order_mode,
		zigzag_group_size,
		morton_total,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_ranges_buf,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_16x16_reordered_persistent_v3(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
		renderCUDA16x16ReorderedPersistentV3<BLOCK_X / 8, BLOCK_Y / 4>,
		32, 0);
	int grid_size = min(num_sms * max_active_blocks, total_tiles);

	// Reset tile counter
	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	renderCUDA16x16ReorderedPersistentV3<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid_size, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, total_tiles,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_desc_buf,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_16x16_reordered_persistent_v5(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
		renderCUDA16x16ReorderedPersistentV5<BLOCK_X / 8, BLOCK_Y / 4>,
		32, 0);
	int grid_size = min(num_sms * max_active_blocks, total_tiles);

	// Reset tile counter
	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	renderCUDA16x16ReorderedPersistentV5<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid_size, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, total_tiles,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_desc_buf,
		render_load_info_3g(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_16x16_reordered_persistent_packed(int num_rendered,
	int width, int height,
	float* packed_features,
	float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;

	int x_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	int y_blocks = (height + BLOCK_Y - 1) / BLOCK_Y;
	int total_tiles = x_blocks * y_blocks;

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
		renderCUDA16x16ReorderedPersistent<BLOCK_X / 8, BLOCK_Y / 4>,
		32, 0);
	int grid_size = min(num_sms * max_active_blocks, total_tiles);

	unsigned int zero = 0;
	cudaMemcpyToSymbolAsync(g_tile_counter, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);

	renderCUDA16x16ReorderedPersistent<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid_size, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height, total_tiles,
		(float2*)nullptr,
		(float4*)nullptr,
		(float4*)nullptr,
		tile_desc_buf,
		render_load_info(gaussian_values_sorted, packed_features, conic_opacity),
		bg_color,
		out_color);
}

// V4a: optimized pixel shader (REGRESSED - __fmaf_rn conflicts with fast_math)
// V4b: V3 shader + max_T done check instead of AND chain
// V4c: V3 shader + NO done check in fast loop (saves __any_sync cost)
// ============================================================================

// V4b: max_T done check only
template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA16x16ReorderedV4b(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int4* __restrict__ tile_desc,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;

	uint32_t linear_idx = blockIdx.y * (uint32_t)gridDim.x + blockIdx.x;
	int4 desc = __ldg(&tile_desc[linear_idx]);
	int tile_origin_x = desc.x * BLOCK_X;
	int tile_origin_y = desc.y * BLOCK_Y;
	int2 range = { desc.z, desc.w };

	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			T[i][j] = 1.0f;

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			C[i][j] = { 0.0f, 0.0f, 0.0f };

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
			offset = point_id + 2;
		else if (lane == 4)
			offset = point_id + 3;
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
			offset = point_list[point_id + 0];
		else if (point_id + 1 < range.y)
			offset = point_list[point_id + 1];

		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
			load_enable = load_enable && point_id + 2 < range.y;
		else if (lane == 4)
			load_enable = load_enable && point_id + 3 < range.y;
		else if ((lane & 4) == 0)
			load_enable = load_enable && point_id + 0 < range.y;
		else
			load_enable = load_enable && point_id + 1 < range.y;

		if (load_enable)
			buf = load_lane_value(data, lg2_scale, offset);

		load_enable = data != nullptr;

		bool done = false;

		// ---- 4-gaussian fast loop: max_T done check ----
		while (__any_sync(~0, point_id + 9 < range.y && !done))
		{
			// Pair A
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;
			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);
			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);
			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);
			point_id += 2;
			buf = ldg_buf;

			// Pair B
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);
			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);
			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			// max_T done check (single float reduction vs 8-way AND)
			float max_T = 0.0f;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					max_T = fmaxf(max_T, T[i][j]);
			done = max_T < 0.0001f;

			point_id += 2;
			buf = ldg_buf;
		}

		// ---- 2-gaussian loop for remainder ----
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			float max_T = 0.0f;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					max_T = fmaxf(max_T, T[i][j]);
			done = max_T < 0.0001f;

			point_id += 2;
			buf = ldg_buf;
		}

		// ---- Tail loop with bounds checking ----
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			if (lane == 0)
				load_enable = load_enable && point_id + 4 < range.y;
			else if (lane == 4)
				load_enable = load_enable && point_id + 5 < range.y;
			else if ((lane & 4) == 0)
				load_enable = load_enable && point_id + 2 < range.y;
			else
				load_enable = load_enable && point_id + 3 < range.y;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			if (point_id + 1 >= range.y) break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			float max_T = 0.0f;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					max_T = fmaxf(max_T, T[i][j]);
			done = max_T < 0.0001f;
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color,
					{ tile_origin_x + thread_col_base + j,
					  tile_origin_y + thread_row_base + i },
					width, height, C[i][j], T[i][j]);
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int px = tile_origin_x + thread_col_base + j;
				int py = tile_origin_y + thread_row_base + i;
				if (px < width && py < height)
				{
					int pix_id = width * py + px;
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
	}
}

// V4c: NO done check in fast loop — saves __any_sync cost
// Only the 2-gaussian remainder and tail loops check done
template<int THREAD_X, int THREAD_Y>
__global__ __launch_bounds__(32) void renderCUDA16x16ReorderedV4c(
	const uint64_t* __restrict__ sorted_keys,
	int num_rendered,
	const uint32_t* __restrict__ point_list,
	int width, int height,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int4* __restrict__ tile_desc,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	constexpr int BLOCK_X = 16;
	constexpr int BLOCK_Y = 16;

	uint32_t linear_idx = blockIdx.y * (uint32_t)gridDim.x + blockIdx.x;
	int4 desc = __ldg(&tile_desc[linear_idx]);
	int tile_origin_x = desc.x * BLOCK_X;
	int tile_origin_y = desc.y * BLOCK_Y;
	int2 range = { desc.z, desc.w };

	int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
	int thread_col_base = (int)threadIdx.x * THREAD_X;
	int thread_row_base = (int)threadIdx.y * THREAD_Y;

	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			T[i][j] = 1.0f;

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
			C[i][j] = { 0.0f, 0.0f, 0.0f };

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
			offset = point_id + 2;
		else if (lane == 4)
			offset = point_id + 3;
		else if ((lane & 4) == 0 && point_id + 0 < range.y)
			offset = point_list[point_id + 0];
		else if (point_id + 1 < range.y)
			offset = point_list[point_id + 1];

		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
			load_enable = load_enable && point_id + 2 < range.y;
		else if (lane == 4)
			load_enable = load_enable && point_id + 3 < range.y;
		else if ((lane & 4) == 0)
			load_enable = load_enable && point_id + 0 < range.y;
		else
			load_enable = load_enable && point_id + 1 < range.y;

		if (load_enable)
			buf = load_lane_value(data, lg2_scale, offset);

		load_enable = data != nullptr;

		// ---- 4-gaussian fast loop: NO done check ----
		// The done check (__any_sync) costs ~16 cycles per iteration.
		// Most pixels don't saturate until near the end of the gaussian list,
		// so the done check rarely fires. We remove it entirely and rely on
		// the bounds check (point_id + 5 < range.y) for loop termination.
		while (point_id + 5 < range.y)
		{
			// Pair A
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;
			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);
			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);
			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);
			point_id += 2;
			buf = ldg_buf;
		}

		// ---- Tail loop with bounds and done checking ----
		bool done = false;
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0) offset = point_id + 4;
			if (lane == 4) offset = point_id + 5;

			if (lane == 0)
				load_enable = load_enable && point_id + 4 < range.y;
			else if (lane == 4)
				load_enable = load_enable && point_id + 5 < range.y;
			else if ((lane & 4) == 0)
				load_enable = load_enable && point_id + 2 < range.y;
			else
				load_enable = load_enable && point_id + 3 < range.y;

			float ldg_buf;
			if (load_enable)
				ldg_buf = load_lane_value(data, lg2_scale, offset);

			get_gaussian_features(xy, rgb, con_o, buf, 0);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			if (point_id + 1 >= range.y) break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
			pixel_shader_axis_shared_16x16<THREAD_X, THREAD_Y>(
				C, T, tile_origin_x, tile_origin_y,
				thread_col_base, thread_row_base, xy, con_o, rgb);

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
					done = done && T[i][j] < 0.0001f;
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
				write_color(out_color, bg_color,
					{ tile_origin_x + thread_col_base + j,
					  tile_origin_y + thread_row_base + i },
					width, height, C[i][j], T[i][j]);
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int px = tile_origin_x + thread_col_base + j;
				int py = tile_origin_y + thread_row_base + i;
				if (px < width && py < height)
				{
					int pix_id = width * py + px;
					out_color[pix_id].x = encode(bg_color.x);
					out_color[pix_id].y = encode(bg_color.y);
					out_color[pix_id].z = encode(bg_color.z);
				}
			}
	}
}

void render_16x16_reordered_v4(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	renderCUDA16x16ReorderedV4b<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_desc_buf,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void render_16x16_reordered_v4c(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	constexpr int BLOCK_X = 16, BLOCK_Y = 16;
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	renderCUDA16x16ReorderedV4c<BLOCK_X / 8, BLOCK_Y / 4>
		<<<grid, dim3(8, 4, 1), 0, stream>>>(
		gaussian_keys_sorted,
		num_rendered,
		gaussian_values_sorted,
		width, height,
		points_xy,
		rgb_depth,
		conic_opacity,
		tile_desc_buf,
		render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
		bg_color,
		out_color);
}

void compute_tile_order_packed_hilbert(
	int2* tile_ranges_buf,
	int total_tiles,
	int x_blocks,
	int y_blocks,
	int4* tile_desc_buf,
	uint32_t* tile_order,
	uint32_t* tile_counts_buf,
	uint32_t* tile_ids_buf,
	char* sort_temp,
	size_t sort_temp_bytes,
	int zigzag_group_size,
	int hilbert_bucket_size,
	cudaStream_t stream)
{
	int threads = 256;
	int blocks = (total_tiles + threads - 1) / threads;
	extractTileCountsHilbert<<<blocks, threads, 0, stream>>>(
		tile_ranges_buf, total_tiles, x_blocks, y_blocks,
		hilbert_bucket_size,
		tile_counts_buf, tile_ids_buf);

	cub::DeviceRadixSort::SortPairsDescending(
		sort_temp, sort_temp_bytes,
		tile_counts_buf, tile_counts_buf,
		tile_ids_buf, tile_order,
		total_tiles, 0, 32, stream);

	if (zigzag_group_size > 1)
	{
		gatherTileDescZigzag<<<blocks, threads, 0, stream>>>(
			tile_order, tile_ranges_buf, total_tiles, x_blocks,
			zigzag_group_size, tile_desc_buf);
	}
	else
	{
		gatherTileDesc<<<blocks, threads, 0, stream>>>(
			tile_order, tile_ranges_buf, total_tiles, x_blocks, tile_desc_buf);
	}
}

void regather_gaussians(
	int4* tile_desc_buf,
	int total_tiles,
	const uint32_t* new_offsets,
	const uint32_t* old_point_list,
	const float2* old_xy, const float4* old_rgb, const float4* old_conic,
	uint32_t* new_point_list,
	float2* new_xy, float4* new_rgb, float4* new_conic,
	cudaStream_t stream)
{
	// Launch one block per tile, 256 threads per block
	regatherGaussianData<<<total_tiles, 256, 0, stream>>>(
		tile_desc_buf, new_offsets, total_tiles,
		old_point_list,
		old_xy, old_rgb, old_conic,
		new_point_list,
		new_xy, new_rgb, new_conic);

	// Update tile_desc with new ranges
	int threads = 256;
	int blocks = (total_tiles + threads - 1) / threads;
	computeNewTileOffsets<<<blocks, threads, 0, stream>>>(
		tile_desc_buf, new_offsets, total_tiles);
}

} // namespace polysplat
