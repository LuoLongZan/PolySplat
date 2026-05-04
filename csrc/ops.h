#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "glm/glm.hpp"

constexpr int POLYSPLAT_WARP_SIZE = 32;
constexpr int POLYSPLAT_TOPK_SMEM_GAUSSIANS = 1365;
constexpr int POLYSPLAT_TOPK_SMEM_FEATURE_FLOATS = 9;
constexpr int POLYSPLAT_TOPK_SMEM_BYTES =
	POLYSPLAT_TOPK_SMEM_GAUSSIANS * POLYSPLAT_TOPK_SMEM_FEATURE_FLOATS * sizeof(float);

// Packed point_list encoding for inline top-k slot:
//   bit 31      = is_topk flag
//   bits  0..30 = smem slot index (when flag=1) or gaussian_id (when flag=0)
constexpr uint32_t POLYSPLAT_TOPK_FLAG_BIT = 1u << 31;

#define POLYSPLAT_CHECK_CUDA(x)                                                                   \
	{                                                                                   \
		cudaError_t status = x;                                                         \
		if (status != cudaSuccess) {                                                    \
			fprintf(stderr, "%s\nline = %d\n", cudaGetErrorString(status), __LINE__);   \
			exit(1);                                                                    \
		}                                                                               \
	}

namespace polysplat {

union cov3d_t
{
    float2 f2[3];
    float s[6];
};

union shs_deg3_t
{
    float4 f4[12];
    glm::vec3 v3[16];
};

void preprocess(int P,
	glm::vec3* positions, shs_deg3_t* shs, float* opacities, cov3d_t* cov3Ds,
	int width, int height, int block_x, int block_y,
	glm::vec3 cam_position, glm::mat3 cam_rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	int* curr_offset, float4* packed_features = nullptr, cudaStream_t stream = 0);

void sort_gaussian(int num_rendered,
	int width, int height, int block_x, int block_y,
	char* list_sorting_space, size_t sorting_size,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted, cudaStream_t stream = 0);

size_t get_sort_buffer_size(int num_rendered, cudaStream_t stream = 0);

// =====================================================================
// ES orchestration entry points.
// =====================================================================
// Scan workspace sizing (DeviceScan::ExclusiveSum on int32[P]).
size_t get_es_scan_buffer_size(int P, cudaStream_t stream = 0);

// Step A: depth sort + gather tiles_per_gauss + exclusive scan.
//   tiles_per_gauss:            [P] input from Pass-A
//   depth_natural:              [P] input from Pass-A (FLT_MAX sentinel for invalid)
//   identity_buf:               [P] scratch (filled with 0..P-1)
//   perm_buf:                   [P] output: perm[s] = gauss_id at depth-sorted position s
//   depth_sorted_buf:           [P] scratch
//   tiles_per_gauss_sorted_buf: [P] output: tiles_per_gauss[perm[s]]
//   cum_offsets_sorted_buf:     [P] output: exclusive scan
//   total_num_rendered_out:     [1] device int — sum of tiles_per_gauss
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
	cudaStream_t stream = 0);

// Step B: 32-bit stable tile sort + rebuild 64-bit keys (tile << 32) for render kernel.
void es_tile_sort(
	int num_rendered,
	int width, int height, int block_x, int block_y,
	uint32_t* tile_keys_unsorted,
	uint32_t* gauss_values_unsorted,
	uint32_t* tile_keys_sorted,
	uint32_t* gauss_values_sorted,
	uint64_t* gaussian_keys_sorted_out,
	char* cub_sort_scratch, size_t cub_sort_scratch_bytes,
	cudaStream_t stream = 0);

// Async D2H read of curr_offset via pinned host buffer + non-blocking event.
// Lower host overhead than synchronous cudaMemcpy(D2H) on hot paths.
void fetch_num_rendered_async(const int* d_curr_offset, int* out_num_rendered, cudaStream_t stream = 0);

void gather_features(int num_rendered,
	uint32_t* sorted_values,
	float2* src_xy, float4* src_rgb_depth, float4* src_conic_opacity,
	float2* dst_xy, float4* dst_rgb_depth, float4* dst_conic_opacity,
	cudaStream_t stream = 0);

void preprocess_half_sh(int P,
	glm::vec3* positions, __half* shs_half, float* opacities, cov3d_t* cov3Ds,
	int width, int height, int block_x, int block_y,
	glm::vec3 cam_position, glm::mat3 cam_rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	int* curr_offset, float4* packed_features = nullptr, cudaStream_t stream = 0);

// ES Pass-A: count tiles per Gaussian + store metadata. No key emission.
// Outputs (same semantics as preprocess, with additions):
//   points_xy, rgb_depth, conic_opacity, packed_features — shape for render
//   tiles_per_gauss[P]       — per-Gaussian count of valid ellipse-tile pairs
//   conic_power_raw[P]       — float4(conic.x, conic.y, conic.z, power) for Pass-B
//   depth_natural[P]         — p_view.z (FLT_MAX for invalid Gaussians)
//   rect_bounds[P]           — int2(packed rect_min_xy, packed rect_max_xy)
void preprocess_es_pass_a(int P,
	glm::vec3* positions, shs_deg3_t* shs, float* opacities, cov3d_t* cov3Ds,
	int width, int height, int block_x, int block_y,
	glm::vec3 cam_position, glm::mat3 cam_rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	int32_t* tiles_per_gauss, float4* conic_power_raw, float* depth_natural, int2* rect_bounds,
	float4* packed_features = nullptr, cudaStream_t stream = 0);

void preprocess_es_pass_a_half_sh(int P,
	glm::vec3* positions, __half* shs_half, float* opacities, cov3d_t* cov3Ds,
	int width, int height, int block_x, int block_y,
	glm::vec3 cam_position, glm::mat3 cam_rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	int32_t* tiles_per_gauss, float4* conic_power_raw, float* depth_natural, int2* rect_bounds,
	float4* packed_features = nullptr, cudaStream_t stream = 0);

// ES Pass-B: emit 32-bit tile keys in depth-sorted order. Reads Pass-A metadata.
// `perm[s]` = original gauss_id at sorted position s (from depth sort).
// `cum_offsets_sorted[s]` = exclusive scan of tiles_per_gauss in sorted order.
// Output:
//   tile_keys_unsorted[M]       — 32-bit tile ids, order = depth-sorted emission
//   gauss_values_unsorted[M]    — corresponding gauss_ids
void preprocess_es_pass_b(int P,
	int width, int height, int block_x, int block_y,
	const uint32_t* perm,
	const int32_t* cum_offsets_sorted,
	const int32_t* tiles_per_gauss_sorted,
	const float2* points_xy,
	const float4* conic_power_raw,
	const int2* rect_bounds,
	uint32_t* tile_keys_unsorted,
	uint32_t* gauss_values_unsorted,
	cudaStream_t stream = 0);

void render_16x16(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_preranges(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_preranges_smem(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_preranges_smem_persistent(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_preranges_smem_persistent_lite(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

// Dynamic Thresholding variant: skips ex2 for power < log2(1/255).
void render_16x16_preranges_smem_persistent_lite_dt(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

// E2E fused variant: inline binary search per tile (no precomputed tile_ranges).
void render_16x16_preranges_smem_persistent_lite_fused(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_preranges_naive(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_preranges_smem_v2(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_topk_smem(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	const float2* topk_xy, const float4* topk_rgb_depth, const float4* topk_conic,
	int num_topk,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_topk_smem_persistent(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	const float2* topk_xy, const float4* topk_rgb_depth, const float4* topk_conic,
	int num_topk,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_topk_smem_persistent_v2(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	const float2* topk_xy, const float4* topk_rgb_depth, const float4* topk_conic,
	int num_topk,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void precompute_tile_ranges_scan(int num_rendered,
	int width, int height, int block_x, int block_y,
	uint64_t* gaussian_keys_sorted,
	int2* tile_ranges_buf, cudaStream_t stream = 0);

void precompute_tile_ranges(int num_rendered,
	int width, int height, int block_x, int block_y,
	uint64_t* gaussian_keys_sorted,
	int2* tile_ranges_buf, cudaStream_t stream = 0);

void compute_tile_order(
	int2* tile_ranges_buf,
	int total_tiles,
	uint32_t* tile_order,
	uint32_t* tile_counts_buf,
	uint32_t* tile_ids_buf,
	char* sort_temp,
	size_t sort_temp_bytes,
	cudaStream_t stream = 0);

size_t get_tile_order_sort_temp_size(int total_tiles);

void render_16x16_reordered(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* tile_ranges_buf, uint32_t* tile_order,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

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
	int zigzag_group_size = 0,
	cudaStream_t stream = 0);

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
	cudaStream_t stream = 0);

void render_16x16_reordered_v2(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_reordered_persistent(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int4* tile_desc_buf,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_unroll2(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_16x16_split(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_24x16_split(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_32x16(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_32x16_split(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

void render_32x32(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	float3 bg_color, uchar3* out_color, cudaStream_t stream = 0);

} // namespace polysplat
