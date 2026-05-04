#include "../ops.h"

namespace polysplat {
namespace {

__global__ void gatherGaussianFeatures(
	int num_rendered,
	const uint32_t* __restrict__ sorted_values,
	const float2* __restrict__ src_xy,
	const float4* __restrict__ src_rgb_depth,
	const float4* __restrict__ src_conic_opacity,
	float2* __restrict__ dst_xy,
	float4* __restrict__ dst_rgb_depth,
	float4* __restrict__ dst_conic_opacity)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_rendered) return;
	uint32_t idx = sorted_values[i];
	dst_xy[i] = __ldg(&src_xy[idx]);
	dst_rgb_depth[i] = __ldg(&src_rgb_depth[idx]);
	dst_conic_opacity[i] = __ldg(&src_conic_opacity[idx]);
}

// Pack gaussian features into a 32B-aligned struct (8 floats = 2 float4s).
// Layout: [xy.x, xy.y, r, g, b, con.x, con.y, con.z]
// con.w is kept in the original conic_opacity array
__global__ void packGaussianFeatures(
	int num_gaussians,
	const float2* __restrict__ src_xy,
	const float4* __restrict__ src_rgb_depth,
	const float4* __restrict__ src_conic_opacity,
	float4* __restrict__ dst_packed)  // num_gaussians * 2 float4s
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_gaussians) return;
	float2 xy = __ldg(&src_xy[i]);
	float4 rgb = __ldg(&src_rgb_depth[i]);
	float4 con = __ldg(&src_conic_opacity[i]);
	int base = i * 2;
	dst_packed[base + 0] = make_float4(xy.x, xy.y, rgb.x, rgb.y);
	dst_packed[base + 1] = make_float4(rgb.z, con.x, con.y, con.z);
}

} // namespace

void gather_features(int num_rendered,
	uint32_t* sorted_values,
	float2* src_xy, float4* src_rgb_depth, float4* src_conic_opacity,
	float2* dst_xy, float4* dst_rgb_depth, float4* dst_conic_opacity,
	cudaStream_t stream)
{
	if (num_rendered == 0) return;
	constexpr int BLOCK = 256;
	gatherGaussianFeatures<<<(num_rendered + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
		num_rendered, sorted_values,
		src_xy, src_rgb_depth, src_conic_opacity,
		dst_xy, dst_rgb_depth, dst_conic_opacity);
}

void pack_features(int num_gaussians,
	float2* src_xy, float4* src_rgb_depth, float4* src_conic_opacity,
	float4* dst_packed,
	cudaStream_t stream)
{
	if (num_gaussians == 0) return;
	constexpr int BLOCK = 256;
	packGaussianFeatures<<<(num_gaussians + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
		num_gaussians, src_xy, src_rgb_depth, src_conic_opacity, dst_packed);
}

} // namespace polysplat
