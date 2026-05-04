#include "../ops.h"

#ifndef CUDA_VERSION
#define CUDA_VERSION 8000
#endif

#define GLM_FORCE_CUDA
#include "../glm/glm.hpp"

namespace polysplat {
namespace {

constexpr float log2e = 1.4426950216293334961f;
constexpr float ln2 = 0.69314718055f;

// Unified adaptive culling parameters (Balanced profile)
constexpr float K_CUTOFF = 8.0f;              // E1: reverted to baseline value (was 5.0)
constexpr float TAU = 0.03125f;               // 2^{-K} = 1/32
constexpr float RHO = 2.0f;                   // penetration factor
constexpr float LN_RHO = 0.6931472f;          // ln(ρ) = ln(2)
constexpr float B_MIN_SQ = 0.09f;             // b_min² = 0.3²
constexpr float O_THIN = 0.5f;                // opacity cap for thin-Gaussian culling

// Spherical harmonics coefficients (compile-time constants for PTX immediate embedding)
#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f
#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 (-1.0925484305920792f)
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 (-1.0925484305920792f)
#define SH_C2_4 0.5462742152960396f
#define SH_C3_0 (-0.5900435899266435f)
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 (-0.4570457994644658f)
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 (-0.4570457994644658f)
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 (-0.5900435899266435f)

__forceinline__ __device__ float fast_max_f32(float a, float b)
{
	float d;
	asm volatile("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
	return d;
}

__forceinline__ __device__ float fast_sqrt_f32(float x)
{
	float y;
	asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}

__forceinline__ __device__ float fast_rsqrt_f32(float x)
{
	float y;
	asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}

__forceinline__ __device__ float fast_lg2_f32(float x)
{
	float y;
	asm volatile("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}

__forceinline__ __device__ int warp_reduce_max(int v)
{
	for (int d = POLYSPLAT_WARP_SIZE >> 1; d > 0; d >>= 1)
	{
		int n = __shfl_xor_sync(~0u, v, d);
		v = max(v, n);
	}
	return v;
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float3 transformPoint4x3(const glm::vec3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const glm::vec3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ void getRect(const float2 p, int width, int height, int2& rect_min, int2& rect_max, dim3 grid, int block_x, int block_y)
{
	rect_min = {
		min((int)grid.x, max((int)0, (int)((p.x - width) / (float)block_x))),
		min((int)grid.y, max((int)0, (int)((p.y - height) / (float)block_y)))
	};
	rect_max = {
		min((int)grid.x, max((int)0, (int)((p.x + width) / (float)block_x) + 1)),
		min((int)grid.y, max((int)0, (int)((p.y + height) / (float)block_y) + 1))
	};
}

// Forward version of 2D covariance matrix computation
__forceinline__ __device__ float3 computeCov2D(const glm::vec3& position, float focal_x, float focal_y, float tan_fovx, float tan_fovy,
	cov3d_t cov3D, glm::mat4 viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(position, (float*)&viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

	glm::mat3 W = glm::mat3(
		((float*)&viewmatrix)[0], ((float*)&viewmatrix)[4], ((float*)&viewmatrix)[8],
		((float*)&viewmatrix)[1], ((float*)&viewmatrix)[5], ((float*)&viewmatrix)[9],
		((float*)&viewmatrix)[2], ((float*)&viewmatrix)[6], ((float*)&viewmatrix)[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D.s[0], cov3D.s[1], cov3D.s[2],
		cov3D.s[1], cov3D.s[3], cov3D.s[4],
		cov3D.s[2], cov3D.s[4], cov3D.s[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__forceinline__ __device__ glm::vec3 computeColorFromSH(int idx, glm::vec3 p_orig, glm::vec3 campos, const shs_deg3_t* shs)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 dir = p_orig - campos;
	float l2 = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
	float rsqrt_l2 = fast_rsqrt_f32(l2);
	dir *= rsqrt_l2;

	auto sh = ((const shs_deg3_t*)shs)[idx];
	glm::vec3 result = SH_C0 * sh.v3[0] += 0.5f;

	float x = dir.x;
	float y = dir.y;
	float z = dir.z;
	result = result - SH_C1 * y * sh.v3[1] + SH_C1 * z * sh.v3[2] - SH_C1 * x * sh.v3[3];

	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, yz = y * z, xz = x * z;
	result = result +
		SH_C2_0 * xy * sh.v3[4] +
		SH_C2_1 * yz * sh.v3[5] +
		SH_C2_2 * (2.0f * zz - xx - yy) * sh.v3[6] +
		SH_C2_3 * xz * sh.v3[7] +
		SH_C2_4 * (xx - yy) * sh.v3[8];

	result = result +
		SH_C3_0 * y * (3.0f * xx - yy) * sh.v3[9] +
		SH_C3_1 * xy * z * sh.v3[10] +
		SH_C3_2 * y * (4.0f * zz - xx - yy) * sh.v3[11] +
		SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh.v3[12] +
		SH_C3_4 * x * (4.0f * zz - xx - yy) * sh.v3[13] +
		SH_C3_5 * z * (xx - yy) * sh.v3[14] +
		SH_C3_6 * x * (xx - 3.0f * yy) * sh.v3[15];

	result.x = fast_max_f32(result.x, 0.0f);
	result.y = fast_max_f32(result.y, 0.0f);
	result.z = fast_max_f32(result.z, 0.0f);
	return result;
}

// FP16 variant: loads half-precision SH coefficients with on-the-fly conversion.
// Halves SH memory traffic (96B vs 192B per gaussian) with negligible quality loss.
// Uses vectorized float4 loads (6 × 16B) instead of 48 individual __ldg calls.
__forceinline__ __device__ glm::vec3 computeColorFromSH_half(int idx, glm::vec3 p_orig, glm::vec3 campos, const __half* shs)
{
	glm::vec3 dir = p_orig - campos;
	float l2 = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
	float rsqrt_l2 = fast_rsqrt_f32(l2);
	dir *= rsqrt_l2;

	// Vectorized load: 96 bytes as 6 × float4 (6 load instructions, not 48)
	const float4* base4 = (const float4*)(shs + idx * 48);
	float4 raw[6];
	raw[0] = __ldg(&base4[0]);
	raw[1] = __ldg(&base4[1]);
	raw[2] = __ldg(&base4[2]);
	raw[3] = __ldg(&base4[3]);
	raw[4] = __ldg(&base4[4]);
	raw[5] = __ldg(&base4[5]);
	// Reinterpret as 48 halfs in registers
	const __half* h = (const __half*)raw;

	// Convert on-demand via register access (no extra memory loads)
	#define SH_V3(k) glm::vec3(__half2float(h[(k)*3+0]), __half2float(h[(k)*3+1]), __half2float(h[(k)*3+2]))

	glm::vec3 result = SH_C0 * SH_V3(0) += 0.5f;

	float x = dir.x, y = dir.y, z = dir.z;
	result = result - SH_C1 * y * SH_V3(1) + SH_C1 * z * SH_V3(2) - SH_C1 * x * SH_V3(3);

	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, yz = y * z, xz = x * z;
	result = result +
		SH_C2_0 * xy * SH_V3(4) +
		SH_C2_1 * yz * SH_V3(5) +
		SH_C2_2 * (2.0f * zz - xx - yy) * SH_V3(6) +
		SH_C2_3 * xz * SH_V3(7) +
		SH_C2_4 * (xx - yy) * SH_V3(8);

	result = result +
		SH_C3_0 * y * (3.0f * xx - yy) * SH_V3(9) +
		SH_C3_1 * xy * z * SH_V3(10) +
		SH_C3_2 * y * (4.0f * zz - xx - yy) * SH_V3(11) +
		SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * SH_V3(12) +
		SH_C3_4 * x * (4.0f * zz - xx - yy) * SH_V3(13) +
		SH_C3_5 * z * (xx - yy) * SH_V3(14) +
		SH_C3_6 * x * (xx - 3.0f * yy) * SH_V3(15);

	#undef SH_V3

	result.x = fast_max_f32(result.x, 0.0f);
	result.y = fast_max_f32(result.y, 0.0f);
	result.z = fast_max_f32(result.z, 0.0f);
	return result;
}

__forceinline__ __device__ bool segment_intersect_ellipse(float a, float b, float c, float d, float l, float r)
{
	float delta = b * b - 4.0f * a * c;
	// return delta >= 0.0f && t1 <= sqrt(delta) && t2 >= -sqrt(delta)
	float t1 = (l - d) * (2.0f * a) + b;
	float t2 = (r - d) * (2.0f * a) + b;
	return delta >= 0.0f && (t1 <= 0.0f || t1 * t1 <= delta) && (t2 >= 0.0f || t2 * t2 <= delta);
}

__forceinline__ __device__ bool block_intersect_ellipse(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power)
{
	float a, b, c, dx, dy;
	float w = 2.0f * power;

	if (center.x * 2.0f < pix_min.x + pix_max.x)
	{
		dx = center.x - pix_min.x;
	}
	else
	{
		dx = center.x - pix_max.x;
	}
	a = conic.z;
	b = -2.0f * conic.y * dx;
	c = conic.x * dx * dx - w;

	if (segment_intersect_ellipse(a, b, c, center.y, pix_min.y, pix_max.y))
	{
		return true;
	}

	if (center.y * 2.0f < pix_min.y + pix_max.y)
	{
		dy = center.y - pix_min.y;
	}
	else
	{
		dy = center.y - pix_max.y;
	}
	a = conic.x;
	b = -2.0f * conic.y * dy;
	c = conic.z * dy * dy - w;

	if (segment_intersect_ellipse(a, b, c, center.x, pix_min.x, pix_max.x))
	{
		return true;
	}

	return false;
}

__forceinline__ __device__ bool block_contains_center(int2 pix_min, int2 pix_max, float2 center)
{
	return center.x >= pix_min.x && center.x <= pix_max.x && center.y >= pix_min.y && center.y <= pix_max.y;
}

__global__ void preprocessCUDA(
	int P,
	const glm::vec3* __restrict__ positions,
	const float* __restrict__ opacities,
	const void* __restrict__ shs,
	bool shs_half,
	glm::mat4 viewmatrix,
	glm::mat4 projmatrix,
	glm::vec3 cam_position,
	const int W, int H,
	int block_x, int block_y,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	float2* __restrict__ points_xy,
	cov3d_t* __restrict__ cov3Ds,
	float4* __restrict__ rgb_depth,
	float4* __restrict__ conic_opacity,
	float4* __restrict__ packed_features,  // cache-line-aligned packed layout (16 floats per gaussian)
	int* __restrict__ curr_offset,
	uint64_t* __restrict__ gaussian_keys_unsorted,
	uint32_t* __restrict__ gaussian_values_unsorted,
	const dim3 grid)
{
	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	int warp_id = blockIdx.x * blockDim.z + threadIdx.z;
	int idx_vec = warp_id * POLYSPLAT_WARP_SIZE + lane;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	bool point_valid = false;
	glm::vec3 p_orig;
	int width = 0;
	int height = 0;
	float3 p_view;
	float2 point_xy;
	float3 conic;
	float opacity;
	float power;
	float log2_opacity;
	int2 rect_min;
	int2 rect_max;
		if (idx_vec < P)
		{
			do {
				// 用 do-while(false) 包一层，便于在任意裁剪条件触发时直接 break 提前退出。
				// 取出该 Gaussian 的 3D 中心位置，并先变换到相机/视图坐标系。
				p_orig = positions[idx_vec];
				p_view = transformPoint4x3(p_orig, (float*)&viewmatrix);
				// 近裁剪：离相机过近或落在相机近平面内的点直接丢弃，
				// 否则后续的透视投影和屏幕空间协方差会变得很不稳定。
				if (p_view.z <= 0.2f)
					break;
				opacity = opacities[idx_vec];
				// Level 1a: 透明度裁剪 — opacity < τ = 2^{-K} 时，
				// 对所有像素 α(p) ≤ o < τ，且 P < 0 导致 AABB 无物理意义。
				// E3: TAU opacity prune disabled
				// if (opacity < TAU)
				// 	break;

				// 将 3D 中心点乘投影矩阵，得到齐次裁剪空间坐标；
				// 再做透视除法，得到 NDC 空间坐标 p_proj。
	            float4 p_hom = transformPoint4x4(p_orig, (float*)&projmatrix);
	            float p_w = 1.0f / (p_hom.w + 0.0000001f);
				float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

				// 将 3D Gaussian 的协方差投影到屏幕空间，
				// 得到 2D 协方差矩阵 [cov.x cov.y; cov.y cov.z]。
				float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3Ds[idx_vec], viewmatrix);

				// 对 2D 协方差求逆，转成椭圆二次型参数 conic。
				// 后续 EWA splatting / rasterization 会用它来快速评估
				// 某个像素相对该 Gaussian 中心的衰减权重。
				float det = (cov.x * cov.z - cov.y * cov.y);
				float det_inv = 1.f / det;
				conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

				// 根据透明度估计该 Gaussian 在屏幕上的“有效影响范围”。
				// power 越大，说明需要保留的有效覆盖区域越大。
				log2_opacity = fast_lg2_f32(opacity);
				power = ln2 * K_CUTOFF + ln2 * log2_opacity;

				// Level 1b: 有效半短轴裁剪 — 又细又淡的 Gaussian 浪费率极高。
				// 利用 λ_min ≥ det(Σ)/tr(Σ) 下界，避免特征值分解。
				float trace = cov.x + cov.z;
				if (2.0f * det * power < B_MIN_SQ * trace && opacity < O_THIN)
					break;
				// 用协方差主对角近似估计屏幕空间包围盒的半宽/半高，
				// 得到一个保守的 axis-aligned bounding box（像素单位）。
				width = (int)(1.414214f * fast_sqrt_f32(cov.x * power) + 1.0f);
				height = (int)(1.414214f * fast_sqrt_f32(cov.z * power) + 1.0f);
				//TODO：根据协方差的旋转，计算更紧的包围盒。
				// 如果包围盒太大了，说明这个 Gaussian 会覆盖很多 tile。为了避免后续过多的 key/value 对生成和排序，
				//TODO: a-s codesign，这个包围框的设计没有考虑不透明度，其实这里设计框的时候就可以有意的对于更透明的，设计更紧的包围，甚至可以小于椭圆自身！！

				// 将 NDC 坐标映射到像素坐标。
				point_xy = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
				// 根据中心点和包围盒半径，计算它可能覆盖到哪些 tile。
				getRect(point_xy, width, height, rect_min, rect_max, grid, block_x, block_y);
				// 只要覆盖矩形非空，就认为这个点后续值得继续参与 binning / rasterization。
				point_valid = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) > 0;
			} while (false);
		}

	bool single_tile = point_valid && (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 1;
	if (single_tile)
	{
		int2 pix_min = { rect_min.x * block_x, rect_min.y * block_y };
		int2 pix_max = { pix_min.x + block_x - 1, pix_min.y + block_y - 1 };
		// Level 3: Per-tile valid check with penetration factor ρ
		float power_valid = power /* LN_RHO removed — see docs/correctness_investigation_20260419.md */;
		bool valid = block_contains_center(pix_min, pix_max, point_xy) ||
			(power_valid > 0.0f &&
			 block_intersect_ellipse(pix_min, pix_max, point_xy, conic, power_valid));
		if (valid)
		{
			uint64_t key = rect_min.y * grid.x + rect_min.x;
			key <<= 32;
			key |= __float_as_uint(p_view.z);
			int offset = atomicAdd(curr_offset, 1);
			gaussian_keys_unsorted[offset] = key;
			gaussian_values_unsorted[offset] = idx_vec;
		}
		point_valid = false;
	}

	// Generate no key/value pair for invisible Gaussians
	int my_tile_count = point_valid ? (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) : 0;
	int multi_tiles = __ballot_sync(~0u, point_valid);
	int multi_count = __popc(multi_tiles);
	int max_tile_count = warp_reduce_max(my_tile_count);
	bool vertex_valid = single_tile;
	if (multi_count <= 2 || max_tile_count >= 32)
	{
		while (multi_tiles)
		{
			int i = __ffs(multi_tiles) - 1;
			multi_tiles &= multi_tiles - 1;
			// Find this Gaussian's offset in buffer for writing keys/values.
			float2 my_point_xy = {
				__shfl_sync(~0u, point_xy.x, i),
				__shfl_sync(~0u, point_xy.y, i)
			};
			float3 my_conic = {
				__shfl_sync(~0u, conic.x, i),
				__shfl_sync(~0u, conic.y, i),
				__shfl_sync(~0u, conic.z, i),
			};
			int2 my_rect_min = {
				__shfl_sync(~0u, rect_min.x, i),
				__shfl_sync(~0u, rect_min.y, i)
			};
			int2 my_rect_max = {
				__shfl_sync(~0u, rect_max.x, i),
				__shfl_sync(~0u, rect_max.y, i)
			};
			float my_depth = __shfl_sync(~0u, p_view.z, i);
			float my_power = __shfl_sync(~0u, power, i);
			int idx = warp_id * POLYSPLAT_WARP_SIZE + i;

			// For each tile that the bounding rect overlaps, emit a
			// key/value pair. The key is |  tile ID  |      depth      |,
			// and the value is the ID of the Gaussian. Sorting the values
			// with this key yields Gaussian IDs in a list, such that they
			// are first sorted by tile and then by depth.
			for (int y0 = my_rect_min.y; y0 < my_rect_max.y; y0 += blockDim.y)
			{
				int y = y0 + threadIdx.y;
				for (int x0 = my_rect_min.x; x0 < my_rect_max.x; x0 += blockDim.x)
				{
					int x = x0 + threadIdx.x;
					bool valid = y < my_rect_max.y && x < my_rect_max.x;

					if (valid)
					{
						int2 pix_min = { x * block_x, y * block_y };
						int2 pix_max = { pix_min.x + block_x - 1, pix_min.y + block_y - 1 };
						// Level 3: Per-tile valid check with penetration factor ρ
						float my_power_valid = my_power /* LN_RHO removed — see docs/correctness_investigation_20260419.md */;
						valid = block_contains_center(pix_min, pix_max, my_point_xy) ||
							(my_power_valid > 0.0f &&
							 block_intersect_ellipse(pix_min, pix_max, my_point_xy, my_conic, my_power_valid));
					}

					int mask = __ballot_sync(~0u, valid);
					if (mask == 0)
					{
						continue;
					}
					int my_offset;
					if (lane == 0)
					{
						my_offset = atomicAdd(curr_offset, __popc(mask));
					}
					vertex_valid = vertex_valid || i == lane;
					int count = __popc(mask & ((1u << lane) - 1u));
					uint64_t key = y * grid.x + x;
					key <<= 32;
					key |= __float_as_uint(my_depth);
					my_offset = __shfl_sync(~0u, my_offset, 0);
					if (valid)
					{
						gaussian_keys_unsorted[my_offset + count] = key;
						gaussian_values_unsorted[my_offset + count] = idx;
					}
				}
			}
		}
	}
	else
	{
		int prefix = my_tile_count;
		for (int d = 1; d < POLYSPLAT_WARP_SIZE; d <<= 1)
		{
			int n = __shfl_up_sync(~0u, prefix, d);
			if (lane >= d)
			{
				prefix += n;
			}
		}
		int total_tasks = __shfl_sync(~0u, prefix, POLYSPLAT_WARP_SIZE - 1);
		prefix -= my_tile_count;
		vertex_valid = vertex_valid || point_valid;
		int default_owner = __ffs(multi_tiles) - 1;

		// Flatten the warp's small multi-tile gaussians into a single task stream.
		for (int base = 0; base < total_tasks; base += POLYSPLAT_WARP_SIZE)
		{
			int task_id = base + lane;
			bool task_valid = task_id < total_tasks;
			int owner = default_owner;
			unsigned int bits = multi_tiles;
			while (bits)
			{
				int candidate = __ffs((int)bits) - 1;
				int cand_prefix = __shfl_sync(~0u, prefix, candidate);
				int cand_count = __shfl_sync(~0u, my_tile_count, candidate);
				if (task_valid && task_id >= cand_prefix && task_id < cand_prefix + cand_count)
				{
					owner = candidate;
				}
				bits &= bits - 1;
			}

			int o_rect_min_x = __shfl_sync(~0u, rect_min.x, owner);
			int o_rect_min_y = __shfl_sync(~0u, rect_min.y, owner);
			int o_rect_max_x = __shfl_sync(~0u, rect_max.x, owner);
			int o_prefix = __shfl_sync(~0u, prefix, owner);

			int rect_w = o_rect_max_x - o_rect_min_x;
			int local_off = task_id - o_prefix;
			int tile_y = 0;
			int rem = local_off;
			if (task_valid)
			{
				while (rem >= rect_w)
				{
					rem -= rect_w;
					tile_y++;
				}
			}
			int tile_x = rem;
			int x = o_rect_min_x + tile_x;
			int y = o_rect_min_y + tile_y;

			float2 o_point_xy = {
				__shfl_sync(~0u, point_xy.x, owner),
				__shfl_sync(~0u, point_xy.y, owner)
			};
			float3 o_conic = {
				__shfl_sync(~0u, conic.x, owner),
				__shfl_sync(~0u, conic.y, owner),
				__shfl_sync(~0u, conic.z, owner)
			};
			float o_power = __shfl_sync(~0u, power, owner);
			float o_depth = __shfl_sync(~0u, p_view.z, owner);
			int o_idx = warp_id * POLYSPLAT_WARP_SIZE + owner;

			bool valid = false;
			if (task_valid)
			{
				int2 pix_min = { x * block_x, y * block_y };
				int2 pix_max = { pix_min.x + block_x - 1, pix_min.y + block_y - 1 };
				// Level 3: Per-tile valid check with penetration factor ρ
				float o_power_valid = o_power /* LN_RHO removed — see docs/correctness_investigation_20260419.md */;
				valid = block_contains_center(pix_min, pix_max, o_point_xy) ||
					(o_power_valid > 0.0f &&
					 block_intersect_ellipse(pix_min, pix_max, o_point_xy, o_conic, o_power_valid));
			}

			int mask = __ballot_sync(~0u, valid);
			if (mask == 0)
			{
				continue;
			}
			int my_offset;
			if (lane == 0)
			{
				my_offset = atomicAdd(curr_offset, __popc(mask));
			}
			my_offset = __shfl_sync(~0u, my_offset, 0);
			int count = __popc(mask & ((1u << lane) - 1u));
			if (valid)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= __float_as_uint(o_depth);
				gaussian_keys_unsorted[my_offset + count] = key;
				gaussian_values_unsorted[my_offset + count] = o_idx;
			}
		}
	}
	if (vertex_valid)
	{
		points_xy[idx_vec] = point_xy;
		float4 con_scaled = { (-0.5f * log2e) * conic.x, -log2e * conic.y, (-0.5f * log2e) * conic.z, log2_opacity };
		conic_opacity[idx_vec] = con_scaled;
		auto color = shs_half
			? computeColorFromSH_half(idx_vec, p_orig, cam_position, (const __half*)shs)
			: computeColorFromSH(idx_vec, p_orig, cam_position, (const shs_deg3_t*)shs);
		rgb_depth[idx_vec] = { color.r, color.g, color.b, p_view.z };
		// Write packed features (8 floats = 32B per gaussian, lg2_scale=5)
		if (packed_features)
		{
			int base = idx_vec * 2;
			packed_features[base + 0] = make_float4(point_xy.x, point_xy.y, color.r, color.g);
			packed_features[base + 1] = make_float4(color.b, con_scaled.x, con_scaled.y, con_scaled.z);
		}
	}
}

glm::mat4 getViewMatrix(glm::vec3 position, glm::mat3 rotation)
{
	return glm::mat4(
		glm::vec4(rotation[0], 0.0f),
		glm::vec4(rotation[1], 0.0f),
		glm::vec4(rotation[2], 0.0f),
		glm::vec4(rotation * -position, 1.0f));
}

glm::mat4 getProjectionMatrix(int width, int height, glm::vec3 position, glm::mat3 rotation, float focal_x, float focal_y, float zFar, float zNear)
{
	float top = height / (2.0f * focal_y) * zNear;
	float bottom = -top;
	float right = width / (2.0f * focal_x) * zNear;
	float left = -right;

	glm::mat4 P;
	memset(&P, 0, sizeof P);
	float z_sign = 1.0f;

	P[0][0] = 2.0f * zNear / (right - left);
	P[1][1] = 2.0f * zNear / (top - bottom);
	P[0][2] = (right + left) / (right - left);
	P[1][2] = (top + bottom) / (top - bottom);
	P[3][2] = z_sign;
	P[2][2] = z_sign * zFar / (zFar - zNear);
	P[2][3] = -(zFar * zNear) / (zFar - zNear);
	return glm::transpose(P) * getViewMatrix(position, rotation);
}

// =====================================================================
// ES Pass-A kernel (Early Sorting — count tiles per Gaussian, no emit)
//
// Identical culling / shape / ellipse logic as `preprocessCUDA`, BUT:
//   - skips atomicAdd(curr_offset, ...) and the 64-bit key emission.
//   - counts valid ellipse-tile pairs per Gaussian into `tiles_per_gauss[idx_vec]`.
//   - additionally stores metadata for Pass-B:
//       conic_power_raw[idx_vec]  = (conic.x, conic.y, conic.z, power)
//       depth_natural[idx_vec]    = p_view.z (FLT_MAX sentinel for invalid Gaussians)
//       rect_bounds[idx_vec]      = packed (rect_min_xy, rect_max_xy) as int2
//   - shape outputs (points_xy / conic_opacity / rgb_depth / packed_features) still
//     written identically to baseline so render kernel sees the same data.
//
// BIT-IDENTICAL GUARANTEE (critical): all ellipse checks run with the SAME
// (point_xy, conic, power, pix_min, pix_max) values as baseline — the only
// changes are replacing atomicAdd/key-write with atomic shmem counting and
// warp-local accumulation. This ensures Pass-A's count exactly equals the
// number of intersections baseline would have emitted.
// =====================================================================
__global__ void preprocessCUDA_ES_PassA(
	int P,
	const glm::vec3* __restrict__ positions,
	const float* __restrict__ opacities,
	const void* __restrict__ shs,
	bool shs_half,
	glm::mat4 viewmatrix,
	glm::mat4 projmatrix,
	glm::vec3 cam_position,
	const int W, int H,
	int block_x, int block_y,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	float2* __restrict__ points_xy,
	cov3d_t* __restrict__ cov3Ds,
	float4* __restrict__ rgb_depth,
	float4* __restrict__ conic_opacity,
	float4* __restrict__ packed_features,
	// ES-specific outputs:
	int32_t* __restrict__ tiles_per_gauss,
	float4* __restrict__ conic_power_raw,
	float* __restrict__ depth_natural,
	int2* __restrict__ rect_bounds,
	const dim3 grid)
{
	// Per-block shmem accumulator for the flattened-task branch. Each of the
	// 128 threads in the block (dim3(8,4,4)) gets its own slot — avoids
	// warp-level reductions that would need an owner-aware scan.
	__shared__ int s_flat_count[128];
	int local_tid = threadIdx.z * 32 + (threadIdx.y * blockDim.x + threadIdx.x);
	s_flat_count[local_tid] = 0;
	__syncthreads();

	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	int warp_id = blockIdx.x * blockDim.z + threadIdx.z;
	int idx_vec = warp_id * POLYSPLAT_WARP_SIZE + lane;

	bool point_valid = false;
	glm::vec3 p_orig;
	int width = 0;
	int height = 0;
	float3 p_view;
	float2 point_xy;
	float3 conic;
	float opacity;
	float power;
	float log2_opacity;
	int2 rect_min;
	int2 rect_max;

	if (idx_vec < P)
	{
		do {
			p_orig = positions[idx_vec];
			p_view = transformPoint4x3(p_orig, (float*)&viewmatrix);
			if (p_view.z <= 0.2f)
				break;
			opacity = opacities[idx_vec];

			float4 p_hom = transformPoint4x4(p_orig, (float*)&projmatrix);
			float p_w = 1.0f / (p_hom.w + 0.0000001f);
			float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

			float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3Ds[idx_vec], viewmatrix);

			float det = (cov.x * cov.z - cov.y * cov.y);
			float det_inv = 1.f / det;
			conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

			log2_opacity = fast_lg2_f32(opacity);
			power = ln2 * K_CUTOFF + ln2 * log2_opacity;

			float trace = cov.x + cov.z;
			if (2.0f * det * power < B_MIN_SQ * trace && opacity < O_THIN)
				break;
			width = (int)(1.414214f * fast_sqrt_f32(cov.x * power) + 1.0f);
			height = (int)(1.414214f * fast_sqrt_f32(cov.z * power) + 1.0f);

			point_xy = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
			getRect(point_xy, width, height, rect_min, rect_max, grid, block_x, block_y);
			point_valid = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) > 0;
		} while (false);
	}

	// my_count accumulates tile-counts for THIS lane's Gaussian (single-tile + multi-tile paths).
	int my_count = 0;

	// ---------- Single-tile path ----------
	bool single_tile = point_valid && (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 1;
	if (single_tile)
	{
		int2 pix_min = { rect_min.x * block_x, rect_min.y * block_y };
		int2 pix_max = { pix_min.x + block_x - 1, pix_min.y + block_y - 1 };
		float power_valid = power;
		bool valid = block_contains_center(pix_min, pix_max, point_xy) ||
			(power_valid > 0.0f &&
			 block_intersect_ellipse(pix_min, pix_max, point_xy, conic, power_valid));
		if (valid)
		{
			my_count = 1;
		}
		point_valid = false;
	}

	int my_tile_count = point_valid ? (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) : 0;
	int multi_tiles = __ballot_sync(~0u, point_valid);
	int multi_count = __popc(multi_tiles);
	int max_tile_count = warp_reduce_max(my_tile_count);
	bool vertex_valid = single_tile;

	// ---------- Multi-tile path (warp-coop, owner-per-iteration) ----------
	if (multi_count <= 2 || max_tile_count >= 32)
	{
		while (multi_tiles)
		{
			int i = __ffs(multi_tiles) - 1;
			multi_tiles &= multi_tiles - 1;
			float2 my_point_xy = {
				__shfl_sync(~0u, point_xy.x, i),
				__shfl_sync(~0u, point_xy.y, i)
			};
			float3 my_conic = {
				__shfl_sync(~0u, conic.x, i),
				__shfl_sync(~0u, conic.y, i),
				__shfl_sync(~0u, conic.z, i),
			};
			int2 my_rect_min = {
				__shfl_sync(~0u, rect_min.x, i),
				__shfl_sync(~0u, rect_min.y, i)
			};
			int2 my_rect_max = {
				__shfl_sync(~0u, rect_max.x, i),
				__shfl_sync(~0u, rect_max.y, i)
			};
			float my_power = __shfl_sync(~0u, power, i);

			// iter_sum is uniform across the warp (mask is warp-wide).
			int iter_sum = 0;
			for (int y0 = my_rect_min.y; y0 < my_rect_max.y; y0 += blockDim.y)
			{
				int y = y0 + threadIdx.y;
				for (int x0 = my_rect_min.x; x0 < my_rect_max.x; x0 += blockDim.x)
				{
					int x = x0 + threadIdx.x;
					bool valid = y < my_rect_max.y && x < my_rect_max.x;

					if (valid)
					{
						int2 pix_min = { x * block_x, y * block_y };
						int2 pix_max = { pix_min.x + block_x - 1, pix_min.y + block_y - 1 };
						float my_power_valid = my_power;
						valid = block_contains_center(pix_min, pix_max, my_point_xy) ||
							(my_power_valid > 0.0f &&
							 block_intersect_ellipse(pix_min, pix_max, my_point_xy, my_conic, my_power_valid));
					}

					int mask = __ballot_sync(~0u, valid);
					if (mask == 0)
					{
						continue;
					}
					iter_sum += __popc(mask);
					if (i == lane)
						vertex_valid = true;
				}
			}
			if (lane == i)
				my_count = iter_sum;
		}
	}
	// ---------- Flattened-task path (task pooling across warp Gaussians) ----------
	else
	{
		int prefix = my_tile_count;
		for (int d = 1; d < POLYSPLAT_WARP_SIZE; d <<= 1)
		{
			int n = __shfl_up_sync(~0u, prefix, d);
			if (lane >= d)
				prefix += n;
		}
		int total_tasks = __shfl_sync(~0u, prefix, POLYSPLAT_WARP_SIZE - 1);
		prefix -= my_tile_count;
		vertex_valid = vertex_valid || point_valid;
		int default_owner = __ffs(multi_tiles) - 1;

		for (int base = 0; base < total_tasks; base += POLYSPLAT_WARP_SIZE)
		{
			int task_id = base + lane;
			bool task_valid = task_id < total_tasks;
			int owner = default_owner;
			unsigned int bits = multi_tiles;
			while (bits)
			{
				int candidate = __ffs((int)bits) - 1;
				int cand_prefix = __shfl_sync(~0u, prefix, candidate);
				int cand_count = __shfl_sync(~0u, my_tile_count, candidate);
				if (task_valid && task_id >= cand_prefix && task_id < cand_prefix + cand_count)
				{
					owner = candidate;
				}
				bits &= bits - 1;
			}

			int o_rect_min_x = __shfl_sync(~0u, rect_min.x, owner);
			int o_rect_min_y = __shfl_sync(~0u, rect_min.y, owner);
			int o_rect_max_x = __shfl_sync(~0u, rect_max.x, owner);
			int o_prefix = __shfl_sync(~0u, prefix, owner);

			int rect_w = o_rect_max_x - o_rect_min_x;
			int local_off = task_id - o_prefix;
			int tile_y = 0;
			int rem = local_off;
			if (task_valid)
			{
				while (rem >= rect_w)
				{
					rem -= rect_w;
					tile_y++;
				}
			}
			int tile_x = rem;
			int x = o_rect_min_x + tile_x;
			int y = o_rect_min_y + tile_y;

			float2 o_point_xy = {
				__shfl_sync(~0u, point_xy.x, owner),
				__shfl_sync(~0u, point_xy.y, owner)
			};
			float3 o_conic = {
				__shfl_sync(~0u, conic.x, owner),
				__shfl_sync(~0u, conic.y, owner),
				__shfl_sync(~0u, conic.z, owner)
			};
			float o_power = __shfl_sync(~0u, power, owner);

			bool valid = false;
			if (task_valid)
			{
				int2 pix_min = { x * block_x, y * block_y };
				int2 pix_max = { pix_min.x + block_x - 1, pix_min.y + block_y - 1 };
				float o_power_valid = o_power;
				valid = block_contains_center(pix_min, pix_max, o_point_xy) ||
					(o_power_valid > 0.0f &&
					 block_intersect_ellipse(pix_min, pix_max, o_point_xy, o_conic, o_power_valid));
			}

			int mask = __ballot_sync(~0u, valid);
			if (mask == 0)
			{
				continue;
			}
			// Each valid task contributes 1 to its owner's count in shmem.
			if (valid)
			{
				atomicAdd(&s_flat_count[threadIdx.z * 32 + owner], 1);
			}
		}
	}

	__syncthreads();
	int flat_count = s_flat_count[local_tid];

	// Write outputs: shape (same as baseline) + ES metadata.
	if (idx_vec < P)
	{
		if (vertex_valid)
		{
			points_xy[idx_vec] = point_xy;
			float4 con_scaled = { (-0.5f * log2e) * conic.x, -log2e * conic.y, (-0.5f * log2e) * conic.z, log2_opacity };
			conic_opacity[idx_vec] = con_scaled;
			auto color = shs_half
				? computeColorFromSH_half(idx_vec, p_orig, cam_position, (const __half*)shs)
				: computeColorFromSH(idx_vec, p_orig, cam_position, (const shs_deg3_t*)shs);
			rgb_depth[idx_vec] = { color.r, color.g, color.b, p_view.z };
			if (packed_features)
			{
				int base = idx_vec * 2;
				packed_features[base + 0] = make_float4(point_xy.x, point_xy.y, color.r, color.g);
				packed_features[base + 1] = make_float4(color.b, con_scaled.x, con_scaled.y, con_scaled.z);
			}
			// ES metadata (raw conic + power for Pass-B ellipse re-check)
			conic_power_raw[idx_vec] = make_float4(conic.x, conic.y, conic.z, power);
			depth_natural[idx_vec] = p_view.z;
			int32_t rmin = (rect_min.x & 0xFFFF) | ((rect_min.y & 0xFFFF) << 16);
			int32_t rmax = (rect_max.x & 0xFFFF) | ((rect_max.y & 0xFFFF) << 16);
			rect_bounds[idx_vec] = make_int2(rmin, rmax);
		}
		else
		{
			// Invalid Gaussian: sentinel depth pushes it to the end of the sort;
			// tiles_per_gauss = 0 makes Pass-B skip it.
			depth_natural[idx_vec] = 3.4028234e38f;  // FLT_MAX
		}
		tiles_per_gauss[idx_vec] = my_count + flat_count;
	}
}

// =====================================================================
// ES Pass-B kernel — emit 32-bit tile keys in depth-sorted order.
//
// Each warp processes 32 consecutive depth-sorted Gaussians (via perm[]).
// Reuses Pass-A's metadata (conic_power_raw, rect_bounds) — NO projection /
// covariance / SH work here. Runs identical ellipse culling as Pass-A, so
// emission count per Gaussian MUST equal Pass-A's count (guaranteed by
// bit-identical determinism of block_intersect_ellipse given same inputs).
//
// Output key width: 32 bits (tile_id only). Depth ordering within each tile
// is preserved because:
//   1. Pass-B visits Gaussians in depth-sorted order (smallest depth first).
//   2. For a given Gaussian, it emits all its tile-intersection records at
//      contiguous offsets [base_offset, base_offset + count).
//   3. The final 32-bit radix sort on tile_id is stable (CUB guarantees),
//      so original order (= depth order) within each tile is preserved.
// =====================================================================
__global__ void preprocessCUDA_ES_PassB(
	int P,
	int block_x, int block_y,
	const uint32_t* __restrict__ perm,                     // [P] perm[s] = gauss_id at sorted pos s
	const int32_t* __restrict__ cum_offsets_sorted,        // [P] exclusive scan of tiles_per_gauss_sorted
	const int32_t* __restrict__ tiles_per_gauss_sorted,    // [P] = tiles_per_gauss[perm[s]]
	const float2* __restrict__ points_xy,                  // [P] natural order
	const float4* __restrict__ conic_power_raw,            // [P] natural order (conic.xyz, power)
	const int2* __restrict__ rect_bounds,                  // [P] natural order (packed rect_min, rect_max)
	uint32_t* __restrict__ tile_keys_unsorted,             // [M] output — 32-bit tile_ids
	uint32_t* __restrict__ gauss_values_unsorted,          // [M] output — gauss_ids
	const dim3 grid)
{
	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	int warp_id = blockIdx.x * blockDim.z + threadIdx.z;
	int sorted_idx = warp_id * POLYSPLAT_WARP_SIZE + lane;

	bool point_valid = false;
	uint32_t gid = 0;
	float2 point_xy = { 0.0f, 0.0f };
	float3 conic = { 0.0f, 0.0f, 0.0f };
	float power = 0.0f;
	int2 rect_min = { 0, 0 };
	int2 rect_max = { 0, 0 };
	int base_offset = 0;

	if (sorted_idx < P)
	{
		int tile_count = tiles_per_gauss_sorted[sorted_idx];
		if (tile_count > 0)
		{
			gid = perm[sorted_idx];
			base_offset = cum_offsets_sorted[sorted_idx];
			point_xy = points_xy[gid];
			float4 cp = conic_power_raw[gid];
			conic.x = cp.x; conic.y = cp.y; conic.z = cp.z;
			power = cp.w;
			int2 rb = rect_bounds[gid];
			// unpack as signed int16
			rect_min.x = (int)(int16_t)(rb.x & 0xFFFF);
			rect_min.y = (int)(int16_t)((rb.x >> 16) & 0xFFFF);
			rect_max.x = (int)(int16_t)(rb.y & 0xFFFF);
			rect_max.y = (int)(int16_t)((rb.y >> 16) & 0xFFFF);
			point_valid = true;
		}
	}

	// ---------- Single-tile path ----------
	bool single_tile = point_valid && (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 1;
	if (single_tile)
	{
		int2 pix_min = { rect_min.x * block_x, rect_min.y * block_y };
		int2 pix_max = { pix_min.x + block_x - 1, pix_min.y + block_y - 1 };
		float power_valid = power;
		bool valid = block_contains_center(pix_min, pix_max, point_xy) ||
			(power_valid > 0.0f &&
			 block_intersect_ellipse(pix_min, pix_max, point_xy, conic, power_valid));
		// Pass-A counted tile_count==1 for this Gaussian, so valid should be true here.
		// Emit unconditionally (if not valid, Pass-A would have had tile_count==0 and we'd skip via point_valid).
		if (valid)
		{
			uint32_t tile_id = (uint32_t)rect_min.y * grid.x + (uint32_t)rect_min.x;
			tile_keys_unsorted[base_offset] = tile_id;
			gauss_values_unsorted[base_offset] = gid;
		}
		point_valid = false;
	}

	int my_tile_count_rect = point_valid ? (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) : 0;
	int multi_tiles = __ballot_sync(~0u, point_valid);
	int multi_count = __popc(multi_tiles);
	int max_tile_count = warp_reduce_max(my_tile_count_rect);

	// ---------- Multi-tile path (warp-coop, per-Gaussian local cursor) ----------
	if (multi_count <= 2 || max_tile_count >= 32)
	{
		while (multi_tiles)
		{
			int i = __ffs(multi_tiles) - 1;
			multi_tiles &= multi_tiles - 1;
			float2 my_point_xy = {
				__shfl_sync(~0u, point_xy.x, i),
				__shfl_sync(~0u, point_xy.y, i)
			};
			float3 my_conic = {
				__shfl_sync(~0u, conic.x, i),
				__shfl_sync(~0u, conic.y, i),
				__shfl_sync(~0u, conic.z, i)
			};
			int2 my_rect_min = {
				__shfl_sync(~0u, rect_min.x, i),
				__shfl_sync(~0u, rect_min.y, i)
			};
			int2 my_rect_max = {
				__shfl_sync(~0u, rect_max.x, i),
				__shfl_sync(~0u, rect_max.y, i)
			};
			float my_power = __shfl_sync(~0u, power, i);
			int my_base_offset = __shfl_sync(~0u, base_offset, i);
			uint32_t my_gid = __shfl_sync(~0u, (int)gid, i);

			int local_cursor = 0;
			for (int y0 = my_rect_min.y; y0 < my_rect_max.y; y0 += blockDim.y)
			{
				int y = y0 + threadIdx.y;
				for (int x0 = my_rect_min.x; x0 < my_rect_max.x; x0 += blockDim.x)
				{
					int x = x0 + threadIdx.x;
					bool valid = y < my_rect_max.y && x < my_rect_max.x;
					if (valid)
					{
						int2 pix_min = { x * block_x, y * block_y };
						int2 pix_max = { pix_min.x + block_x - 1, pix_min.y + block_y - 1 };
						float my_power_valid = my_power;
						valid = block_contains_center(pix_min, pix_max, my_point_xy) ||
							(my_power_valid > 0.0f &&
							 block_intersect_ellipse(pix_min, pix_max, my_point_xy, my_conic, my_power_valid));
					}
					int mask = __ballot_sync(~0u, valid);
					if (mask == 0) continue;
					int count_in_mask = __popc(mask & ((1u << lane) - 1u));
					if (valid)
					{
						uint32_t tile_id = (uint32_t)y * grid.x + (uint32_t)x;
						int off = my_base_offset + local_cursor + count_in_mask;
						tile_keys_unsorted[off] = tile_id;
						gauss_values_unsorted[off] = my_gid;
					}
					local_cursor += __popc(mask);
				}
			}
		}
	}
	// ---------- Flattened-task path (per-owner cursor via shfl broadcast) ----------
	else
	{
		int prefix = my_tile_count_rect;
		for (int d = 1; d < POLYSPLAT_WARP_SIZE; d <<= 1)
		{
			int n = __shfl_up_sync(~0u, prefix, d);
			if (lane >= d) prefix += n;
		}
		int total_tasks = __shfl_sync(~0u, prefix, POLYSPLAT_WARP_SIZE - 1);
		prefix -= my_tile_count_rect;
		int default_owner = __ffs(multi_tiles) - 1;

		// Per-lane cursor for THIS lane's Gaussian (incremented when it's an owner).
		int my_owner_cursor = 0;

		for (int base = 0; base < total_tasks; base += POLYSPLAT_WARP_SIZE)
		{
			int task_id = base + lane;
			bool task_valid = task_id < total_tasks;
			int owner = default_owner;
			unsigned int bits = multi_tiles;
			while (bits)
			{
				int candidate = __ffs((int)bits) - 1;
				int cand_prefix = __shfl_sync(~0u, prefix, candidate);
				int cand_count = __shfl_sync(~0u, my_tile_count_rect, candidate);
				if (task_valid && task_id >= cand_prefix && task_id < cand_prefix + cand_count)
				{
					owner = candidate;
				}
				bits &= bits - 1;
			}

			int o_rect_min_x = __shfl_sync(~0u, rect_min.x, owner);
			int o_rect_min_y = __shfl_sync(~0u, rect_min.y, owner);
			int o_rect_max_x = __shfl_sync(~0u, rect_max.x, owner);
			int o_prefix = __shfl_sync(~0u, prefix, owner);

			int rect_w = o_rect_max_x - o_rect_min_x;
			int local_off = task_id - o_prefix;
			int tile_y = 0;
			int rem = local_off;
			if (task_valid)
			{
				while (rem >= rect_w)
				{
					rem -= rect_w;
					tile_y++;
				}
			}
			int tile_x = rem;
			int x = o_rect_min_x + tile_x;
			int y = o_rect_min_y + tile_y;

			float2 o_point_xy = {
				__shfl_sync(~0u, point_xy.x, owner),
				__shfl_sync(~0u, point_xy.y, owner)
			};
			float3 o_conic = {
				__shfl_sync(~0u, conic.x, owner),
				__shfl_sync(~0u, conic.y, owner),
				__shfl_sync(~0u, conic.z, owner)
			};
			float o_power = __shfl_sync(~0u, power, owner);

			bool valid = false;
			if (task_valid)
			{
				int2 pix_min = { x * block_x, y * block_y };
				int2 pix_max = { pix_min.x + block_x - 1, pix_min.y + block_y - 1 };
				float o_power_valid = o_power;
				valid = block_contains_center(pix_min, pix_max, o_point_xy) ||
					(o_power_valid > 0.0f &&
					 block_intersect_ellipse(pix_min, pix_max, o_point_xy, o_conic, o_power_valid));
			}

			int mask = __ballot_sync(~0u, valid);
			if (mask == 0) continue;

			// Emit per-candidate-owner: compute per-owner submask & offsets, emit in parallel.
			unsigned int cand_bits = multi_tiles;
			while (cand_bits)
			{
				int k = __ffs((int)cand_bits) - 1;
				cand_bits &= cand_bits - 1;

				int mask_k = __ballot_sync(~0u, valid && (owner == k));
				int count_k = __popc(mask_k);
				if (count_k == 0) continue;

				int o_base = __shfl_sync(~0u, base_offset, k);
				uint32_t o_gid = __shfl_sync(~0u, (int)gid, k);
				int o_cursor = __shfl_sync(~0u, my_owner_cursor, k);

				int lane_off = __popc(mask_k & ((1u << lane) - 1u));

				if (valid && owner == k)
				{
					uint32_t tile_id = (uint32_t)y * grid.x + (uint32_t)x;
					int off = o_base + o_cursor + lane_off;
					tile_keys_unsorted[off] = tile_id;
					gauss_values_unsorted[off] = o_gid;
				}

				if (lane == k) my_owner_cursor = o_cursor + count_k;
			}
		}
	}
}

} // namespace

void preprocess(int P,
	glm::vec3* positions, shs_deg3_t* shs, float* opacities, cov3d_t* cov3Ds,
	int width, int height, int block_x, int block_y,
	glm::vec3 cam_position, glm::mat3 cam_rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	int* curr_offset, float4* packed_features, cudaStream_t stream)
{
	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);

	glm::mat4 view_matrix = getViewMatrix(cam_position, cam_rotation);
	glm::mat4 proj_matrix = getProjectionMatrix(width, height, cam_position, cam_rotation, focal_x, focal_y, zFar, zNear);
	float tan_fovx = width / (2.0f * focal_x);
	float tan_fovy = height / (2.0f * focal_y);

	preprocessCUDA<<<(P + 127) / 128, dim3(8, 4, 4), 0, stream>>>(
		P,
		positions,
		opacities,
		(const void*)shs,
		false,
		view_matrix,
		proj_matrix,
		cam_position,
		width, height,
		block_x, block_y,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		points_xy,
		cov3Ds,
		rgb_depth,
		conic_opacity,
		packed_features,
		curr_offset,
		gaussian_keys_unsorted,
		gaussian_values_unsorted,
		grid);
}

void preprocess_half_sh(int P,
	glm::vec3* positions, __half* shs_half, float* opacities, cov3d_t* cov3Ds,
	int width, int height, int block_x, int block_y,
	glm::vec3 cam_position, glm::mat3 cam_rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	int* curr_offset, float4* packed_features, cudaStream_t stream)
{
	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);

	glm::mat4 view_matrix = getViewMatrix(cam_position, cam_rotation);
	glm::mat4 proj_matrix = getProjectionMatrix(width, height, cam_position, cam_rotation, focal_x, focal_y, zFar, zNear);
	float tan_fovx = width / (2.0f * focal_x);
	float tan_fovy = height / (2.0f * focal_y);

	preprocessCUDA<<<(P + 127) / 128, dim3(8, 4, 4), 0, stream>>>(
		P,
		positions,
		opacities,
		(const void*)shs_half,
		true,
		view_matrix,
		proj_matrix,
		cam_position,
		width, height,
		block_x, block_y,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		points_xy,
		cov3Ds,
		rgb_depth,
		conic_opacity,
		packed_features,
		curr_offset,
		gaussian_keys_unsorted,
		gaussian_values_unsorted,
		grid);
}

// =====================================================================
// ES Pass-A host wrapper: count tiles per Gaussian + store shape/metadata.
// No key emission — Pass-B (in depth-sorted order) emits 32-bit tile keys.
// =====================================================================
void preprocess_es_pass_a(int P,
	glm::vec3* positions, shs_deg3_t* shs, float* opacities, cov3d_t* cov3Ds,
	int width, int height, int block_x, int block_y,
	glm::vec3 cam_position, glm::mat3 cam_rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	int32_t* tiles_per_gauss, float4* conic_power_raw, float* depth_natural, int2* rect_bounds,
	float4* packed_features, cudaStream_t stream)
{
	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);

	glm::mat4 view_matrix = getViewMatrix(cam_position, cam_rotation);
	glm::mat4 proj_matrix = getProjectionMatrix(width, height, cam_position, cam_rotation, focal_x, focal_y, zFar, zNear);
	float tan_fovx = width / (2.0f * focal_x);
	float tan_fovy = height / (2.0f * focal_y);

	preprocessCUDA_ES_PassA<<<(P + 127) / 128, dim3(8, 4, 4), 0, stream>>>(
		P,
		positions,
		opacities,
		(const void*)shs,
		false,
		view_matrix,
		proj_matrix,
		cam_position,
		width, height,
		block_x, block_y,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		points_xy,
		cov3Ds,
		rgb_depth,
		conic_opacity,
		packed_features,
		tiles_per_gauss,
		conic_power_raw,
		depth_natural,
		rect_bounds,
		grid);
}

void preprocess_es_pass_a_half_sh(int P,
	glm::vec3* positions, __half* shs_half, float* opacities, cov3d_t* cov3Ds,
	int width, int height, int block_x, int block_y,
	glm::vec3 cam_position, glm::mat3 cam_rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	int32_t* tiles_per_gauss, float4* conic_power_raw, float* depth_natural, int2* rect_bounds,
	float4* packed_features, cudaStream_t stream)
{
	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);

	glm::mat4 view_matrix = getViewMatrix(cam_position, cam_rotation);
	glm::mat4 proj_matrix = getProjectionMatrix(width, height, cam_position, cam_rotation, focal_x, focal_y, zFar, zNear);
	float tan_fovx = width / (2.0f * focal_x);
	float tan_fovy = height / (2.0f * focal_y);

	preprocessCUDA_ES_PassA<<<(P + 127) / 128, dim3(8, 4, 4), 0, stream>>>(
		P,
		positions,
		opacities,
		(const void*)shs_half,
		true,
		view_matrix,
		proj_matrix,
		cam_position,
		width, height,
		block_x, block_y,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		points_xy,
		cov3Ds,
		rgb_depth,
		conic_opacity,
		packed_features,
		tiles_per_gauss,
		conic_power_raw,
		depth_natural,
		rect_bounds,
		grid);
}

// =====================================================================
// ES Pass-B host wrapper: emit 32-bit tile keys in depth-sorted order.
// =====================================================================
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
	cudaStream_t stream)
{
	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);
	preprocessCUDA_ES_PassB<<<(P + 127) / 128, dim3(8, 4, 4), 0, stream>>>(
		P,
		block_x, block_y,
		perm,
		cum_offsets_sorted,
		tiles_per_gauss_sorted,
		points_xy,
		conic_power_raw,
		rect_bounds,
		tile_keys_unsorted,
		gauss_values_unsorted,
		grid);
}

} // namespace polysplat
