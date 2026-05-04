#include "ops.h"

#include <torch/extension.h>

#include <fstream>
#include <iostream>
#include <string>

namespace polysplat {
namespace {

struct VertexStorage
{
    glm::vec3 position;
    glm::vec3 normal;
    float shs[48];
    float opacity;
    glm::vec3 scale;
    glm::vec4 rotation;
};

void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
    // Create scaling matrix
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    // Normalize quaternion to get valid rotation
    glm::vec4 q = rot;// / glm::length(rot);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // Compute rotation matrix from quaternion
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;

    // Compute 3D world covariance matrix Sigma
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> loadPly_torch(const std::string& name)
{
    std::ifstream file(name, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error(std::string("Failed to open file: ") + name);
    }
    bool end_header = false;
    int numVertex = 0;
    std::string s;
    while (!file.eof())
    {
        file >> s;
        if (s == "vertex")
        {
            file >> numVertex;
            if (numVertex <= 0)
            {
                throw std::runtime_error("Vertex number is not positive");
            }
        }
        else if (s == "end_header")
        {
            end_header = true;
            file.get();
            break;
        }
    }
    if (!end_header)
    {
        throw std::runtime_error("Cannot find end of header");
    }
    torch::Device device(torch::kCPU);
    torch::TensorOptions options(torch::kFloat32);
    torch::Tensor positionTensor = torch::empty({numVertex,3}, options.device());
    torch::Tensor shsTensor = torch::empty({numVertex,48}, options.device(device));
    torch::Tensor opacityTensor = torch::empty({numVertex}, options.device(device));
    torch::Tensor cov3dTensor = torch::empty({numVertex,6}, options.device(device));
    auto position = (glm::vec3*)positionTensor.contiguous().data_ptr<float>();
    auto shs = (glm::vec3*)shsTensor.contiguous().data_ptr<float>();
    auto opacity = opacityTensor.contiguous().data_ptr<float>();
    auto cov3d = cov3dTensor.contiguous().data_ptr<float>();
    for (int i = 0; i < numVertex; i++)
    {
        VertexStorage buf;
        file.read(reinterpret_cast<char*>(&buf), sizeof(VertexStorage));
        position[i] = buf.position;
        constexpr int SH_N = 16;
        //memcpy(&shs[i * SH_N], buf.shs, 48 * sizeof(float));
        shs[i * SH_N] = { buf.shs[0], buf.shs[1], buf.shs[2] };
        for (auto j = 1; j < SH_N; j++)
        {
            shs[i * SH_N + j] = { buf.shs[(j - 1) + 3], buf.shs[(j - 1) + SH_N + 2], buf.shs[(j - 1) + SH_N * 2 + 1] };
        }
        opacity[i] = 1.0f / (1.0f + std::exp(-buf.opacity));
        buf.scale.x = std::exp(buf.scale.x);
        buf.scale.y = std::exp(buf.scale.y);
        buf.scale.z = std::exp(buf.scale.z);
        buf.rotation = glm::normalize(buf.rotation);
        computeCov3D(buf.scale, 1.0, buf.rotation, &cov3d[i * 6]);
    }
    return std::make_tuple(numVertex, positionTensor, shsTensor, opacityTensor, cov3dTensor);
}

void preprocess_torch(
	torch::Tensor& orig_points, torch::Tensor& shs, torch::Tensor& opacities, torch::Tensor& cov3Ds,
	int width, int height, int block_x, int block_y,
    torch::Tensor& position, torch::Tensor& rotation,
    float focal_x, float focal_y, float zFar, float zNear,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_unsorted, torch::Tensor& gaussian_values_unsorted,
	torch::Tensor& curr_offset)
{
    auto position_data = position.contiguous().data_ptr<float>();
    auto rotation_data = rotation.contiguous().data_ptr<float>();
    preprocess(
        (int)opacities.size(0),
        (glm::vec3*)orig_points.contiguous().data_ptr<float>(),
        (shs_deg3_t*)shs.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (cov3d_t*)cov3Ds.contiguous().data_ptr<float>(),
        width, height, block_x, block_y,
        glm::vec3({position_data[0], position_data[1], position_data[2]}), 
        glm::mat3({{rotation_data[0], rotation_data[1], rotation_data[2]},
                {rotation_data[3], rotation_data[4], rotation_data[5]},
                {rotation_data[6], rotation_data[7], rotation_data[8]}}),
        focal_x, focal_y, zFar, zNear,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(), (uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
        curr_offset.data_ptr<int>());
}

void preprocess_half_sh_torch(
	torch::Tensor& orig_points, torch::Tensor& shs_half, torch::Tensor& opacities, torch::Tensor& cov3Ds,
	int width, int height, int block_x, int block_y,
	torch::Tensor& position, torch::Tensor& rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_unsorted, torch::Tensor& gaussian_values_unsorted,
	torch::Tensor& curr_offset)
{
	auto position_data = position.contiguous().data_ptr<float>();
	auto rotation_data = rotation.contiguous().data_ptr<float>();
	preprocess_half_sh(
		(int)opacities.size(0),
		(glm::vec3*)orig_points.contiguous().data_ptr<float>(),
		(__half*)shs_half.contiguous().data_ptr<at::Half>(),
		opacities.contiguous().data_ptr<float>(),
		(cov3d_t*)cov3Ds.contiguous().data_ptr<float>(),
		width, height, block_x, block_y,
		glm::vec3({position_data[0], position_data[1], position_data[2]}),
		glm::mat3({{rotation_data[0], rotation_data[1], rotation_data[2]},
				{rotation_data[3], rotation_data[4], rotation_data[5]},
				{rotation_data[6], rotation_data[7], rotation_data[8]}}),
		focal_x, focal_y, zFar, zNear,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
		curr_offset.data_ptr<int>());
}

// =====================================================================
// ES Pass-A: count tiles per Gaussian + store metadata (no key emission).
// Use for verification against baseline (sum of counts should equal num_rendered).
// =====================================================================
void preprocess_es_pass_a_torch(
	torch::Tensor& orig_points, torch::Tensor& shs, torch::Tensor& opacities, torch::Tensor& cov3Ds,
	int width, int height, int block_x, int block_y,
	torch::Tensor& position, torch::Tensor& rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& tiles_per_gauss, torch::Tensor& conic_power_raw,
	torch::Tensor& depth_natural, torch::Tensor& rect_bounds)
{
	auto position_data = position.contiguous().data_ptr<float>();
	auto rotation_data = rotation.contiguous().data_ptr<float>();
	preprocess_es_pass_a(
		(int)opacities.size(0),
		(glm::vec3*)orig_points.contiguous().data_ptr<float>(),
		(shs_deg3_t*)shs.contiguous().data_ptr<float>(),
		opacities.contiguous().data_ptr<float>(),
		(cov3d_t*)cov3Ds.contiguous().data_ptr<float>(),
		width, height, block_x, block_y,
		glm::vec3({position_data[0], position_data[1], position_data[2]}),
		glm::mat3({{rotation_data[0], rotation_data[1], rotation_data[2]},
				{rotation_data[3], rotation_data[4], rotation_data[5]},
				{rotation_data[6], rotation_data[7], rotation_data[8]}}),
		focal_x, focal_y, zFar, zNear,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		tiles_per_gauss.contiguous().data_ptr<int32_t>(),
		(float4*)conic_power_raw.contiguous().data_ptr<float>(),
		depth_natural.contiguous().data_ptr<float>(),
		(int2*)rect_bounds.contiguous().data_ptr<int32_t>());
}

void preprocess_es_pass_a_half_sh_torch(
	torch::Tensor& orig_points, torch::Tensor& shs_half, torch::Tensor& opacities, torch::Tensor& cov3Ds,
	int width, int height, int block_x, int block_y,
	torch::Tensor& position, torch::Tensor& rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& tiles_per_gauss, torch::Tensor& conic_power_raw,
	torch::Tensor& depth_natural, torch::Tensor& rect_bounds)
{
	auto position_data = position.contiguous().data_ptr<float>();
	auto rotation_data = rotation.contiguous().data_ptr<float>();
	preprocess_es_pass_a_half_sh(
		(int)opacities.size(0),
		(glm::vec3*)orig_points.contiguous().data_ptr<float>(),
		(__half*)shs_half.contiguous().data_ptr<at::Half>(),
		opacities.contiguous().data_ptr<float>(),
		(cov3d_t*)cov3Ds.contiguous().data_ptr<float>(),
		width, height, block_x, block_y,
		glm::vec3({position_data[0], position_data[1], position_data[2]}),
		glm::mat3({{rotation_data[0], rotation_data[1], rotation_data[2]},
				{rotation_data[3], rotation_data[4], rotation_data[5]},
				{rotation_data[6], rotation_data[7], rotation_data[8]}}),
		focal_x, focal_y, zFar, zNear,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		tiles_per_gauss.contiguous().data_ptr<int32_t>(),
		(float4*)conic_power_raw.contiguous().data_ptr<float>(),
		depth_natural.contiguous().data_ptr<float>(),
		(int2*)rect_bounds.contiguous().data_ptr<int32_t>());
}

// ES Pass-B: emit 32-bit tile keys in depth-sorted order.
void preprocess_es_pass_b_torch(
	int P,
	int width, int height, int block_x, int block_y,
	torch::Tensor& perm,
	torch::Tensor& cum_offsets_sorted,
	torch::Tensor& tiles_per_gauss_sorted,
	torch::Tensor& points_xy,
	torch::Tensor& conic_power_raw,
	torch::Tensor& rect_bounds,
	torch::Tensor& tile_keys_unsorted,
	torch::Tensor& gauss_values_unsorted)
{
	preprocess_es_pass_b(
		P, width, height, block_x, block_y,
		(const uint32_t*)perm.contiguous().data_ptr<int32_t>(),
		cum_offsets_sorted.contiguous().data_ptr<int32_t>(),
		tiles_per_gauss_sorted.contiguous().data_ptr<int32_t>(),
		(const float2*)points_xy.contiguous().data_ptr<float>(),
		(const float4*)conic_power_raw.contiguous().data_ptr<float>(),
		(const int2*)rect_bounds.contiguous().data_ptr<int32_t>(),
		(uint32_t*)tile_keys_unsorted.contiguous().data_ptr<int32_t>(),
		(uint32_t*)gauss_values_unsorted.contiguous().data_ptr<int32_t>());
}

// ES step A: depth sort + gather + scan. Writes `total_num_rendered_out` as device scalar.
void es_depth_sort_and_scan_torch(
	int P,
	torch::Tensor& tiles_per_gauss,
	torch::Tensor& depth_natural,
	torch::Tensor& identity_buf,
	torch::Tensor& perm_buf,
	torch::Tensor& depth_sorted_buf,
	torch::Tensor& tiles_per_gauss_sorted_buf,
	torch::Tensor& cum_offsets_sorted_buf,
	torch::Tensor& total_num_rendered_out,
	torch::Tensor& cub_sort_scratch,
	torch::Tensor& cub_scan_scratch)
{
	es_depth_sort_and_scan(
		P,
		tiles_per_gauss.contiguous().data_ptr<int32_t>(),
		depth_natural.contiguous().data_ptr<float>(),
		(uint32_t*)identity_buf.contiguous().data_ptr<int32_t>(),
		(uint32_t*)perm_buf.contiguous().data_ptr<int32_t>(),
		depth_sorted_buf.contiguous().data_ptr<float>(),
		tiles_per_gauss_sorted_buf.contiguous().data_ptr<int32_t>(),
		cum_offsets_sorted_buf.contiguous().data_ptr<int32_t>(),
		total_num_rendered_out.contiguous().data_ptr<int32_t>(),
		(char*)cub_sort_scratch.contiguous().data_ptr(),
		cub_sort_scratch.size(0),
		(char*)cub_scan_scratch.contiguous().data_ptr(),
		cub_scan_scratch.size(0));
}

void es_tile_sort_torch(
	int num_rendered,
	int width, int height, int block_x, int block_y,
	torch::Tensor& tile_keys_unsorted,
	torch::Tensor& gauss_values_unsorted,
	torch::Tensor& tile_keys_sorted,
	torch::Tensor& gauss_values_sorted,
	torch::Tensor& gaussian_keys_sorted_out,
	torch::Tensor& cub_sort_scratch)
{
	es_tile_sort(
		num_rendered,
		width, height, block_x, block_y,
		(uint32_t*)tile_keys_unsorted.contiguous().data_ptr<int32_t>(),
		(uint32_t*)gauss_values_unsorted.contiguous().data_ptr<int32_t>(),
		(uint32_t*)tile_keys_sorted.contiguous().data_ptr<int32_t>(),
		(uint32_t*)gauss_values_sorted.contiguous().data_ptr<int32_t>(),
		(uint64_t*)gaussian_keys_sorted_out.contiguous().data_ptr<int64_t>(),
		(char*)cub_sort_scratch.contiguous().data_ptr(),
		cub_sort_scratch.size(0));
}

size_t get_es_scan_buffer_size_torch(int P)
{
	return get_es_scan_buffer_size(P);
}

void sort_gaussian_torch(int num_rendered,
    int width, int height, int block_x, int block_y,
	torch::Tensor& list_sorting_space,
	torch::Tensor& gaussian_keys_unsorted, torch::Tensor& gaussian_values_unsorted,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted)
{
    sort_gaussian(num_rendered,
        width, height, block_x, block_y,
        (char*)list_sorting_space.contiguous().data_ptr(), list_sorting_space.size(0),
        (uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(), (uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(), (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>());
}

size_t get_sort_buffer_size_torch(int num_rendered)
{
    return get_sort_buffer_size(num_rendered);
}

void gather_features_torch(int num_rendered,
	torch::Tensor& sorted_values,
	torch::Tensor& src_xy, torch::Tensor& src_rgb_depth, torch::Tensor& src_conic_opacity,
	torch::Tensor& dst_xy, torch::Tensor& dst_rgb_depth, torch::Tensor& dst_conic_opacity)
{
	gather_features(num_rendered,
		(uint32_t*)sorted_values.contiguous().data_ptr<int>(),
		(float2*)src_xy.contiguous().data_ptr<float>(),
		(float4*)src_rgb_depth.contiguous().data_ptr<float>(),
		(float4*)src_conic_opacity.contiguous().data_ptr<float>(),
		(float2*)dst_xy.contiguous().data_ptr<float>(),
		(float4*)dst_rgb_depth.contiguous().data_ptr<float>(),
		(float4*)dst_conic_opacity.contiguous().data_ptr<float>());
}

void render_16x16_topk_smem_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& topk_xy, torch::Tensor& topk_rgb_depth, torch::Tensor& topk_conic,
	int num_topk,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_topk_smem(
		num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(const float2*)topk_xy.contiguous().data_ptr<float>(),
		(const float4*)topk_rgb_depth.contiguous().data_ptr<float>(),
		(const float4*)topk_conic.contiguous().data_ptr<float>(),
		num_topk,
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_topk_smem_persistent_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& topk_xy, torch::Tensor& topk_rgb_depth, torch::Tensor& topk_conic,
	int num_topk,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_topk_smem_persistent(
		num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(const float2*)topk_xy.contiguous().data_ptr<float>(),
		(const float4*)topk_rgb_depth.contiguous().data_ptr<float>(),
		(const float4*)topk_conic.contiguous().data_ptr<float>(),
		num_topk,
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_topk_smem_persistent_v2_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& topk_xy, torch::Tensor& topk_rgb_depth, torch::Tensor& topk_conic,
	int num_topk,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_topk_smem_persistent_v2(
		num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(const float2*)topk_xy.contiguous().data_ptr<float>(),
		(const float4*)topk_rgb_depth.contiguous().data_ptr<float>(),
		(const float4*)topk_conic.contiguous().data_ptr<float>(),
		num_topk,
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void precompute_tile_ranges_torch(int num_rendered,
	int width, int height, int block_x, int block_y,
	torch::Tensor& gaussian_keys_sorted,
	torch::Tensor& tile_ranges_buf)
{
	precompute_tile_ranges(num_rendered, width, height, block_x, block_y,
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(int2*)tile_ranges_buf.data_ptr<int>());
}

void precompute_tile_ranges_scan_torch(int num_rendered,
	int width, int height, int block_x, int block_y,
	torch::Tensor& gaussian_keys_sorted,
	torch::Tensor& tile_ranges_buf)
{
	precompute_tile_ranges_scan(num_rendered, width, height, block_x, block_y,
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(int2*)tile_ranges_buf.data_ptr<int>());
}

void compute_tile_order_torch(
	torch::Tensor& tile_ranges_buf,
	int total_tiles,
	torch::Tensor& tile_order,
	torch::Tensor& tile_counts_buf,
	torch::Tensor& tile_ids_buf,
	torch::Tensor& sort_temp)
{
	compute_tile_order(
		(int2*)tile_ranges_buf.data_ptr<int>(),
		total_tiles,
		(uint32_t*)tile_order.data_ptr<int>(),
		(uint32_t*)tile_counts_buf.data_ptr<int>(),
		(uint32_t*)tile_ids_buf.data_ptr<int>(),
		(char*)sort_temp.data_ptr(),
		sort_temp.size(0));
}

size_t get_tile_order_sort_temp_size_torch(int total_tiles)
{
	return get_tile_order_sort_temp_size(total_tiles);
}

void render_16x16_reordered_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& tile_ranges_buf, torch::Tensor& tile_order,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_reordered(
		num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(int2*)tile_ranges_buf.data_ptr<int>(),
		(uint32_t*)tile_order.data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void compute_tile_order_packed_torch(
	torch::Tensor& tile_ranges_buf,
	int total_tiles,
	int x_blocks,
	torch::Tensor& tile_desc_buf,
	torch::Tensor& tile_order,
	torch::Tensor& tile_counts_buf,
	torch::Tensor& tile_ids_buf,
	torch::Tensor& sort_temp,
	int zigzag_group_size)
{
	compute_tile_order_packed(
		(int2*)tile_ranges_buf.data_ptr<int>(),
		total_tiles,
		x_blocks,
		(int4*)tile_desc_buf.data_ptr<int>(),
		(uint32_t*)tile_order.data_ptr<int>(),
		(uint32_t*)tile_counts_buf.data_ptr<int>(),
		(uint32_t*)tile_ids_buf.data_ptr<int>(),
		(char*)sort_temp.data_ptr(),
		sort_temp.size(0),
		zigzag_group_size);
}

void compute_tile_order_packed_morton_torch(
	torch::Tensor& tile_ranges_buf,
	int total_tiles,
	int x_blocks,
	torch::Tensor& tile_desc_buf,
	torch::Tensor& tile_order,
	torch::Tensor& tile_counts_buf,
	torch::Tensor& tile_ids_buf,
	torch::Tensor& sort_temp,
	int zigzag_group_size,
	int morton_bucket_size)
{
	compute_tile_order_packed_morton(
		(int2*)tile_ranges_buf.data_ptr<int>(),
		total_tiles,
		x_blocks,
		(int4*)tile_desc_buf.data_ptr<int>(),
		(uint32_t*)tile_order.data_ptr<int>(),
		(uint32_t*)tile_counts_buf.data_ptr<int>(),
		(uint32_t*)tile_ids_buf.data_ptr<int>(),
		(char*)sort_temp.data_ptr(),
		sort_temp.size(0),
		zigzag_group_size,
		morton_bucket_size);
}

void render_16x16_reordered_v2_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& tile_desc_buf,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_reordered_v2(
		num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(int4*)tile_desc_buf.data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_reordered_persistent_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& tile_desc_buf,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_reordered_persistent(
		num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(int4*)tile_desc_buf.data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_unroll2_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    render_16x16_unroll2(num_rendered,
        width, height,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
        float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
        (uchar3*)out_color.data_ptr());
}

void render_16x16_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    render_16x16(num_rendered,
        width, height,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
        float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
        (uchar3*)out_color.data_ptr());
}

void render_16x16_preranges_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_values_sorted,
	torch::Tensor& tile_ranges_buf,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_preranges(num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(int2*)tile_ranges_buf.data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_preranges_smem_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_values_sorted,
	torch::Tensor& tile_ranges_buf,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_preranges_smem(num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(int2*)tile_ranges_buf.data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_preranges_smem_persistent_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_values_sorted,
	torch::Tensor& tile_desc_buf,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_preranges_smem_persistent(num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(int4*)tile_desc_buf.data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_preranges_smem_persistent_lite_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_values_sorted,
	torch::Tensor& tile_ranges_buf,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_preranges_smem_persistent_lite(num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(int2*)tile_ranges_buf.data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_preranges_smem_persistent_lite_dt_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_values_sorted,
	torch::Tensor& tile_ranges_buf,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_preranges_smem_persistent_lite_dt(num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(int2*)tile_ranges_buf.data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_preranges_smem_persistent_lite_fused_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_preranges_smem_persistent_lite_fused(num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_preranges_naive_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_values_sorted,
	torch::Tensor& tile_ranges_buf,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_preranges_naive(num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(int2*)tile_ranges_buf.data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_preranges_smem_v2_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_preranges_smem_v2(num_rendered,
		width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());
}

void render_16x16_split_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    render_16x16_split(num_rendered,
        width, height,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
        float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
        (uchar3*)out_color.data_ptr());
}

void render_24x16_split_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    render_24x16_split(num_rendered,
        width, height,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
        float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
        (uchar3*)out_color.data_ptr());
}

void render_32x16_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    render_32x16(num_rendered,
        width, height,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
        float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
        (uchar3*)out_color.data_ptr());
}

void render_32x16_split_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    render_32x16_split(num_rendered,
        width, height,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
        float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
        (uchar3*)out_color.data_ptr());
}

void render_32x32_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    render_32x32(num_rendered,
        width, height,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
        float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
        (uchar3*)out_color.data_ptr());
}

// Combined sort + render: reads curr_offset from device, sorts, and renders
// in a single C++ call, eliminating Python-level CPU sync overhead.
int sort_and_render_torch(
	int width, int height, int block_x, int block_y,
	const std::string& render_variant,
	torch::Tensor& curr_offset,
	torch::Tensor& list_sorting_space,
	torch::Tensor& gaussian_keys_unsorted, torch::Tensor& gaussian_values_unsorted,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    // Read num_rendered from device
    int num_rendered = 0;
    polysplat::fetch_num_rendered_async(curr_offset.data_ptr<int>(), &num_rendered);

    // Sort
    sort_gaussian(num_rendered,
        width, height, block_x, block_y,
        (char*)list_sorting_space.contiguous().data_ptr(), list_sorting_space.size(0),
        (uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>());

    // Render
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    float3 bg = float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]};
    auto xy_ptr = (float2*)points_xy.contiguous().data_ptr<float>();
    auto rgb_ptr = (float4*)rgb_depth.contiguous().data_ptr<float>();
    auto conic_ptr = (float4*)conic_opacity.contiguous().data_ptr<float>();
    auto keys_ptr = (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>();
    auto vals_ptr = (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>();
    auto out_ptr = (uchar3*)out_color.data_ptr();

    if (render_variant == "default") {
        if (block_x == 16 && block_y == 16)
            render_16x16(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
        else if (block_x == 32 && block_y == 16)
            render_32x16(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
        else if (block_x == 32 && block_y == 32)
            render_32x32(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
    } else if (render_variant == "unroll2") {
        render_16x16_unroll2(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
    } else if (render_variant == "split") {
        if (block_x == 16 && block_y == 16)
            render_16x16_split(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
        else if (block_x == 24 && block_y == 16)
            render_24x16_split(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
        else if (block_x == 32 && block_y == 16)
            render_32x16_split(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
    }

    return num_rendered;
}

// Fully fused forward: preprocess + sort + render in a single C++ call.
// Eliminates ALL Python overhead between kernel launches.
int forward_fused_torch(
	torch::Tensor& orig_points, torch::Tensor& shs, torch::Tensor& opacities, torch::Tensor& cov3Ds,
	int width, int height, int block_x, int block_y,
	torch::Tensor& position, torch::Tensor& rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	const std::string& render_variant,
	torch::Tensor& curr_offset,
	torch::Tensor& list_sorting_space,
	torch::Tensor& gaussian_keys_unsorted, torch::Tensor& gaussian_values_unsorted,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    // Zero curr_offset
    cudaMemsetAsync(curr_offset.data_ptr<int>(), 0, sizeof(int));

    // Preprocess
    auto position_data = position.contiguous().data_ptr<float>();
    auto rotation_data = rotation.contiguous().data_ptr<float>();
    preprocess(
        (int)opacities.size(0),
        (glm::vec3*)orig_points.contiguous().data_ptr<float>(),
        (shs_deg3_t*)shs.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (cov3d_t*)cov3Ds.contiguous().data_ptr<float>(),
        width, height, block_x, block_y,
        glm::vec3({position_data[0], position_data[1], position_data[2]}),
        glm::mat3({{rotation_data[0], rotation_data[1], rotation_data[2]},
                {rotation_data[3], rotation_data[4], rotation_data[5]},
                {rotation_data[6], rotation_data[7], rotation_data[8]}}),
        focal_x, focal_y, zFar, zNear,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
        curr_offset.data_ptr<int>());

    // Read num_rendered from device (syncs with preprocess completion)
    int num_rendered = 0;
    polysplat::fetch_num_rendered_async(curr_offset.data_ptr<int>(), &num_rendered);

    // Sort
    sort_gaussian(num_rendered,
        width, height, block_x, block_y,
        (char*)list_sorting_space.contiguous().data_ptr(), list_sorting_space.size(0),
        (uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>());

    // Render
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    float3 bg = float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]};
    auto xy_ptr = (float2*)points_xy.contiguous().data_ptr<float>();
    auto rgb_ptr = (float4*)rgb_depth.contiguous().data_ptr<float>();
    auto conic_ptr = (float4*)conic_opacity.contiguous().data_ptr<float>();
    auto keys_ptr = (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>();
    auto vals_ptr = (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>();
    auto out_ptr = (uchar3*)out_color.data_ptr();

    if (render_variant == "default") {
        if (block_x == 16 && block_y == 16)
            render_16x16(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
        else if (block_x == 32 && block_y == 16)
            render_32x16(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
        else if (block_x == 32 && block_y == 32)
            render_32x32(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
    } else if (render_variant == "unroll2") {
        render_16x16_unroll2(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
    } else if (render_variant == "split") {
        if (block_x == 16 && block_y == 16)
            render_16x16_split(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
        else if (block_x == 24 && block_y == 16)
            render_24x16_split(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
        else if (block_x == 32 && block_y == 16)
            render_32x16_split(num_rendered, width, height, xy_ptr, rgb_ptr, conic_ptr, keys_ptr, vals_ptr, bg, out_ptr);
    }

    return num_rendered;
}

// E2E forward (uses separate precompute_tile_ranges kernel before render).
// Eliminates Python dispatch between all 4 phases but keeps the existing render path.
int forward_fused_e2e_preranges_torch(
	torch::Tensor& orig_points, torch::Tensor& shs_half, torch::Tensor& opacities, torch::Tensor& cov3Ds,
	int width, int height, int block_x, int block_y,
	torch::Tensor& position, torch::Tensor& rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	torch::Tensor& curr_offset,
	torch::Tensor& list_sorting_space,
	torch::Tensor& gaussian_keys_unsorted, torch::Tensor& gaussian_values_unsorted,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& tile_ranges_buf,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	cudaMemsetAsync(curr_offset.data_ptr<int>(), 0, sizeof(int));

	auto position_data = position.contiguous().data_ptr<float>();
	auto rotation_data = rotation.contiguous().data_ptr<float>();
	preprocess_half_sh(
		(int)opacities.size(0),
		(glm::vec3*)orig_points.contiguous().data_ptr<float>(),
		(__half*)shs_half.contiguous().data_ptr<at::Half>(),
		opacities.contiguous().data_ptr<float>(),
		(cov3d_t*)cov3Ds.contiguous().data_ptr<float>(),
		width, height, block_x, block_y,
		glm::vec3({position_data[0], position_data[1], position_data[2]}),
		glm::mat3({{rotation_data[0], rotation_data[1], rotation_data[2]},
				{rotation_data[3], rotation_data[4], rotation_data[5]},
				{rotation_data[6], rotation_data[7], rotation_data[8]}}),
		focal_x, focal_y, zFar, zNear,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
		curr_offset.data_ptr<int>());

	int num_rendered = 0;
	polysplat::fetch_num_rendered_async(curr_offset.data_ptr<int>(), &num_rendered);

	sort_gaussian(num_rendered,
		width, height, block_x, block_y,
		(char*)list_sorting_space.contiguous().data_ptr(), list_sorting_space.size(0),
		(uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>());

	// Separate precompute_tile_ranges kernel (same as Python path)
	precompute_tile_ranges(num_rendered, width, height, block_x, block_y,
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(int2*)tile_ranges_buf.data_ptr<int>());

	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_preranges_smem_persistent_lite(
		num_rendered, width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		(int2*)tile_ranges_buf.data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());

	return num_rendered;
}

// E2E fully-fused forward: preprocess_half_sh + sort + render_preranges_smem_persistent_lite_fused.
// Single C++ call eliminates Python dispatch overhead between all kernel phases.
// Uses the fused-tile-range render kernel so no separate precompute_tile_ranges is needed.
int forward_fused_e2e_torch(
	torch::Tensor& orig_points, torch::Tensor& shs_half, torch::Tensor& opacities, torch::Tensor& cov3Ds,
	int width, int height, int block_x, int block_y,
	torch::Tensor& position, torch::Tensor& rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	torch::Tensor& curr_offset,
	torch::Tensor& list_sorting_space,
	torch::Tensor& gaussian_keys_unsorted, torch::Tensor& gaussian_values_unsorted,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
	// Zero curr_offset on current stream
	cudaMemsetAsync(curr_offset.data_ptr<int>(), 0, sizeof(int));

	// Preprocess (FP16 SH variant — reduces SH memory traffic 2x)
	auto position_data = position.contiguous().data_ptr<float>();
	auto rotation_data = rotation.contiguous().data_ptr<float>();
	preprocess_half_sh(
		(int)opacities.size(0),
		(glm::vec3*)orig_points.contiguous().data_ptr<float>(),
		(__half*)shs_half.contiguous().data_ptr<at::Half>(),
		opacities.contiguous().data_ptr<float>(),
		(cov3d_t*)cov3Ds.contiguous().data_ptr<float>(),
		width, height, block_x, block_y,
		glm::vec3({position_data[0], position_data[1], position_data[2]}),
		glm::mat3({{rotation_data[0], rotation_data[1], rotation_data[2]},
				{rotation_data[3], rotation_data[4], rotation_data[5]},
				{rotation_data[6], rotation_data[7], rotation_data[8]}}),
		focal_x, focal_y, zFar, zNear,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
		curr_offset.data_ptr<int>());

	// Read num_rendered from device (CUB DeviceRadixSort needs num_items on host).
	// This single cudaMemcpy replaces Python-side .cpu() sync.
	int num_rendered = 0;
	polysplat::fetch_num_rendered_async(curr_offset.data_ptr<int>(), &num_rendered);

	// Sort (CUB radix sort on (tile_id | depth) composite key)
	sort_gaussian(num_rendered,
		width, height, block_x, block_y,
		(char*)list_sorting_space.contiguous().data_ptr(), list_sorting_space.size(0),
		(uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>());

	// Render with inline binary search (no separate precompute_tile_ranges!)
	auto bg_color_data = bg_color.contiguous().data_ptr<float>();
	render_16x16_preranges_smem_persistent_lite_fused(
		num_rendered, width, height,
		(float2*)points_xy.contiguous().data_ptr<float>(),
		(float4*)rgb_depth.contiguous().data_ptr<float>(),
		(float4*)conic_opacity.contiguous().data_ptr<float>(),
		(uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
		(uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
		float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
		(uchar3*)out_color.data_ptr());

	return num_rendered;
}

} // namespace
} // namespace polysplat

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    auto ops = m.def_submodule("ops", "my custom operators");

    ops.def(
        "loadPly",
        &polysplat::loadPly_torch,
        "load .ply file and return gaussian model data");

    ops.def(
        "preprocess",
        &polysplat::preprocess_torch,
        "preprocess gaussian model data and generate key-value pairs");

    ops.def(
        "preprocess_half_sh",
        &polysplat::preprocess_half_sh_torch,
        "preprocess with FP16 SH coefficients (halves SH memory traffic)");

    ops.def(
        "preprocess_es_pass_a",
        &polysplat::preprocess_es_pass_a_torch,
        "ES Pass-A: count tiles per Gaussian + store metadata (no key emission)");

    ops.def(
        "preprocess_es_pass_a_half_sh",
        &polysplat::preprocess_es_pass_a_half_sh_torch,
        "ES Pass-A with FP16 SH");

    ops.def(
        "preprocess_es_pass_b",
        &polysplat::preprocess_es_pass_b_torch,
        "ES Pass-B: emit 32-bit tile keys in depth-sorted order");

    ops.def(
        "es_depth_sort_and_scan",
        &polysplat::es_depth_sort_and_scan_torch,
        "ES step A: depth sort + gather + exclusive scan");

    ops.def(
        "es_tile_sort",
        &polysplat::es_tile_sort_torch,
        "ES step B: 32-bit stable tile sort + rebuild 64-bit keys");

    ops.def(
        "get_es_scan_buffer_size",
        &polysplat::get_es_scan_buffer_size_torch,
        "CUB DeviceScan workspace size");

    ops.def(
        "sort_gaussian",
        &polysplat::sort_gaussian_torch,
        "sort gaussian key-value pairs");

    ops.def(
        "get_sort_buffer_size",
        &polysplat::get_sort_buffer_size_torch,
        "get sort buffer size");

    ops.def(
        "gather_features",
        &polysplat::gather_features_torch,
        "gather Gaussian features into tile-sorted order");

    ops.def(
        "render_16x16",
        &polysplat::render_16x16_torch,
        "sort key-value pairs and render");

    ops.def(
        "render_16x16_preranges",
        &polysplat::render_16x16_preranges_torch,
        "default render with precomputed tile ranges (no binary search)");

    ops.def(
        "render_16x16_preranges_smem",
        &polysplat::render_16x16_preranges_smem_torch,
        "preranges render with shared memory batched gaussian loading");

    ops.def(
        "render_16x16_preranges_smem_persistent",
        &polysplat::render_16x16_preranges_smem_persistent_torch,
        "persistent work-stealing render with cp.async smem + Morton/zigzag tile ordering");

    ops.def(
        "render_16x16_preranges_smem_persistent_lite",
        &polysplat::render_16x16_preranges_smem_persistent_lite_torch,
        "lightweight persistent work-stealing render with cp.async smem (no tile reorder)");

    ops.def(
        "render_16x16_preranges_smem_persistent_lite_dt",
        &polysplat::render_16x16_preranges_smem_persistent_lite_dt_torch,
        "Dynamic Thresholding variant: skips ex2 for power < log2(1/255)");

    ops.def(
        "render_16x16_preranges_naive",
        &polysplat::render_16x16_preranges_naive_torch,
        "ablation: preranges render with direct __ldg loads, no cp.async or smem");

    ops.def(
        "render_16x16_preranges_smem_v2",
        &polysplat::render_16x16_preranges_smem_v2_torch,
        "preranges smem render with fused binary search (no separate tile_ranges kernel)");

    ops.def(
        "render_16x16_topk_smem",
        &polysplat::render_16x16_topk_smem_torch,
        "render with per-block shared-memory cache for the globally hottest Gaussians");

    ops.def(
        "render_16x16_topk_smem_persistent",
        &polysplat::render_16x16_topk_smem_persistent_torch,
        "render with persistent kernel + shared-memory cache for top-k Gaussians");

    ops.def(
        "render_16x16_topk_smem_persistent_v2",
        &polysplat::render_16x16_topk_smem_persistent_v2_torch,
        "render with V2 persistent kernel (2-warps-per-tile, lower reg pressure)");

    ops.def(
        "precompute_tile_ranges",
        &polysplat::precompute_tile_ranges_torch,
        "precompute tile ranges from sorted keys");

    ops.def(
        "precompute_tile_ranges_scan",
        &polysplat::precompute_tile_ranges_scan_torch,
        "precompute tile ranges via boundary scan (O(num_rendered))");

    ops.def(
        "compute_tile_order",
        &polysplat::compute_tile_order_torch,
        "compute tile reordering (descending by gaussian count)");

    ops.def(
        "get_tile_order_sort_temp_size",
        &polysplat::get_tile_order_sort_temp_size_torch,
        "get temp buffer size for tile order sort");

    ops.def(
        "render_16x16_reordered",
        &polysplat::render_16x16_reordered_torch,
        "render with tile reordering for better load balance");

    ops.def(
        "compute_tile_order_packed",
        &polysplat::compute_tile_order_packed_torch,
        "compute packed int4 tile descriptor (col,row,range.x,range.y) ordered by count desc");

    ops.def(
        "compute_tile_order_packed_morton",
        &polysplat::compute_tile_order_packed_morton_torch,
        "packed tile descriptor with count-bucket + Morton spatial locality");

    ops.def(
        "render_16x16_reordered_v2",
        &polysplat::render_16x16_reordered_v2_torch,
        "render with packed int4 tile descriptor (reordered_v2)");

    ops.def(
        "render_16x16_reordered_persistent",
        &polysplat::render_16x16_reordered_persistent_torch,
        "persistent work-stealing render over packed int4 descriptors (reordered combo)");

    ops.def(
        "render_16x16_unroll2",
        &polysplat::render_16x16_unroll2_torch,
        "render with partial unrolling for I-cache optimization");

    ops.def(
        "render_16x16_split",
        &polysplat::render_16x16_split_torch,
        "sort key-value pairs and render with two warps per 16x16 tile");

    ops.def(
        "render_24x16_split",
        &polysplat::render_24x16_split_torch,
        "sort key-value pairs and render with two warps per 24x16 tile");

    ops.def(
        "render_32x16",
        &polysplat::render_32x16_torch,
        "sort key-value pairs and render");

    ops.def(
        "render_32x16_split",
        &polysplat::render_32x16_split_torch,
        "sort key-value pairs and render with two warps per 32x16 tile");

    ops.def(
        "render_32x32",
        &polysplat::render_32x32_torch,
        "sort key-value pairs and render");

    ops.def(
        "sort_and_render",
        &polysplat::sort_and_render_torch,
        "combined sort + render in single C++ call");

    ops.def(
        "forward_fused",
        &polysplat::forward_fused_torch,
        "fully fused forward: preprocess + sort + render in single C++ call");

    ops.def(
        "render_16x16_preranges_smem_persistent_lite_fused",
        &polysplat::render_16x16_preranges_smem_persistent_lite_fused_torch,
        "persistent_lite render with inline binary search (no precompute_tile_ranges)");

    ops.def(
        "forward_fused_e2e",
        &polysplat::forward_fused_e2e_torch,
        "E2E fused forward: preprocess_half_sh + sort + render_preranges_smem_persistent_lite_fused");

    ops.def(
        "forward_fused_e2e_preranges",
        &polysplat::forward_fused_e2e_preranges_torch,
        "E2E forward with separate precompute_tile_ranges + render_persistent_lite");
}
