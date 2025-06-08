#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA forward declarations
std::vector<torch::Tensor> fused_ray_render_cuda_forward(
    torch::Tensor rays,
    torch::Tensor density_planes,
    torch::Tensor density_lines,
    torch::Tensor app_planes,
    torch::Tensor app_lines,
    torch::Tensor basis_mat_weight,
    torch::Tensor basis_mat_bias,
    torch::Tensor aabb,
    torch::Tensor grid_size,
    float step_size,
    int n_samples,
    bool white_bg,
    bool is_train,
    float distance_scale,
    float ray_march_weight_thres);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fused_ray_render_forward(
    torch::Tensor rays,
    torch::Tensor density_planes,
    torch::Tensor density_lines,
    torch::Tensor app_planes,
    torch::Tensor app_lines,
    torch::Tensor basis_mat_weight,
    torch::Tensor basis_mat_bias,
    torch::Tensor aabb,
    torch::Tensor grid_size,
    float step_size,
    int n_samples,
    bool white_bg,
    bool is_train,
    float distance_scale,
    float ray_march_weight_thres)
{

    CHECK_INPUT(rays);
    CHECK_INPUT(density_planes);
    CHECK_INPUT(density_lines);
    CHECK_INPUT(app_planes);
    CHECK_INPUT(app_lines);
    CHECK_INPUT(basis_mat_weight);
    CHECK_INPUT(aabb);
    CHECK_INPUT(grid_size);

    return fused_ray_render_cuda_forward(
        rays, density_planes, density_lines, app_planes, app_lines,
        basis_mat_weight, basis_mat_bias, aabb, grid_size,
        step_size, n_samples, white_bg, is_train, distance_scale, ray_march_weight_thres);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &fused_ray_render_forward, "Fused ray renderer forward");
}
