#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor octree_render_forward_cuda(
    const torch::Tensor& rays,
    const torch::Tensor& density_plane,
    const torch::Tensor& density_line,
    const torch::Tensor& app_plane,
    const torch::Tensor& app_line,
    const torch::Tensor& basis_mat,
    const torch::Tensor& aabb,
    const int N_samples,
    const bool ndc_ray,
    const bool white_bg,
    const float distance_scale);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> octree_render_forward(
    const torch::Tensor& rays,
    const torch::Tensor& density_plane,
    const torch::Tensor& density_line,
    const torch::Tensor& app_plane,
    const torch::Tensor& app_line,
    const torch::Tensor& basis_mat,
    const torch::Tensor& aabb,
    const int N_samples,
    const bool ndc_ray,
    const bool white_bg,
    const float distance_scale) {
    
    CHECK_INPUT(rays);
    CHECK_INPUT(density_plane);
    CHECK_INPUT(density_line);
    CHECK_INPUT(app_plane);
    CHECK_INPUT(app_line);
    CHECK_INPUT(basis_mat);
    CHECK_INPUT(aabb);
    
    // Call CUDA kernel
    auto result = octree_render_forward_cuda(
        rays, density_plane, density_line, app_plane, app_line,
        basis_mat, aabb, N_samples, ndc_ray, white_bg, distance_scale
    );

    return {result};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("octree_render_forward", &octree_render_forward, "Octree Render Forward (CUDA)");
}
