#include <torch/extension.h>
#include <vector>

// Function declaration from the original CUDA .cu file
std::vector<at::Tensor> fused_plane_line_forward_cuda(
    at::Tensor planes, at::Tensor lines, at::Tensor coord_plane, at::Tensor coord_line);

// Function declaration from the new flexible CUDA .cu file
std::vector<at::Tensor> fused_plane_line_flexible_forward_cuda(
    std::vector<at::Tensor> planes, std::vector<at::Tensor> lines, at::Tensor coord_plane, at::Tensor coord_line);

// Function declaration from the new single component CUDA .cu file
std::vector<at::Tensor> fused_plane_line_single_forward_cuda(
    at::Tensor plane, at::Tensor line, at::Tensor coord_plane, at::Tensor coord_line, at::Tensor output);

// Original C++ wrapper function
std::vector<at::Tensor> fused_plane_line_forward(
    at::Tensor planes, at::Tensor lines, at::Tensor coord_plane, at::Tensor coord_line)
{
    return fused_plane_line_forward_cuda(planes, lines, coord_plane, coord_line);
}

// New C++ wrapper function for flexible processing
std::vector<at::Tensor> fused_plane_line_flexible_forward(
    std::vector<at::Tensor> planes, std::vector<at::Tensor> lines, at::Tensor coord_plane, at::Tensor coord_line)
{
    return fused_plane_line_flexible_forward_cuda(planes, lines, coord_plane, coord_line);
}

// New C++ wrapper function for individual components
std::vector<at::Tensor> fused_plane_line_single_forward(
    at::Tensor plane, at::Tensor line, at::Tensor coord_plane, at::Tensor coord_line, at::Tensor output)
{
    return fused_plane_line_single_forward_cuda(plane, line, coord_plane, coord_line, output);
}

// PYBIND11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &fused_plane_line_forward, "Fused Plane Line forward (CUDA)");
    m.def("forward_flexible", &fused_plane_line_flexible_forward, "Fused Plane Line Flexible forward (CUDA)");
    m.def("forward_single", &fused_plane_line_single_forward, "Fused Plane Line Single Component forward (CUDA)");
}