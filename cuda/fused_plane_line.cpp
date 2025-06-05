#include <torch/extension.h>
#include <vector>

// Function declaration from the CUDA .cu file
std::vector<at::Tensor> fused_plane_line_forward_cuda(
    at::Tensor planes, at::Tensor lines, at::Tensor coord_plane, at::Tensor coord_line);

// C++ wrapper function
std::vector<at::Tensor> fused_plane_line_forward(
    at::Tensor planes, at::Tensor lines, at::Tensor coord_plane, at::Tensor coord_line) {
    return fused_plane_line_forward_cuda(planes, lines, coord_plane, coord_line);
}

// PYBIND11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_plane_line_forward, "Fused Plane Line forward (CUDA)");
}