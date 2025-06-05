#include <torch/extension.h>
#include <vector>

// Fix the declaration here to match fused_plane_line.cu
std::vector<torch::Tensor> fused_plane_line_forward_cuda(
    torch::Tensor planes,
    torch::Tensor lines,
    torch::Tensor coord_plane,
    torch::Tensor coord_line);

std::vector<torch::Tensor> fused_plane_line_forward(
    torch::Tensor planes,
    torch::Tensor lines,
    torch::Tensor coord_plane,
    torch::Tensor coord_line) {
    return fused_plane_line_forward_cuda(planes, lines, coord_plane, coord_line);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_plane_line_forward, "Fused Plane Line forward (CUDA)");
}
