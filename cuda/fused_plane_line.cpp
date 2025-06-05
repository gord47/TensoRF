#include <torch/extension.h>
#include <vector>

// Match the function signature defined in fused_plane_line.cu
std::vector<at::Tensor> fused_plane_line_forward_cuda(
    at::Tensor plane, at::Tensor line1, at::Tensor line2, at::Tensor line3);

std::vector<at::Tensor> fused_plane_line_forward(
    at::Tensor plane, at::Tensor line1, at::Tensor line2, at::Tensor line3) {
    return fused_plane_line_forward_cuda(plane, line1, line2, line3);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_plane_line_forward, "Fused Plane Line forward (CUDA)");
}