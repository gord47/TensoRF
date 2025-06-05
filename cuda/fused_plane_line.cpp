#include <torch/extension.h>

// Declare the CUDA function
torch::Tensor fused_plane_line_forward_cuda(torch::Tensor input);

torch::Tensor fused_plane_line_forward(torch::Tensor input) {
    return fused_plane_line_forward_cuda(input);
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_plane_line_forward, "Fused plane line forward (CUDA)");
}