#include <torch/extension.h>

// Declare the CUDA function
torch::Tensor fused_plane_line_forward_cuda(torch::Tensor input);


// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_plane_line_forward_cuda, "Fused plane line forward (CUDA)");
}