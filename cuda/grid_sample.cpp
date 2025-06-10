#include <torch/extension.h>
#include "grid_sample_kernel.h"

torch::Tensor grid_sample_cuda_forward(
    torch::Tensor input,
    torch::Tensor grid,
    bool align_corners
) {
    AT_CHECK(input.requires_grad() == false, "input must not require grad");
    AT_CHECK(grid.requires_grad() == false, "grid must not require grad");
    
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto out_H = grid.size(1);
    auto out_W = grid.size(2);
    
    auto output = torch::zeros({N, C, out_H, out_W}, input.options());
    
    launch_grid_sample_kernel(
        input.contiguous().data_ptr<float>(),
        grid.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        out_H, out_W,
        align_corners
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &grid_sample_cuda_forward, "Grid sample forward (CUDA)");
}