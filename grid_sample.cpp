#include <torch/extension.h>

void launch_grid_sample_kernel(
    const float* input, const float* grid, float* output,
    int N, int C, int H, int W, int H_out, int W_out, bool align_corners);

torch::Tensor grid_sample_cuda_forward(
    torch::Tensor input,
    torch::Tensor grid,
    bool align_corners
) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto H_out = grid.size(1);
    auto W_out = grid.size(2);

    auto output = torch::zeros(
        {N, C, H_out, W_out},
        input.options()
    );

    launch_grid_sample_kernel(
        input.data_ptr<float>(),
        grid.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        align_corners
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &grid_sample_cuda_forward, "Grid sample forward (CUDA)");
}