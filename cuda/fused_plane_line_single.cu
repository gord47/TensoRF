#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "interpolation_utils.cuh"

// Kernel for a single plane-line component
__global__ void fused_plane_line_single_kernel(
    const float *__restrict__ plane,
    const float *__restrict__ line,
    const float *__restrict__ coord_plane,
    const float *__restrict__ coord_line,
    float *__restrict__ out,
    int C, int H, int W, int L, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    float acc = 0.0f;
    for (int c = 0; c < C; ++c)
    {
        const float *plane_c = plane + c * H * W;
        const float *line_c = line + c * L;

        float x = coord_plane[i * 2 + 0];
        float y = coord_plane[i * 2 + 1];
        float z = coord_line[i];

        float p = bilinear_interp(plane_c, x, y, H, W);
        float l = linear_interp(line_c, z, L);
        acc += p * l;
    }

    // Atomically add to output for thread safety when multiple kernels write to the same output
    atomicAdd(&out[i], acc);
}

std::vector<torch::Tensor> fused_plane_line_single_forward_cuda(
    torch::Tensor plane,
    torch::Tensor line,
    torch::Tensor coord_plane,
    torch::Tensor coord_line,
    torch::Tensor output)
{
    // Handle 4D input tensors [1, C, H, W] or [1, C, L, 1]
    int C = plane.size(1);
    int H = plane.size(2);
    int W = plane.size(3);
    int L = line.size(2);
    int N = coord_plane.size(0);
    printf("C=%d, H=%d, W=%d, L=%d, N=%d\n", C, H, W, L, N);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    fused_plane_line_single_kernel<<<blocks, threads>>>(
        plane.contiguous().data_ptr<float>(),
        line.contiguous().data_ptr<float>(),
        coord_plane.contiguous().data_ptr<float>(),
        coord_line.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        C, H, W, L, N);

    return {output};
}
