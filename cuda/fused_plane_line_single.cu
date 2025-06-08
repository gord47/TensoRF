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
    float x = coord_plane[i * 2 + 0];
    float y = coord_plane[i * 2 + 1];
    float z = coord_line[i];
    x = (x + 1.f) * 0.5f * (W - 1);  // Only do this once
    y = (y + 1.f) * 0.5f * (H - 1);
    z = (z + 1.f) * 0.5f * (L - 1);
    float acc = 0.0f;
    for (int c = 0; c < C; ++c)
    {
        const float *plane_c = coord_plane + c * H * W;
        const float *line_c = coord_line + c * L;
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
    TORCH_CHECK(plane.dim() == 4 || plane.dim() == 3, "Plane must be 3D or 4D");
    TORCH_CHECK(line.dim() == 4 || line.dim() == 2, "Line must be 2D or 4D");
    // Handle 4D input tensors [1, C, H, W] or [1, C, L, 1]
    int C = plane.size(1);
    int H = plane.size(2);
    int W = plane.size(3);
    int L = line.size(2);
    int N = coord_plane.size(0);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    auto plane_squeezed = plane.contiguous();  // [C, H, W]
    auto line_squeezed = line.contiguous();  // [C, L]
    fused_plane_line_single_kernel<<<blocks, threads>>>(
        plane_squeezed.data_ptr<float>(),
        line_squeezed.data_ptr<float>(),
        coord_plane.contiguous().data_ptr<float>(),
        coord_line.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        C, H, W, L, N);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    return {output};
}
