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
        x = (x + 1.f) * 0.5f * (W - 1);  // Only do this once
        y = (y + 1.f) * 0.5f * (H - 1);
        z = (z + 1.f) * 0.5f * (L - 1);

        float p = bilinear_interp(plane_c, x, y, H, W);
        float l = linear_interp(line_c, z, L);
        if (i==0){
            printf("x=%.4f y=%.4f z=%.4f p=%.6f l=%.6f plane=%.6f line=%.6f plane_c=%.6f line_c=%.6f p*l=%.6f\n", x, y, z, p, l, plane, line, plane_c, line_c, p * l);
        }
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
    auto plane_cpu = plane.to(torch::kCPU);  // copy tensor to CPU
    auto plane_acc = plane_cpu.accessor<float, 4>();  // 4D: [1, C, H, W]

    std::cout << "Plane values (sample):" << std::endl;
    for (int c = 0; c < std::min(C, 2); ++c) {
        for (int h = 0; h < std::min(H, 2); ++h) {
            for (int w = 0; w < std::min(W, 2); ++w) {
                std::cout << "plane[0][" << c << "][" << h << "][" << w << "] = "
                        << plane_acc[0][c][h][w] << std::endl;
            }
        }
    }
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
