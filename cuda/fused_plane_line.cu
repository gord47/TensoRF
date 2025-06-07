#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "interpolation_utils.cuh"

__global__ void fused_plane_line_kernel(
    const float *__restrict__ planes,
    const float *__restrict__ lines,
    const float *__restrict__ coord_plane,
    const float *__restrict__ coord_line,
    float *__restrict__ out,
    int C, int H, int W, int L, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    float acc = 0.0f;
    for (int axis = 0; axis < 3; ++axis)
    {
        for (int c = 0; c < C; ++c)
        {
            const float *plane = planes + axis * C * H * W + c * H * W;
            const float *line = lines + axis * C * L + c * L;

            float x = coord_plane[axis * N * 2 + i * 2 + 0];
            float y = coord_plane[axis * N * 2 + i * 2 + 1];
            float z = coord_line[axis * N + i];

            float p = bilinear_interp(plane, x, y, H, W);
            float l = linear_interp(line, z, L);
            acc += p * l;
        }
    }
    out[i] = acc;
}

std::vector<torch::Tensor> fused_plane_line_forward_cuda(
    torch::Tensor planes,
    torch::Tensor lines,
    torch::Tensor coord_plane,
    torch::Tensor coord_line)
{
    int C = planes.size(1), H = planes.size(2), W = planes.size(3);
    int L = lines.size(2), N = coord_plane.size(1);

    auto output = torch::zeros({N}, planes.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    fused_plane_line_kernel<<<blocks, threads>>>(
        planes.contiguous().data_ptr<float>(),
        lines.contiguous().data_ptr<float>(),
        coord_plane.contiguous().data_ptr<float>(),
        coord_line.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        C, H, W, L, N);

    return {output};
}
