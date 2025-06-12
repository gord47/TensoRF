#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "interpolation_utils.cuh"

// Original kernel for uniform-sized planes/lines
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
            const float *line = lines + axis * C * L * 1 + c * L * 1; // L*1 because lines shape is [3, C, L, 1]

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

// Flexible kernel for variable-sized planes/lines
__global__ void fused_plane_line_flexible_kernel(
    const float **__restrict__ plane_ptrs,
    const float **__restrict__ line_ptrs,
    const int *__restrict__ plane_dims, // [3 x 3] array: [H0,W0,C0, H1,W1,C1, H2,W2,C2]
    const int *__restrict__ line_dims,  // [3 x 2] array: [L0,C0, L1,C1, L2,C2]
    const float *__restrict__ coord_plane,
    const float *__restrict__ coord_line,
    float *__restrict__ out,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    float acc = 0.0f;
    for (int axis = 0; axis < 3; ++axis)
    {
        int H = plane_dims[axis * 3 + 0];
        int W = plane_dims[axis * 3 + 1];
        int C_plane = plane_dims[axis * 3 + 2];

        int L = line_dims[axis * 2 + 0];
        int C_line = line_dims[axis * 2 + 1];

        // Use minimum of plane and line channels
        int C = min(C_plane, C_line);

        const float *plane_base = plane_ptrs[axis];
        const float *line_base = line_ptrs[axis];

        float x = coord_plane[axis * N * 2 + i * 2 + 0];
        float y = coord_plane[axis * N * 2 + i * 2 + 1];
        float z = coord_line[axis * N + i];

        for (int c = 0; c < C; ++c)
        {
            const float *plane = plane_base + c * H * W;
            const float *line = line_base + c * L;

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

// New flexible function for variable-sized planes/lines
std::vector<torch::Tensor> fused_plane_line_flexible_forward_cuda(
    std::vector<torch::Tensor> planes,
    std::vector<torch::Tensor> lines,
    torch::Tensor coord_plane,
    torch::Tensor coord_line)
{
    int N = coord_plane.size(1);
    auto output = torch::zeros({N}, planes[0].options());

    // Prepare device arrays for plane/line pointers and dimensions
    std::vector<const float *> h_plane_ptrs(3), h_line_ptrs(3);
    std::vector<int> h_plane_dims(9), h_line_dims(6); // 3*3 for planes, 3*2 for lines

    for (int i = 0; i < 3; ++i)
    {
        h_plane_ptrs[i] = planes[i].contiguous().data_ptr<float>();
        h_line_ptrs[i] = lines[i].contiguous().data_ptr<float>();

        // planes[i] shape: [1, C, H, W]
        h_plane_dims[i * 3 + 0] = planes[i].size(2); // H
        h_plane_dims[i * 3 + 1] = planes[i].size(3); // W
        h_plane_dims[i * 3 + 2] = planes[i].size(1); // C

        // lines[i] shape: [1, C, L, 1]
        h_line_dims[i * 2 + 0] = lines[i].size(2); // L
        h_line_dims[i * 2 + 1] = lines[i].size(1); // C
    }

    // Allocate device memory
    const float **d_plane_ptrs, **d_line_ptrs;
    int *d_plane_dims, *d_line_dims;

    cudaMalloc(&d_plane_ptrs, 3 * sizeof(float *));
    cudaMalloc(&d_line_ptrs, 3 * sizeof(float *));
    cudaMalloc(&d_plane_dims, 9 * sizeof(int));
    cudaMalloc(&d_line_dims, 6 * sizeof(int));

    // Copy to device
    cudaMemcpy(d_plane_ptrs, h_plane_ptrs.data(), 3 * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_line_ptrs, h_line_ptrs.data(), 3 * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_plane_dims, h_plane_dims.data(), 9 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_line_dims, h_line_dims.data(), 6 * sizeof(int), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    fused_plane_line_flexible_kernel<<<blocks, threads>>>(
        d_plane_ptrs,
        d_line_ptrs,
        d_plane_dims,
        d_line_dims,
        coord_plane.contiguous().data_ptr<float>(),
        coord_line.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        N);

    // Cleanup device memory
    cudaFree(d_plane_ptrs);
    cudaFree(d_line_ptrs);
    cudaFree(d_plane_dims);
    cudaFree(d_line_dims);

    return {output};
}
