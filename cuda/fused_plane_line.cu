#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__device__ float bilinear_interp(const float* plane, float x, float y, int H, int W) {
    // grid_sample with align_corners=True: x, y ∈ [-1, 1] → index in [0, size - 1]
    float fx = (x + 1.f) * 0.5f * (W - 1);
    float fy = (y + 1.f) * 0.5f * (H - 1);

    fx = fminf(fmaxf(fx, 0.0f), W - 1.0f);
    fy = fminf(fmaxf(fy, 0.0f), H - 1.0f);

    int x0 = floorf(fx), y0 = floorf(fy);
    int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    float dx = fx - x0, dy = fy - y0;

    float val = 0.0f;
    val += plane[y0 * W + x0] * (1 - dx) * (1 - dy);
    val += plane[y0 * W + x1] * dx * (1 - dy);
    val += plane[y1 * W + x0] * (1 - dx) * dy;
    val += plane[y1 * W + x1] * dx * dy;
    return val;
}

__device__ float linear_interp(const float* line, float z, int L) {
    float fz = (z + 1.f) * 0.5f * (L - 1);
    fz = fminf(fmaxf(fz, 0.0f), L - 1.0f);

    int z0 = floorf(fz), z1 = min(z0 + 1, L - 1);
    float dz = fz - z0;
    return line[z0] * (1 - dz) + line[z1] * dz;
}

__global__ void fused_plane_line_kernel(
    const float* __restrict__ planes,
    const float* __restrict__ lines,
    const float* __restrict__ coord_plane, // [3, N, 2] flattened
    const float* __restrict__ coord_line,  // [3, N] flattened
    float* __restrict__ out,
    int C, int H, int W, int L, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float acc = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        for (int c = 0; c < C; ++c) {
            const float* plane = planes + axis * C * H * W + c * H * W;
            const float* line  = lines + axis * C * L + c * L;

            float x = coord_plane[(axis * N + i) * 2 + 0];
            float y = coord_plane[(axis * N + i) * 2 + 1];
            float z = coord_line[axis * N + i];

            float p = bilinear_interp(plane, x, y, H, W);
            float l = linear_interp(line, z, L);
            acc += p * l;
        }
    }
    out[i] = acc;
}

std::vector<torch::Tensor> fused_plane_line_forward_cuda(
    torch::Tensor planes,       // [3, C, H, W]
    torch::Tensor lines,        // [3, C, L]
    torch::Tensor coord_plane,  // [3, N, 2]
    torch::Tensor coord_line    // [3, N]
) {
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
        C, H, W, L, N
    );

    return {output};
}
