#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ float bilinear_interp(const float *plane, float x, float y, int H, int W)
{
    x = fmaxf(0.0f, fminf(x * 0.5f + 0.5f, 1.0f));
    y = fmaxf(0.0f, fminf(y * 0.5f + 0.5f, 1.0f));
    float fx = x * (W - 1);
    float fy = y * (H - 1);
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

__device__ __forceinline__ float linear_interp(const float *line, float z, int L)
{
    z = fmaxf(0.0f, fminf(z * 0.5f + 0.5f, 1.0f));
    float fz = z * (L - 1);
    int z0 = floorf(fz), z1 = min(z0 + 1, L - 1);
    float dz = fz - z0;
    return line[z0] * (1 - dz) + line[z1] * dz;
}
