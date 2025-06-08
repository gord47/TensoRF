#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ float bilinear_interp(const float *plane, float x, float y, int H, int W)
{
    float fx = fminf(fmaxf(x, 0.0f), W - 1.0f);
    float fy = fminf(fmaxf(y, 0.0f), H - 1.0f);

    int x0 = static_cast<int>(floorf(fx));
    int x1 = min(x0 + 1, W - 1);
    int y0 = static_cast<int>(floorf(fy));
    int y1 = min(y0 + 1, H - 1);

    float dx = fx - x0;
    float dy = fy - y0;
    float v00 = plane[y0 * W + x0];
    float v01 = plane[y0 * W + x1];
    float v10 = plane[y1 * W + x0];
    float v11 = plane[y1 * W + x1];
    float p_test = v00 * (1 - dx)*(1 - dy)+ v01 * dx*(1 - dy) + v10 * (1 - dx)*dy + v11 * dx * dy;

    float v0 = v00 * (1 - dx) + v01 * dx;
    float v1 = v10 * (1 - dx) + v11 * dx;

    return v0 * (1 - dy) + v1 * dy;
}

__device__ __forceinline__ float linear_interp(const float *line, float z, int L)
{
    // z = fmaxf(0.0f, fminf(z * 0.5f + 0.5f, 1.0f));
    // float fz = z * (L - 1);
    float fz = fminf(fmaxf(z, 0.0f), L - 1.0f);
    int z0 = static_cast<int>(floorf(fz)), z1 = min(z0 + 1, L - 1);
    float dz = fz - z0;
    float v0 = line[z0], v1 = line[z1];
    return v0 * (1.0f - dz) + v1 * dz;
}
