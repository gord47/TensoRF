#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__device__ __forceinline__ float grid_sampler_compute_source_index(
    float coord, int size, bool align_corners) {
    if (align_corners) {
        return ((coord + 1) / 2) * (size - 1);
    } else {
        return ((coord + 1) * size - 1) / 2;
    }
}

__device__ __forceinline__ bool within_bounds_2d(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

__global__ void grid_sample_kernel(
    const float* input,
    const float* grid,
    float* output,
    int N, int C, int H, int W,
    int out_H, int out_W,
    bool align_corners
) {
    int n = blockIdx.z;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= N || s >= out_H || t >= out_W) return;

    int grid_offset = n * out_H * out_W * 2 + s * out_W * 2 + t * 2;
    float x = grid[grid_offset];
    float y = grid[grid_offset + 1];
    
    float ix = grid_sampler_compute_source_index(x, W, align_corners);
    float iy = grid_sampler_compute_source_index(y, H, align_corners);
    
    int ix_nw = static_cast<int>(floorf(ix));
    int iy_nw = static_cast<int>(floorf(iy));
    
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    
    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);
    
    for (int c = 0; c < C; ++c) {
        int input_offset = n * C * H * W + c * H * W;
        
        float nw_val = (within_bounds_2d(iy_nw, ix_nw, H, W)) ? 
                       input[input_offset + iy_nw * W + ix_nw] : 0.0f;
        float ne_val = (within_bounds_2d(iy_ne, ix_ne, H, W)) ? 
                       input[input_offset + iy_ne * W + ix_ne] : 0.0f;
        float sw_val = (within_bounds_2d(iy_sw, ix_sw, H, W)) ? 
                       input[input_offset + iy_sw * W + ix_sw] : 0.0f;
        float se_val = (within_bounds_2d(iy_se, ix_se, H, W)) ? 
                       input[input_offset + iy_se * W + ix_se] : 0.0f;
        
        float out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
        
        int output_offset = n * C * out_H * out_W + c * out_H * out_W + s * out_W + t;
        output[output_offset] = out_val;
    }
}

void launch_grid_sample_kernel(
    const float* input,
    const float* grid,
    float* output,
    int N, int C, int H, int W,
    int out_H, int out_W,
    bool align_corners
) {
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (out_W + threads.x - 1) / threads.x,
        (out_H + threads.y - 1) / threads.y,
        N
    );
    
    grid_sample_kernel<<<blocks, threads>>>(
        input, grid, output, N, C, H, W, out_H, out_W, align_corners
    );
}