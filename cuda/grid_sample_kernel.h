#ifndef GRID_SAMPLE_KERNEL_H
#define GRID_SAMPLE_KERNEL_H

void launch_grid_sample_kernel(
    const float* input, const float* grid, float* output,
    int N, int C, int H, int W, int H_out, int W_out, bool align_corners);

#endif