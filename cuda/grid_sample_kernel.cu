#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void grid_sample_kernel(
    const float* input,
    const float* grid,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    bool align_corners
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;
    if (index >= total_elements) return;

    int n = index / (C * H_out * W_out);
    int c = (index % (C * H_out * W_out)) / (H_out * W_out);
    int h = (index % (H_out * W_out)) / W_out;
    int w = index % W_out;

    int grid_offset = n * H_out * W_out * 2 + h * W_out * 2 + w * 2;
    float x = grid[grid_offset];
    float y = grid[grid_offset + 1];

    // Handle 1D case (line features)
    bool is_1D = (W == 1);
    if (is_1D) {
        // Use only y-coordinate, set x to -1 (leftmost column)
        x = -1.0f;
    }

    float ix, iy;
    if (align_corners) {
        ix = ((x + 1) / 2) * (W - 1);
        iy = ((y + 1) / 2) * (H - 1);
    } else {
        ix = ((x + 1) * W - 1) / 2;
        iy = ((y + 1) * H - 1) / 2;
    }

    int ix0 = floorf(ix);
    int iy0 = floorf(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    float dx = ix - ix0;
    float dy = iy - iy0;
    float w00 = (1 - dx) * (1 - dy);
    float w01 = dx * (1 - dy);
    float w10 = (1 - dx) * dy;
    float w11 = dx * dy;

    int in_offset = n * C * H * W + c * H * W;

    auto get_pixel = [&](int x, int y) {
        return (x >= 0 && x < W && y >= 0 && y < H) ? 
            input[in_offset + y * W + x] : 0.0f;
    };

    float val = w00 * get_pixel(ix0, iy0) +
                w01 * get_pixel(ix1, iy0) +
                w10 * get_pixel(ix0, iy1) +
                w11 * get_pixel(ix1, iy1);

    // For 1D inputs, use bilinear interpolation but ignore x-dimension
    if (is_1D) {
        val = (1 - dy) * get_pixel(0, iy0) + 
              dy * get_pixel(0, iy1);
    }

    int out_offset = n * C * H_out * W_out + c * H_out * W_out + h * W_out + w;
    output[out_offset] = val;
}

void launch_grid_sample_kernel(
    const float* input,
    const float* grid,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    bool align_corners
) {
    int total_elements = N * C * H_out * W_out;
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;
    grid_sample_kernel<<<gridSize, blockSize>>>(
        input, grid, output, N, C, H, W, H_out, W_out, align_corners
    );
}