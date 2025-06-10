#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <vector>

#define BLOCK_SIZE 256
#define MAX_SAMPLES 512

// Debug mode - set to 1 to enable detailed debugging
#define DEBUG_MODE 0

#if DEBUG_MODE
#define DEBUG_PRINTF(fmt, ...) printf("[DEBUG] Thread %d: " fmt "\n", threadIdx.x + blockIdx.x * blockDim.x, ##__VA_ARGS__)
#define DEBUG_CHECK_BOUNDS(ptr, idx, max_size, name)                             \
    if (idx >= max_size || idx < 0)                                              \
    {                                                                            \
        printf("[ERROR] Thread %d: %s bounds violation - idx:%d, max_size:%d\n", \
               threadIdx.x + blockIdx.x * blockDim.x, name, idx, max_size);      \
        return;                                                                  \
    }
#define DEBUG_CHECK_PTR(ptr, name)                           \
    if (!ptr)                                                \
    {                                                        \
        printf("[ERROR] Thread %d: %s is null pointer\n",    \
               threadIdx.x + blockIdx.x * blockDim.x, name); \
        return;                                              \
    }
#else
#define DEBUG_PRINTF(fmt, ...)
#define DEBUG_CHECK_BOUNDS(ptr, idx, max_size, name)
#define DEBUG_CHECK_PTR(ptr, name)
#endif

// Device functions
__device__ __forceinline__ float3 make_float3_from_ptr(const float *ptr)
{
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

__device__ __forceinline__ void atomic_add_float3(float *dst, float3 src)
{
    atomicAdd(&dst[0], src.x);
    atomicAdd(&dst[1], src.y);
    atomicAdd(&dst[2], src.z);
}

__device__ __forceinline__ float grid_sample_2d(
    const float *grid,
    int C, int H, int W,
    float x, float y,
    int c)
{
    DEBUG_CHECK_PTR(grid, "grid_2d");

    // Early bounds check for channel
    if (c >= C || c < 0)
    {
        DEBUG_PRINTF("grid_sample_2d: channel bounds violation c=%d, C=%d", c, C);
        return 0.0f;
    }

    // Optimized coordinate transformation with fused operations
    x = fmaf(x, 2.0f, -1.0f); // 2*x - 1
    y = fmaf(y, 2.0f, -1.0f); // 2*y - 1

    // Convert to grid coordinates with optimized FMA
    float gx = fmaf(x + 1.0f, 0.5f * (W - 1), 0.0f);
    float gy = fmaf(y + 1.0f, 0.5f * (H - 1), 0.0f);

    // Clamp to valid range
    gx = fmaxf(0.0f, fminf(gx, W - 1.0f));
    gy = fmaxf(0.0f, fminf(gy, H - 1.0f));

    // Fast floor with integer conversion
    int x0 = __float2int_rd(gx); // Fast floor
    int y0 = __float2int_rd(gy);
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);

    float wx = gx - x0;
    float wy = gy - y0;

    // Optimized index calculation with base offset
    int base_offset = c * H * W;
    int idx00 = base_offset + y0 * W + x0;
    int idx01 = base_offset + y0 * W + x1;
    int idx10 = base_offset + y1 * W + x0;
    int idx11 = base_offset + y1 * W + x1;

// Bounds checking only in debug mode for performance
#if DEBUG_MODE
    int max_idx = C * H * W - 1;
    if (idx00 > max_idx || idx01 > max_idx || idx10 > max_idx || idx11 > max_idx)
    {
        DEBUG_PRINTF("grid_sample_2d: array bounds violation - max_idx:%d, indices:[%d,%d,%d,%d]",
                     max_idx, idx00, idx01, idx10, idx11);
        return 0.0f;
    }
#endif

    // Load values and perform bilinear interpolation with FMA optimization
    float v00 = grid[idx00];
    float v01 = grid[idx01];
    float v10 = grid[idx10];
    float v11 = grid[idx11];

    // Optimized bilinear interpolation using FMA instructions
    float wx_inv = 1.0f - wx;
    float wy_inv = 1.0f - wy;

    return fmaf(wx_inv * wy_inv, v00,
                fmaf(wx * wy_inv, v01,
                     fmaf(wx_inv * wy, v10, wx * wy * v11)));
}

__device__ __forceinline__ float grid_sample_1d(
    const float *grid,
    int C, int L,
    float x,
    int c)
{
    DEBUG_CHECK_PTR(grid, "grid_1d");

    // Early bounds check for channel
    if (c >= C || c < 0)
    {
        DEBUG_PRINTF("grid_sample_1d: channel bounds violation c=%d, C=%d", c, C);
        return 0.0f;
    }

    // Optimized coordinate transformation
    x = fmaf(x, 2.0f, -1.0f); // 2*x - 1

    // Convert to grid coordinate with FMA
    float gx = fmaf(x + 1.0f, 0.5f * (L - 1), 0.0f);

    // Clamp to valid range
    gx = fmaxf(0.0f, fminf(gx, L - 1.0f));

    // Fast floor and linear interpolation
    int x0 = __float2int_rd(gx); // Fast floor
    int x1 = min(x0 + 1, L - 1);

    float wx = gx - x0;

    int idx0 = c * L + x0;
    int idx1 = c * L + x1;

// Bounds checking only in debug mode for performance
#if DEBUG_MODE
    int max_idx = C * L - 1;
    if (idx0 > max_idx || idx1 > max_idx)
    {
        DEBUG_PRINTF("grid_sample_1d: array bounds violation - max_idx:%d, indices:[%d,%d]",
                     max_idx, idx0, idx1);
        return 0.0f;
    }
#endif

    // Optimized linear interpolation using FMA
    return fmaf(wx, grid[idx1], (1.0f - wx) * grid[idx0]);
}

__device__ __forceinline__ bool ray_aabb_intersect(
    float3 ray_o, float3 ray_d,
    float3 aabb_min, float3 aabb_max,
    float *t_min, float *t_max)
{

    float3 inv_ray_d = make_float3(
        1.0f / (ray_d.x + 1e-6f),
        1.0f / (ray_d.y + 1e-6f),
        1.0f / (ray_d.z + 1e-6f));

    float3 t1 = make_float3(
        (aabb_min.x - ray_o.x) * inv_ray_d.x,
        (aabb_min.y - ray_o.y) * inv_ray_d.y,
        (aabb_min.z - ray_o.z) * inv_ray_d.z);

    float3 t2 = make_float3(
        (aabb_max.x - ray_o.x) * inv_ray_d.x,
        (aabb_max.y - ray_o.y) * inv_ray_d.y,
        (aabb_max.z - ray_o.z) * inv_ray_d.z);

    *t_min = fmaxf(fmaxf(fminf(t1.x, t2.x), fminf(t1.y, t2.y)), fminf(t1.z, t2.z));
    *t_max = fminf(fminf(fmaxf(t1.x, t2.x), fmaxf(t1.y, t2.y)), fmaxf(t1.z, t2.z));

    return *t_max >= *t_min && *t_max > 0.0f;
}

__global__ void fused_ray_render_kernel(
    const float *__restrict__ rays,             // [N, 6] (origin + direction)
    const float *__restrict__ density_planes,   // [3, C_d, H, W]
    const float *__restrict__ density_lines,    // [3, C_d, L, 1]
    const float *__restrict__ app_planes,       // [3, C_a, H, W]
    const float *__restrict__ app_lines,        // [3, C_a, L, 1]
    const float *__restrict__ basis_mat_weight, // [app_dim, total_app_comp]
    const float *__restrict__ basis_mat_bias,   // [app_dim] or nullptr
    const float *__restrict__ aabb,             // [2, 3] (min, max)
    const int *__restrict__ grid_size,          // [3]
    float step_size,
    int n_samples,
    bool white_bg,
    bool is_train,
    float distance_scale,
    float ray_march_weight_thres,
    float density_shift,                         // Add density_shift parameter
    int n_rays,
    int density_n_comp,
    int app_n_comp,
    int app_dim,
    float *__restrict__ rgb_output,   // [N, 3]
    float *__restrict__ depth_output, // [N]
    curandState *__restrict__ rand_states)
{

    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays)
        return;

    // Debug: Print kernel parameters for first few threads
    if (ray_idx < 3)
    {
        DEBUG_PRINTF("Kernel params - ray_idx:%d, grid_size:[%d,%d,%d], density_n_comp:%d, app_n_comp:%d, app_dim:%d",
                     ray_idx, grid_size[0], grid_size[1], grid_size[2], density_n_comp, app_n_comp, app_dim);
    }

    // Check critical pointers
    DEBUG_CHECK_PTR(rays, "rays");
    DEBUG_CHECK_PTR(density_planes, "density_planes");
    DEBUG_CHECK_PTR(density_lines, "density_lines");
    DEBUG_CHECK_PTR(app_planes, "app_planes");
    DEBUG_CHECK_PTR(app_lines, "app_lines");
    DEBUG_CHECK_PTR(basis_mat_weight, "basis_mat_weight");
    DEBUG_CHECK_PTR(aabb, "aabb");
    DEBUG_CHECK_PTR(grid_size, "grid_size");
    DEBUG_CHECK_PTR(rgb_output, "rgb_output");
    DEBUG_CHECK_PTR(depth_output, "depth_output");

    // Load ray data
    float3 ray_o = make_float3(rays[ray_idx * 6 + 0], rays[ray_idx * 6 + 1], rays[ray_idx * 6 + 2]);
    float3 ray_d = make_float3(rays[ray_idx * 6 + 3], rays[ray_idx * 6 + 4], rays[ray_idx * 6 + 5]);

    // Load AABB
    float3 aabb_min = make_float3(aabb[0], aabb[1], aabb[2]);
    float3 aabb_max = make_float3(aabb[3], aabb[4], aabb[5]);

    // Ray-AABB intersection
    float t_min, t_max;
    if (!ray_aabb_intersect(ray_o, ray_d, aabb_min, aabb_max, &t_min, &t_max))
    {
        // Ray doesn't intersect AABB
        if (white_bg)
        {
            rgb_output[ray_idx * 3 + 0] = 1.0f;
            rgb_output[ray_idx * 3 + 1] = 1.0f;
            rgb_output[ray_idx * 3 + 2] = 1.0f;
        }
        else
        {
            rgb_output[ray_idx * 3 + 0] = 0.0f;
            rgb_output[ray_idx * 3 + 1] = 0.0f;
            rgb_output[ray_idx * 3 + 2] = 0.0f;
        }
        depth_output[ray_idx] = t_max;
        return;
    }

    t_min = fmaxf(t_min, 0.0f);

    // Sample points along ray with optimized loop structure
    float3 rgb_acc = make_float3(0.0f, 0.0f, 0.0f);
    float alpha_acc = 0.0f;
    float depth_acc = 0.0f;

    curandState local_rand_state;
    if (is_train && rand_states)
    {
        local_rand_state = rand_states[ray_idx];
    }

    float step_size_t = step_size;
    const float inv_step_size = 1.0f / step_size_t;

    // Pre-compute grid dimensions for efficiency
    const int H = grid_size[1];
    const int W = grid_size[0];
    const int D = grid_size[2];
    const int total_density_comp = 3 * density_n_comp;
    const int total_app_comp = 3 * app_n_comp;

    // Cache AABB normalization factors for better performance
    const float3 aabb_size = make_float3(
        aabb_max.x - aabb_min.x,
        aabb_max.y - aabb_min.y,
        aabb_max.z - aabb_min.z);
    const float3 inv_aabb_size = make_float3(
        1.0f / aabb_size.x,
        1.0f / aabb_size.y,
        1.0f / aabb_size.z);

    for (int i = 0; i < n_samples; i++)
    {
        float t = t_min + (float)i * step_size_t;
        if (t > t_max)
            break;

        // Add jitter for training with optimized random generation
        if (is_train && rand_states)
        {
            t += curand_uniform(&local_rand_state) * step_size_t;
        }

        // Sample point with vectorized computation
        float3 pos = make_float3(
            fmaf(t, ray_d.x, ray_o.x),
            fmaf(t, ray_d.y, ray_o.y),
            fmaf(t, ray_d.z, ray_o.z));

        // Optimized normalization with pre-computed inverse
        float3 norm_pos = make_float3(
            (pos.x - aabb_min.x) * inv_aabb_size.x,
            (pos.y - aabb_min.y) * inv_aabb_size.y,
            (pos.z - aabb_min.z) * inv_aabb_size.z);

        // Early bounds check with optimized comparison
        if (__any_sync(0xffffffff, norm_pos.x < 0.0f || norm_pos.x > 1.0f ||
                                       norm_pos.y < 0.0f || norm_pos.y > 1.0f ||
                                       norm_pos.z < 0.0f || norm_pos.z > 1.0f))
        {
            continue;
        }

        // Compute density feature with optimized memory access
        float sigma_feature = 0.0f;

        // Process density features using optimized concatenated tensor layout
        if (density_planes && density_lines)
        {
            // Pre-load normalized positions for better cache utilization
            const float norm_x = norm_pos.x;
            const float norm_y = norm_pos.y;
            const float norm_z = norm_pos.z;

// XY plane with Z line - unrolled for better performance
#pragma unroll 4
            for (int c = 0; c < density_n_comp; c++)
            {
                float plane_val = grid_sample_2d(density_planes, total_density_comp, H, W, norm_x, norm_y, c);
                float line_val = grid_sample_1d(density_lines, total_density_comp, D, norm_z, c);
                sigma_feature = fmaf(plane_val, line_val, sigma_feature);
            }

            // XZ plane with Y line - offset by density_n_comp
            const int xz_offset = density_n_comp;
#pragma unroll 4
            for (int c = 0; c < density_n_comp; c++)
            {
                int comp_idx = xz_offset + c;
                float plane_val = grid_sample_2d(density_planes, total_density_comp, D, W, norm_x, norm_z, comp_idx);
                float line_val = grid_sample_1d(density_lines, total_density_comp, H, norm_y, comp_idx);
                sigma_feature = fmaf(plane_val, line_val, sigma_feature);
            }

            // YZ plane with X line - offset by 2*density_n_comp
            const int yz_offset = 2 * density_n_comp;
#pragma unroll 4
            for (int c = 0; c < density_n_comp; c++)
            {
                int comp_idx = yz_offset + c;
                float plane_val = grid_sample_2d(density_planes, total_density_comp, D, H, norm_y, norm_z, comp_idx);
                float line_val = grid_sample_1d(density_lines, total_density_comp, W, norm_x, comp_idx);
                sigma_feature = fmaf(plane_val, line_val, sigma_feature);
            }
        }

        // Proper density computation matching PyTorch implementation
        // Apply density_shift and softplus activation like PyTorch
        float shifted_feature = sigma_feature + density_shift; // Use passed density_shift parameter
        float sigma = log1pf(__expf(shifted_feature));         // softplus(x) = log(1 + exp(x))

        // Improved alpha computation with clamped integration
        float dt = step_size_t * distance_scale;
        float sigma_dt = fminf(sigma * dt, 15.0f); // Increased clamp for better dynamic range
        float alpha = 1.0f - __expf(-sigma_dt);    // Fast exponential
        float weight = alpha * (1.0f - alpha_acc);

        // Enhanced early termination for performance
        if (weight < ray_march_weight_thres)
        {
            continue;
        }

        // More aggressive alpha accumulation threshold
        if (alpha_acc > 0.98f)
        {
            break;
        }

        // Compute appearance features with optimized memory access
        if (weight > ray_march_weight_thres)
        {
            // Use shared memory for app features for better performance
            __shared__ float shared_features[BLOCK_SIZE * 16]; // Assuming max 16 features per thread
            float *app_features = &shared_features[threadIdx.x * 16];

// Initialize features efficiently
#pragma unroll 8
            for (int j = 0; j < min(app_dim, 16); j++)
            {
                app_features[j] = 0.0f;
            }

            // Handle remaining features if app_dim > 16
            for (int j = 16; j < app_dim && j < 64; j++)
            {
                app_features[j] = 0.0f;
            }

            // Optimized appearance feature computation with cached values
            const float norm_x = norm_pos.x;
            const float norm_y = norm_pos.y;
            const float norm_z = norm_pos.z;

            // Compute appearance features: process all 3 planes (XY, XZ, YZ) 
            // Each plane contributes app_n_comp features, total = 3 * app_n_comp = total_app_comp
            // Must match PyTorch: concatenate all plane features, then apply basis matrix
            
            // Temporary storage for all combined features (plane * line)
            float combined_features[144]; // Max possible features
            int feature_idx = 0;

            // XY plane with Z line
            for (int c = 0; c < app_n_comp; c++, feature_idx++)
            {
                float plane_val = grid_sample_2d(app_planes, total_app_comp, H, W, norm_x, norm_y, c);
                float line_val = grid_sample_1d(app_lines, total_app_comp, D, norm_z, c);
                combined_features[feature_idx] = plane_val * line_val;
            }

            // XZ plane with Y line
            const int xz_offset = app_n_comp;
            for (int c = 0; c < app_n_comp; c++, feature_idx++)
            {
                int comp_idx = xz_offset + c;
                float plane_val = grid_sample_2d(app_planes, total_app_comp, D, W, norm_x, norm_z, comp_idx);
                float line_val = grid_sample_1d(app_lines, total_app_comp, H, norm_y, comp_idx);
                combined_features[feature_idx] = plane_val * line_val;
            }

            // YZ plane with X line  
            const int yz_offset = 2 * app_n_comp;
            for (int c = 0; c < app_n_comp; c++, feature_idx++)
            {
                int comp_idx = yz_offset + c;
                float plane_val = grid_sample_2d(app_planes, total_app_comp, D, H, norm_y, norm_z, comp_idx);
                float line_val = grid_sample_1d(app_lines, total_app_comp, W, norm_x, comp_idx);
                combined_features[feature_idx] = plane_val * line_val;
            }

            // Now apply basis matrix transformation: combined_features[total_app_comp] -> app_features[app_dim]
            // This matches PyTorch: basis_mat((plane_coef_point * line_coef_point).T)
            for (int d = 0; d < app_dim; d++)
            {
                float result = 0.0f;
                for (int f = 0; f < total_app_comp; f++)
                {
                    int basis_idx = d * total_app_comp + f;
                    result = fmaf(basis_mat_weight[basis_idx], combined_features[f], result);
                }
                app_features[d] = result;
            }

            // Add bias if available
            if (basis_mat_bias)
            {
                for (int d = 0; d < app_dim; d++)
                {
                    app_features[d] += basis_mat_bias[d];
                }
            }

            // Optimized RGB conversion with fast math functions
            float3 rgb;
            if (app_dim >= 3)
            {
                // Enhanced sigmoid with fast math for better performance
                rgb.x = __fdividef(1.0f, 1.0f + __expf(-app_features[0]));
                rgb.y = __fdividef(1.0f, 1.0f + __expf(-app_features[1]));
                rgb.z = __fdividef(1.0f, 1.0f + __expf(-app_features[2]));
            }
            else if (app_dim == 1)
            {
                float gray = fmaxf(0.0f, fminf(1.0f, app_features[0]));
                rgb = make_float3(gray, gray, gray);
            }
            else
            {
                rgb.x = fmaxf(0.0f, fminf(1.0f, app_features[0]));
                rgb.y = app_dim > 1 ? fmaxf(0.0f, fminf(1.0f, app_features[1])) : rgb.x;
                rgb.z = rgb.x;
            }

            // Optimized color accumulation with FMA
            rgb_acc.x = fmaf(weight, rgb.x, rgb_acc.x);
            rgb_acc.y = fmaf(weight, rgb.y, rgb_acc.y);
            rgb_acc.z = fmaf(weight, rgb.z, rgb_acc.z);
            depth_acc = fmaf(weight, t, depth_acc);
        }

        alpha_acc += weight;

        // Early termination if alpha is close to 1
        if (alpha_acc > 0.99f)
            break;
    }

    // Apply background
    if (white_bg)
    {
        rgb_acc.x += (1.0f - alpha_acc);
        rgb_acc.y += (1.0f - alpha_acc);
        rgb_acc.z += (1.0f - alpha_acc);
    }

    // Debug: Check output bounds before writing
    int rgb_idx_r = ray_idx * 3 + 0;
    int rgb_idx_g = ray_idx * 3 + 1;
    int rgb_idx_b = ray_idx * 3 + 2;

    if (ray_idx < 3)
    {
        DEBUG_PRINTF("Final output - ray_idx:%d, rgb_acc:[%.3f,%.3f,%.3f], depth_acc:%.3f, alpha_acc:%.3f",
                     ray_idx, rgb_acc.x, rgb_acc.y, rgb_acc.z, depth_acc, alpha_acc);
        DEBUG_PRINTF("Output indices - rgb:[%d,%d,%d], depth:%d", rgb_idx_r, rgb_idx_g, rgb_idx_b, ray_idx);
    }

    // Check bounds before writing
    DEBUG_CHECK_BOUNDS(rgb_output, rgb_idx_r, n_rays * 3, "rgb_output_r");
    DEBUG_CHECK_BOUNDS(rgb_output, rgb_idx_g, n_rays * 3, "rgb_output_g");
    DEBUG_CHECK_BOUNDS(rgb_output, rgb_idx_b, n_rays * 3, "rgb_output_b");
    DEBUG_CHECK_BOUNDS(depth_output, ray_idx, n_rays, "depth_output");

    // Store results
    rgb_output[ray_idx * 3 + 0] = rgb_acc.x;
    rgb_output[ray_idx * 3 + 1] = rgb_acc.y;
    rgb_output[ray_idx * 3 + 2] = rgb_acc.z;
    depth_output[ray_idx] = depth_acc;

    // Update random state
    if (is_train && rand_states)
    {
        rand_states[ray_idx] = local_rand_state;
    }
}

__global__ void init_curand_kernel(curandState *state, unsigned long seed, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

std::vector<torch::Tensor> fused_ray_render_cuda_forward(
    torch::Tensor rays,
    torch::Tensor density_planes,
    torch::Tensor density_lines,
    torch::Tensor app_planes,
    torch::Tensor app_lines,
    torch::Tensor basis_mat_weight,
    torch::Tensor basis_mat_bias,
    torch::Tensor aabb,
    torch::Tensor grid_size,
    float step_size,
    int n_samples,
    bool white_bg,
    bool is_train,
    float distance_scale,
    float ray_march_weight_thres,
    float density_shift)
{

    const int n_rays = rays.size(0);
    // FIXED: Correct component calculation for concatenated tensors
    // For TensorVMSplit: density_planes shape [48, H, W] = 3 planes * 16 components each
    // So individual density_n_comp = 48/3 = 16
    const int density_n_comp = density_planes.size(0) / 3; // Individual component count per plane
    const int app_n_comp = app_planes.size(0) / 3;         // Individual component count per plane
    const int app_dim = basis_mat_weight.size(0);

#if DEBUG_MODE
    // Print calculation details for debugging
    printf("[DEBUG] Tensor size calculations:\n");
    printf("  density_planes.size(0) = %ld\n", density_planes.size(0));
    printf("  density_planes.size(0) / 3 = %d\n", (int)(density_planes.size(0) / 3));
    printf("  app_planes.size(0) = %ld\n", app_planes.size(0));
    printf("  app_planes.size(0) / 3 = %d\n", (int)(app_planes.size(0) / 3));
    printf("  Final: density_n_comp=%d, app_n_comp=%d\n", density_n_comp, app_n_comp);
#endif

#if DEBUG_MODE
    // Print tensor dimensions for debugging
    printf("[DEBUG] Forward function input tensor sizes:\n");
    printf("  rays: [%ld, %ld]\n", rays.size(0), rays.size(1));
    printf("  density_planes ndim: %d\n", density_planes.dim());
    if (density_planes.dim() >= 4)
    {
        printf("  density_planes: [%ld, %ld, %ld, %ld]\n", density_planes.size(0), density_planes.size(1), density_planes.size(2), density_planes.size(3));
    }
    else if (density_planes.dim() == 3)
    {
        printf("  density_planes: [%ld, %ld, %ld]\n", density_planes.size(0), density_planes.size(1), density_planes.size(2));
    }
    printf("  density_lines ndim: %d\n", density_lines.dim());
    if (density_lines.dim() >= 3)
    {
        printf("  density_lines: [%ld, %ld, %ld]\n", density_lines.size(0), density_lines.size(1), density_lines.size(2));
    }
    else if (density_lines.dim() == 2)
    {
        printf("  density_lines: [%ld, %ld]\n", density_lines.size(0), density_lines.size(1));
    }
    printf("  app_planes ndim: %d\n", app_planes.dim());
    if (app_planes.dim() >= 4)
    {
        printf("  app_planes: [%ld, %ld, %ld, %ld]\n", app_planes.size(0), app_planes.size(1), app_planes.size(2), app_planes.size(3));
    }
    else if (app_planes.dim() == 3)
    {
        printf("  app_planes: [%ld, %ld, %ld]\n", app_planes.size(0), app_planes.size(1), app_planes.size(2));
    }
    printf("  app_lines ndim: %d\n", app_lines.dim());
    if (app_lines.dim() >= 3)
    {
        printf("  app_lines: [%ld, %ld, %ld]\n", app_lines.size(0), app_lines.size(1), app_lines.size(2));
    }
    else if (app_lines.dim() == 2)
    {
        printf("  app_lines: [%ld, %ld]\n", app_lines.size(0), app_lines.size(1));
    }
    printf("  basis_mat_weight: [%ld, %ld]\n", basis_mat_weight.size(0), basis_mat_weight.size(1));
    printf("  grid_size tensor: [%ld]\n", grid_size.size(0));
    printf("  CORRECTED Computed: n_rays=%d, density_n_comp=%d (total=%ld), app_n_comp=%d (total=%ld), app_dim=%d\n",
           n_rays, density_n_comp, density_planes.size(0), app_n_comp, app_planes.size(0), app_dim);

    // Print grid size values
    auto grid_size_cpu = grid_size.cpu();
    auto grid_size_accessor = grid_size_cpu.accessor<int, 1>();
    printf("  Grid dimensions: [%d, %d, %d]\n",
           grid_size_accessor[0], grid_size_accessor[1], grid_size_accessor[2]);
#endif

    // Allocate output tensors with gradient support
    // Create tensors that inherit gradient requirements from input tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(rays.device());
    torch::Tensor rgb_output = torch::zeros({n_rays, 3}, options);
    torch::Tensor depth_output = torch::zeros({n_rays}, options);
    
    // Enable gradients if any input tensor requires gradients
    bool requires_grad = rays.requires_grad() || density_planes.requires_grad() || 
                        density_lines.requires_grad() || app_planes.requires_grad() || 
                        app_lines.requires_grad() || basis_mat_weight.requires_grad();
    
    if (requires_grad) {
        rgb_output.requires_grad_(true);
        depth_output.requires_grad_(true);
    }

    // Setup random states for training
    curandState *rand_states = nullptr;
    if (is_train)
    {
        cudaMalloc(&rand_states, n_rays * sizeof(curandState));
        const int blocks_rand = (n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_curand_kernel<<<blocks_rand, BLOCK_SIZE>>>(rand_states, time(NULL), n_rays);
        cudaDeviceSynchronize();
    }

    // Launch kernel
    const int blocks = (n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Handle optional bias matrix pointer
    float *bias_ptr;
    if (basis_mat_bias.numel() > 0)
    {
        bias_ptr = basis_mat_bias.data_ptr<float>();
    }
    else
    {
        bias_ptr = nullptr;
    }

    fused_ray_render_kernel<<<blocks, BLOCK_SIZE>>>(
        rays.data_ptr<float>(),
        density_planes.data_ptr<float>(),
        density_lines.data_ptr<float>(),
        app_planes.data_ptr<float>(),
        app_lines.data_ptr<float>(),
        basis_mat_weight.data_ptr<float>(),
        bias_ptr,
        aabb.data_ptr<float>(),
        grid_size.data_ptr<int>(),
        step_size,
        n_samples,
        white_bg,
        is_train,
        distance_scale,
        ray_march_weight_thres,
        density_shift,                    // Pass density_shift to kernel
        n_rays,
        density_n_comp,
        app_n_comp,
        app_dim,
        rgb_output.data_ptr<float>(),
        depth_output.data_ptr<float>(),
        rand_states);

    // Cleanup
    if (rand_states)
    {
        cudaFree(rand_states);
    }

    cudaDeviceSynchronize();

    // Return RGB and depth as separate tensors in a vector
    return {rgb_output, depth_output};
}
