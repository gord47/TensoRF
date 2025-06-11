import torch
import torch.nn.functional as F
import cuda.grid_sample as grid_sample_cuda
import torch.cuda.nvtx as nvtx
import time

# Parameters
device = "cuda"
N = 1024  # Number of points
grid_size = 128  # Grid resolution

# Create input tensors similar to tensoRF
density_plane = torch.rand(3, 1, 1, grid_size, grid_size, device=device)
density_line = torch.rand(3, 1, 1, grid_size, 1, device=device)

# Create coordinates (similar to tensoRF's coordinate system)
xyz_sampled = torch.rand(N, 3, device=device) * 2 - 1  # Normalized to [-1, 1]

# Create grid for planes (XY, XZ, YZ)
coordinate_plane = torch.stack((
    xyz_sampled[..., [0, 1]],  # XY plane
    xyz_sampled[..., [0, 2]],  # XZ plane
    xyz_sampled[..., [1, 2]]   # YZ plane
)).detach().view(3, -1, 1, 2)

# Create grid for lines
coordinate_line = torch.stack((
    xyz_sampled[..., [2]],     # Z dimension
    xyz_sampled[..., [1]],     # Y dimension
    xyz_sampled[..., [0]]      # X dimension
))
coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

# Custom CUDA implementation
sigma_feature_custom = torch.zeros((xyz_sampled.shape[0],), device=device)

torch.cuda.synchronize()
nvtx.range_push("custom_cuda_grid_sample")
t0 = time.time()
for idx_plane in range(3):
    plane_coef_point = grid_sample_cuda.forward(
        density_plane[idx_plane], 
        coordinate_plane[idx_plane].unsqueeze(0),
        True  # align_corners=True
    ).view(-1, N)
    
    line_coef_point = grid_sample_cuda.forward(
        density_line[idx_plane], 
        coordinate_line[idx_plane].unsqueeze(0),
        True  # align_corners=True
    ).view(-1, N)
    
    sigma_feature_custom += torch.sum(plane_coef_point * line_coef_point, dim=0)
torch.cuda.synchronize()
t1 = time.time()
nvtx.range_pop()
print(f"Custom CUDA grid_sample time: {t1-t0:.6f} seconds")

# PyTorch's implementation
sigma_feature_pytorch = torch.zeros((xyz_sampled.shape[0],), device=device)

torch.cuda.synchronize()
nvtx.range_push("pytorch_grid_sample")
t0 = time.time()
for idx_plane in range(3):
    plane_coef_point = F.grid_sample(
        density_plane[idx_plane], 
        coordinate_plane[idx_plane].unsqueeze(0),
        align_corners=True
    ).view(-1, N)
    
    line_coef_point = F.grid_sample(
        density_line[idx_plane], 
        coordinate_line[idx_plane].unsqueeze(0),
        align_corners=True
    ).view(-1, N)
    
    sigma_feature_pytorch += torch.sum(plane_coef_point * line_coef_point, dim=0)
torch.cuda.synchronize()
t1 = time.time()
nvtx.range_pop()
print(f"PyTorch grid_sample time: {t1-t0:.6f} seconds")

print("\nCUDA kernel implementation results:")
print(f"- Min: {sigma_feature_custom.min().item():.6f}")
print(f"- Max: {sigma_feature_custom.max().item():.6f}")
print(f"- Mean: {sigma_feature_custom.mean().item():.6f}")
print(f"- Std: {sigma_feature_custom.std().item():.6f}")

print("\nPytorch implementation results:")
print(f"- Min: {sigma_feature_pytorch.min().item():.6f}")
print(f"- Max: {sigma_feature_pytorch.max().item():.6f}")
print(f"- Mean: {sigma_feature_pytorch.mean().item():.6f}")
print(f"- Std: {sigma_feature_pytorch.std().item():.6f}")
# Compare results
diff = torch.abs(sigma_feature_custom - sigma_feature_pytorch)
print("Max difference:", torch.max(diff).item())
print("Mean difference:", torch.mean(diff).item())

# Check for NaN
assert not torch.isnan(sigma_feature_custom).any(), "Custom output contains NaNs"
assert not torch.isnan(sigma_feature_pytorch).any(), "PyTorch output contains NaNs"