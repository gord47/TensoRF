import torch
import torch.nn.functional as F
import grid_sample_cuda

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
)).detach().view(3, N, 1, 2)

# Create grid for lines
coordinate_line = torch.stack((
    xyz_sampled[..., [2]],     # Z dimension
    xyz_sampled[..., [1]],     # Y dimension
    xyz_sampled[..., [0]]      # X dimension
)).detach().view(3, N, 1, 1)

# Custom CUDA implementation
sigma_feature_custom = torch.zeros((xyz_sampled.shape[0],), device=device)

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

# PyTorch's implementation
sigma_feature_pytorch = torch.zeros((xyz_sampled.shape[0],), device=device)

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

# Compare results
diff = torch.abs(sigma_feature_custom - sigma_feature_pytorch)
print("Max difference:", torch.max(diff).item())
print("Mean difference:", torch.mean(diff).item())

# Check for NaN
assert not torch.isnan(sigma_feature_custom).any(), "Custom output contains NaNs"
assert not torch.isnan(sigma_feature_pytorch).any(), "PyTorch output contains NaNs"