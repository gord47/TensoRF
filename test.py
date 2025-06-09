import torch
import torch.nn.functional as F
import numpy as np
from cuda.fused_plane_line import forward_split

def test_cuda_kernel():
    # Set up test parameters
    device = "cuda"
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*80)
    print("Testing CUDA Kernel vs PyTorch Implementation")
    print("="*80)
    
    # Create synthetic data matching TensoRF setup
    N = 1000  # Number of sample points
    C = 12    # Feature channels
    H, W = 128, 128  # Plane resolution
    L = 64           # Line resolution
    
    # Create dummy density planes and lines
    density_plane = [
        torch.randn(1, C, H, W, device=device, requires_grad=True),
        torch.randn(1, C, H, W, device=device, requires_grad=True),
        torch.randn(1, C, H, W, device=device, requires_grad=True)
    ]
    
    density_line = [
        torch.randn(1, C, L, 1, device=device, requires_grad=True),
        torch.randn(1, C, L, 1, device=device, requires_grad=True),
        torch.randn(1, C, L, 1, device=device, requires_grad=True)
    ]
    
    # Create sample points in [-1.5, 1.5] range (typical aabb)
    xyz_sampled = torch.rand(N, 3, device=device) * 3.0 - 1.5
    
    # Set matMode and vecMode as in TensoRF
    matMode = [[0, 1], [0, 2], [1, 2]]
    vecMode = [2, 1, 0]
    
    print("\nCreated synthetic data:")
    print(f"- Density planes: {[p.shape for p in density_plane]}")
    print(f"- Density lines: {[l.shape for l in density_line]}")
    print(f"- Sample points: {xyz_sampled.shape}, range: [{xyz_sampled.min().item():.3f}, {xyz_sampled.max().item():.3f}]")
    
    # Run original PyTorch implementation
    with torch.no_grad():
        # Prepare coordinates as in original implementation
        coordinate_plane = torch.stack((
            xyz_sampled[..., matMode[0]],
            xyz_sampled[..., matMode[1]],
            xyz_sampled[..., matMode[2]]
        )).detach().view(3, -1, 1, 2)
        
        coordinate_line = torch.stack((
            xyz_sampled[..., vecMode[0]],
            xyz_sampled[..., vecMode[1]],
            xyz_sampled[..., vecMode[2]]
        ))
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).detach().view(3, -1, 1, 2)
        
        sigma_feature_pytorch = torch.zeros((xyz_sampled.shape[0],), device=device)
        
        for idx_plane in range(3):
            plane_coef_point = F.grid_sample(
                density_plane[idx_plane], 
                coordinate_plane[[idx_plane]],
                align_corners=True
            ).view(-1, N)
            
            line_coef_point = F.grid_sample(
                density_line[idx_plane], 
                coordinate_line[[idx_plane]],
                align_corners=True
            ).view(-1, N)
            
            sigma_feature_pytorch += torch.sum(plane_coef_point * line_coef_point, dim=0)
    
    print("\nPyTorch implementation results:")
    print(f"- Min: {sigma_feature_pytorch.min().item():.6f}")
    print(f"- Max: {sigma_feature_pytorch.max().item():.6f}")
    print(f"- Mean: {sigma_feature_pytorch.mean().item():.6f}")
    print(f"- Std: {sigma_feature_pytorch.std().item():.6f}")
    
    # Run CUDA kernel implementation
    coord_planes = []
    coord_lines = []
    
    for i in range(3):
        coord_planes.append(xyz_sampled[..., matMode[i]].detach())
        coord_lines.append(xyz_sampled[..., vecMode[i]].detach())
    
    coords_plane_stacked = torch.stack(coord_planes, dim=0)  # [3, N, 2]
    coords_line_stacked = torch.stack(coord_lines, dim=0)    # [3, N]
    
    # Clamp coordinates to safe range
    coords_plane_stacked = coords_plane_stacked.clamp(-1.0, 1.0)
    coords_line_stacked = coords_line_stacked.clamp(-1.0, 1.0)
    
    with torch.no_grad():
        sigma_feature_cuda = forward_split(
            density_plane,
            density_line,
            coords_plane_stacked,
            coords_line_stacked
        )
    
    print("\nCUDA kernel implementation results:")
    print(f"- Min: {sigma_feature_cuda.min().item():.6f}")
    print(f"- Max: {sigma_feature_cuda.max().item():.6f}")
    print(f"- Mean: {sigma_feature_cuda.mean().item():.6f}")
    print(f"- Std: {sigma_feature_cuda.std().item():.6f}")
    
    # Compare results
    abs_diff = (sigma_feature_pytorch - sigma_feature_cuda).abs()
    rel_diff = abs_diff / (sigma_feature_pytorch.abs() + 1e-9)
    
    print("\nComparison results:")
    print(f"- Max absolute difference: {abs_diff.max().item():.6f}")
    print(f"- Max relative difference: {rel_diff.max().item():.6f}")
    print(f"- Mean absolute difference: {abs_diff.mean().item():.6f}")
    print(f"- Mean relative difference: {rel_diff.mean().item():.6f}")
    
    # Check for significant differences
    tolerance = 1e-4
    if abs_diff.max() < tolerance:
        print("\n✅ Test PASSED: Results match within tolerance!")
    else:
        print(f"\n❌ Test FAILED: Differences exceed tolerance ({tolerance})!")
        
        # Find problematic points
        bad_indices = torch.where(abs_diff > tolerance)[0]
        print(f"\nFound {len(bad_indices)} problematic points. Example:")
        for i in bad_indices[:5]:
            print(f"Point {i}:")
            print(f"  PyTorch: {sigma_feature_pytorch[i].item():.6f}")
            print(f"  CUDA:    {sigma_feature_cuda[i].item():.6f}")
            print(f"  Coords:  {xyz_sampled[i].cpu().numpy()}")
            print(f"  Abs diff: {abs_diff[i].item():.6f}")
            print(f"  Rel diff: {rel_diff[i].item():.6f}")

if __name__ == "__main__":
    test_cuda_kernel()