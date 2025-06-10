#!/usr/bin/env python3

"""
Debug script to understand exactly how PyTorch computes appearance features
and what our CUDA implementation should match.
"""

import torch
import torch.nn.functional as F
from models.tensoRF import TensorVMSplit

def debug_appearance_features():
    """Debug the appearance feature computation step by step"""
    print("=" * 80)
    print("DEBUGGING APPEARANCE FEATURE COMPUTATION")
    print("=" * 80)
    
    # Create test model with LEGO config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    aabb = torch.tensor([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]], device=device)
    gridSize = [128, 128, 128]
    
    model = TensorVMSplit(
        aabb=aabb,
        gridSize=gridSize,
        device=device,
        density_n_comp=[16, 16, 16],
        appearance_n_comp=[48, 48, 48],  # This is the key config
        app_dim=27,
        near_far=[0.5, 3.5],
        shadingMode='MLP_PE',
        alphaMask_thres=0.001,
        density_shift=-10,
        distance_scale=25,
        pos_pe=6,
        view_pe=6,
        fea_pe=6,
        featureC=128,
        step_ratio=2.0
    ).to(device)
    
    print(f"Model config:")
    print(f"  app_n_comp: {model.app_n_comp}")
    print(f"  sum(app_n_comp): {sum(model.app_n_comp)}")
    print(f"  app_dim: {model.app_dim}")
    print(f"  basis_mat input features: {model.basis_mat.in_features}")
    print(f"  basis_mat output features: {model.basis_mat.out_features}")
    
    # Check plane and line shapes
    print(f"\nPlane and line shapes:")
    for i, (plane, line) in enumerate(zip(model.app_plane, model.app_line)):
        print(f"  app_plane[{i}]: {plane.shape}")
        print(f"  app_line[{i}]: {line.shape}")
    
    # Create test points
    test_points = torch.rand(100, 3, device=device) * 2 - 1  # [-1, 1]^3
    
    print(f"\nTesting with {test_points.shape[0]} points...")
    
    # Step-by-step feature computation (matching PyTorch implementation)
    coordinate_plane = torch.stack((
        test_points[..., model.matMode[0]], 
        test_points[..., model.matMode[1]], 
        test_points[..., model.matMode[2]]
    )).detach().view(3, -1, 1, 2)
    
    coordinate_line = torch.stack((
        test_points[..., model.vecMode[0]], 
        test_points[..., model.vecMode[1]], 
        test_points[..., model.vecMode[2]]
    ))
    coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
    
    print(f"  coordinate_plane shape: {coordinate_plane.shape}")
    print(f"  coordinate_line shape: {coordinate_line.shape}")
    
    # Process each plane-line pair
    plane_coef_point, line_coef_point = [], []
    for idx_plane in range(len(model.app_plane)):
        plane_feat = F.grid_sample(model.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                 align_corners=True).view(-1, *test_points.shape[:1])
        line_feat = F.grid_sample(model.app_line[idx_plane], coordinate_line[[idx_plane]],
                                align_corners=True).view(-1, *test_points.shape[:1])
        
        plane_coef_point.append(plane_feat)
        line_coef_point.append(line_feat)
        
        print(f"  Plane {idx_plane} - plane_feat shape: {plane_feat.shape}")
        print(f"  Plane {idx_plane} - line_feat shape: {line_feat.shape}")
    
    # Concatenate features
    plane_coef_point = torch.cat(plane_coef_point)
    line_coef_point = torch.cat(line_coef_point)
    
    print(f"\nAfter concatenation:")
    print(f"  plane_coef_point shape: {plane_coef_point.shape}")
    print(f"  line_coef_point shape: {line_coef_point.shape}")
    
    # Element-wise multiplication
    combined_features = plane_coef_point * line_coef_point
    print(f"  combined_features shape: {combined_features.shape}")
    
    # Transpose for basis matrix
    transposed_features = combined_features.T
    print(f"  transposed_features shape: {transposed_features.shape}")
    
    # Check if shapes match basis matrix expectation
    expected_input_features = model.basis_mat.in_features
    actual_input_features = transposed_features.shape[1]
    
    print(f"\nBasis matrix compatibility:")
    print(f"  Expected input features: {expected_input_features}")
    print(f"  Actual input features: {actual_input_features}")
    
    if expected_input_features == actual_input_features:
        print(f"  ✓ Shapes are compatible!")
        
        # Apply basis matrix
        result = model.basis_mat(transposed_features)
        print(f"  Result shape: {result.shape}")
        print(f"  Result mean: {result.mean().item():.6f}")
        print(f"  Result std: {result.std().item():.6f}")
        
    else:
        print(f"  ✗ Shape mismatch! This would cause an error.")
        print(f"  Difference: {actual_input_features - expected_input_features}")
    
    return {
        'expected_features': expected_input_features,
        'actual_features': actual_input_features,
        'app_n_comp': model.app_n_comp,
        'sum_app_n_comp': sum(model.app_n_comp)
    }

def debug_pytorch_vs_cuda_difference():
    """Check why PyTorch works but CUDA has issues"""
    print("\n" + "=" * 80)
    print("DEBUGGING PYTORCH vs CUDA DIFFERENCES")
    print("=" * 80)
    
    info = debug_appearance_features()
    
    print(f"\nAnalysis:")
    print(f"  Config app_n_comp: {info['app_n_comp']}")
    print(f"  Sum of app_n_comp: {info['sum_app_n_comp']}")
    print(f"  Basis matrix expects: {info['expected_features']} features")
    print(f"  PyTorch produces: {info['actual_features']} features")
    
    if info['expected_features'] == info['actual_features']:
        print(f"  ✓ PyTorch implementation is consistent")
        print(f"  → CUDA should produce the same {info['actual_features']} features")
    else:
        print(f"  ✗ There's an inconsistency in the PyTorch model")
        print(f"  → Need to investigate further")
    
    print(f"\nKey insight:")
    print(f"  If PyTorch achieves good PSNR with this feature count,")
    print(f"  then CUDA should match this EXACT behavior, not fix it!")

if __name__ == "__main__":
    try:
        debug_pytorch_vs_cuda_difference()
        
        print(f"\n" + "=" * 80)
        print(f"CONCLUSION")
        print(f"=" * 80)
        print(f"This script reveals exactly how PyTorch computes appearance features.")
        print(f"Our CUDA implementation should match this behavior precisely,")
        print(f"including any apparent 'inconsistencies' that actually work.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
