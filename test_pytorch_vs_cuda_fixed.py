#!/usr/bin/env python3

"""
Test script to compare PyTorch vs CUDA implementations with identical parameters.
This will help us understand why PyTorch works with density_shift=-10 and scale=0.1
while CUDA requires our initialization fix.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from models.tensoRF import TensorVMSplit

def create_test_model():
    """Create a test TensorRF model with standard LEGO parameters"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Standard LEGO parameters
    aabb = torch.tensor([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]], device=device)
    gridSize = [128, 128, 128]
    
    model = TensorVMSplit(
        aabb=aabb,
        gridSize=gridSize,
        device=device,  # Required parameter
        density_n_comp=[16, 16, 16],
        appearance_n_comp=[48, 48, 48],
        app_dim=27,
        near_far=[0.5, 3.5],
        shadingMode='MLP_PE',
        alphaMask_thres=0.001,
        density_shift=-10,  # The problematic parameter
        distance_scale=25,
        pos_pe=6,
        view_pe=6,
        fea_pe=6,
        featureC=128,
        step_ratio=2.0
        # Note: TensorVMSplit uses hardcoded 0.1 initialization scale in init_svd_volume
    ).to(device)
    
    return model

def test_alpha_computation():
    """Test alpha computation with both implementations"""
    print("=" * 80)
    print("TESTING PYTORCH vs CUDA RENDERER COMPARISON")
    print("=" * 80)
    
    print("Testing PyTorch vs CUDA with density_shift=-10 and hardcoded init_scale=0.1")
    
    model = create_test_model()
    
    # Create test rays
    device = model.aabb.device
    rays = torch.randn(1000, 6, device=device)  # [N, 6] - origins and directions
    rays[:, :3] = torch.randn(1000, 3, device=device) * 0.5  # origins near center
    rays[:, 3:] = torch.randn(1000, 3, device=device)  # directions
    rays[:, 3:] = rays[:, 3:] / torch.norm(rays[:, 3:], dim=1, keepdim=True)  # normalize directions
    
    print(f"Created model with density_shift={model.density_shift}")
    print(f"Model device: {device}")
    print(f"Note: TensorVMSplit uses hardcoded initialization scale of 0.1")
    
    # Test PyTorch implementation
    print("\n1. Testing PyTorch Implementation:")
    try:
        os.environ['USE_PYTORCH_RENDERER'] = '1'
        os.environ['USE_CUDA_RENDERER'] = '0'
        
        from renderer import OctreeRender_trilinear_fast
        
        with torch.no_grad():
            rgb_pytorch, _, depth_pytorch, _, _ = OctreeRender_trilinear_fast(
                rays, model, chunk=1000, N_samples=64, 
                white_bg=True, is_train=False, device=device
            )
        
        pytorch_stats = {
            'rgb_mean': rgb_pytorch.mean().item(),
            'rgb_std': rgb_pytorch.std().item(),
            'depth_mean': depth_pytorch.mean().item(),
            'depth_std': depth_pytorch.std().item(),
            'rgb_range': [rgb_pytorch.min().item(), rgb_pytorch.max().item()],
            'depth_range': [depth_pytorch.min().item(), depth_pytorch.max().item()]
        }
        
        print(f"  ✓ PyTorch Success:")
        print(f"    RGB: mean={pytorch_stats['rgb_mean']:.6f}, std={pytorch_stats['rgb_std']:.6f}")
        print(f"    RGB range: [{pytorch_stats['rgb_range'][0]:.3f}, {pytorch_stats['rgb_range'][1]:.3f}]")
        print(f"    Depth: mean={pytorch_stats['depth_mean']:.6f}, std={pytorch_stats['depth_std']:.6f}")
        
    except Exception as e:
        print(f"  ✗ PyTorch Failed: {e}")
        pytorch_stats = None
    
    # Test CUDA implementation
    print("\n2. Testing CUDA Implementation:")
    try:
        os.environ['USE_PYTORCH_RENDERER'] = '0'
        os.environ['USE_CUDA_RENDERER'] = '1'
        
        # Reload the module to pick up environment changes
        import importlib
        import renderer
        importlib.reload(renderer)
        from renderer import OctreeRender_trilinear_fast
        
        with torch.no_grad():
            rgb_cuda, _, depth_cuda, _, _ = OctreeRender_trilinear_fast(
                rays, model, chunk=1000, N_samples=64,
                white_bg=True, is_train=False, device=device
            )
        
        cuda_stats = {
            'rgb_mean': rgb_cuda.mean().item(),
            'rgb_std': rgb_cuda.std().item(), 
            'depth_mean': depth_cuda.mean().item(),
            'depth_std': depth_cuda.std().item(),
            'rgb_range': [rgb_cuda.min().item(), rgb_cuda.max().item()],
            'depth_range': [depth_cuda.min().item(), depth_cuda.max().item()]
        }
        
        print(f"  ✓ CUDA Success:")
        print(f"    RGB: mean={cuda_stats['rgb_mean']:.6f}, std={cuda_stats['rgb_std']:.6f}")
        print(f"    RGB range: [{cuda_stats['rgb_range'][0]:.3f}, {cuda_stats['rgb_range'][1]:.3f}]")
        print(f"    Depth: mean={cuda_stats['depth_mean']:.6f}, std={cuda_stats['depth_std']:.6f}")
        
    except Exception as e:
        print(f"  ✗ CUDA Failed: {e}")
        cuda_stats = None
    
    # Compare results
    if pytorch_stats and cuda_stats:
        print(f"\n3. Comparison:")
        rgb_diff = abs(pytorch_stats['rgb_mean'] - cuda_stats['rgb_mean'])
        depth_diff = abs(pytorch_stats['depth_mean'] - cuda_stats['depth_mean'])
        print(f"  RGB mean difference: {rgb_diff:.6f}")
        print(f"  Depth mean difference: {depth_diff:.6f}")
        
        if rgb_diff < 1e-3 and depth_diff < 1e-3:
            print(f"  ✓ Results are very similar!")
        elif rgb_diff < 1e-2 and depth_diff < 1e-2:
            print(f"  ~ Results are reasonably similar")
        else:
            print(f"  ✗ Results are significantly different!")
    
    print("-" * 60)

def test_density_features():
    """Test the density feature computation directly"""
    print("\n" + "=" * 80)
    print("TESTING DENSITY FEATURE COMPUTATION")
    print("=" * 80)
    
    # Test with the problematic parameters
    model = create_test_model()
    device = model.aabb.device
    
    # Sample some points in the scene
    test_points = torch.rand(100, 3, device=device) * 2 - 1  # [-1, 1]^3
    test_points = test_points.to(device)
    
    print(f"Testing density computation at {test_points.shape[0]} points")
    print(f"Model density_shift: {model.density_shift}")
    
    with torch.no_grad():
        # Get density features
        density_features = model.compute_densityfeature(test_points)
        print(f"Raw density features: mean={density_features.mean():.6f}, std={density_features.std():.6f}")
        print(f"Raw density range: [{density_features.min():.6f}, {density_features.max():.6f}]")
        
        # Apply the same transformation as feature2density
        shifted_features = density_features + model.density_shift
        print(f"After density_shift: mean={shifted_features.mean():.6f}, std={shifted_features.std():.6f}")
        print(f"Shifted range: [{shifted_features.min():.6f}, {shifted_features.max():.6f}]")
        
        # Apply softplus
        sigma = torch.nn.functional.softplus(shifted_features)
        print(f"After softplus (sigma): mean={sigma.mean():.6f}, std={sigma.std():.6f}")
        print(f"Sigma range: [{sigma.min():.6f}, {sigma.max():.6f}]")
        
        # Check how many are effectively zero
        near_zero = (sigma < 1e-5).sum().item()
        print(f"Near-zero sigmas (< 1e-5): {near_zero}/{len(sigma)} ({100*near_zero/len(sigma):.1f}%)")

if __name__ == "__main__":
    print("PyTorch vs CUDA Renderer Comparison Test")
    print("This will help explain why PyTorch works with density_shift=-10 and scale=0.1")
    
    # Reset environment
    os.environ.pop('USE_PYTORCH_RENDERER', None)
    os.environ.pop('USE_CUDA_RENDERER', None)
    
    try:
        test_alpha_computation()
        test_density_features()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("The test results above show the differences between PyTorch and CUDA implementations.")
        print("Key things to look for:")
        print("1. Do both implementations produce similar RGB/depth values?")
        print("2. Are the density features computed similarly?")
        print("3. Does PyTorch handle very small density values differently than CUDA?")
        print("4. Is there a numerical precision difference causing the training issues?")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up environment
        os.environ.pop('USE_PYTORCH_RENDERER', None)
        os.environ.pop('USE_CUDA_RENDERER', None)
