#!/usr/bin/env python3
"""
Comprehensive numerical agreement test between CUDA fused kernel and PyTorch reference.
This script performs detailed comparison across different scenarios to ensure correctness.
"""

import torch
import cuda.fused_plane_line as fused_plane_line
import torch.nn.functional as F
import time
import numpy as np

def compare_implementations(planes, lines, coordinate_plane2, coordinate_line, test_name="Test", verbose=True):
    """Compare CUDA kernel vs PyTorch reference implementation"""
    
    N = coordinate_plane2.shape[1]
    device = planes.device
    
    # CUDA fused kernel
    torch.cuda.synchronize()
    t0 = time.time()
    result_cuda = fused_plane_line.forward(
        planes.contiguous(),
        lines.contiguous(),
        coordinate_plane2.contiguous(),
        coordinate_line.contiguous()
    )[0]
    torch.cuda.synchronize()
    t1 = time.time()
    cuda_time = t1 - t0
    
    # PyTorch reference (batched version)
    torch.cuda.synchronize()
    t0 = time.time()
    coordinate_plane_grid = coordinate_plane2.view(3, N, 1, 2)
    coordinate_line_grid = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, N, 1, 2)
    
    plane_coef_point = F.grid_sample(planes, coordinate_plane_grid, align_corners=True).view(3, -1, N)
    line_coef_point = F.grid_sample(lines, coordinate_line_grid, align_corners=True).view(3, -1, N)
    result_pytorch = torch.sum(plane_coef_point * line_coef_point, dim=0).sum(0)
    torch.cuda.synchronize()
    t1 = time.time()
    pytorch_time = t1 - t0
    
    # Compute differences
    abs_diff = torch.abs(result_cuda - result_pytorch)
    rel_diff = abs_diff / (torch.abs(result_pytorch) + 1e-8)  # Add small epsilon to avoid division by zero
    
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()
    
    # Statistical measures
    cuda_min, cuda_max = result_cuda.min().item(), result_cuda.max().item()
    pytorch_min, pytorch_max = result_pytorch.min().item(), result_pytorch.max().item()
    cuda_mean, cuda_std = result_cuda.mean().item(), result_cuda.std().item()
    pytorch_mean, pytorch_std = result_pytorch.mean().item(), result_pytorch.std().item()
    
    # Check for NaNs and Infs
    cuda_has_nan = torch.isnan(result_cuda).any().item()
    cuda_has_inf = torch.isinf(result_cuda).any().item()
    pytorch_has_nan = torch.isnan(result_pytorch).any().item()
    pytorch_has_inf = torch.isinf(result_pytorch).any().item()
    
    # Tolerance checks
    close_atol_1e5 = torch.allclose(result_cuda, result_pytorch, rtol=1e-5, atol=1e-5)
    close_atol_1e6 = torch.allclose(result_cuda, result_pytorch, rtol=1e-6, atol=1e-6)
    close_atol_1e7 = torch.allclose(result_cuda, result_pytorch, rtol=1e-7, atol=1e-7)
    
    if verbose:
        print(f"\n=== {test_name} ===")
        print(f"Input shapes: planes={planes.shape}, lines={lines.shape}")
        print(f"Coordinate shapes: plane_coords={coordinate_plane2.shape}, line_coords={coordinate_line.shape}")
        print(f"Output shape: {result_cuda.shape}")
        
        print(f"\nTiming:")
        print(f"  CUDA time: {cuda_time:.6f}s")
        print(f"  PyTorch time: {pytorch_time:.6f}s")
        if cuda_time > 0:
            print(f"  Speedup: {pytorch_time/cuda_time:.2f}x")
        
        print(f"\nNumerical Statistics:")
        print(f"  CUDA    - Min: {cuda_min:.6f}, Max: {cuda_max:.6f}, Mean: {cuda_mean:.6f}, Std: {cuda_std:.6f}")
        print(f"  PyTorch - Min: {pytorch_min:.6f}, Max: {pytorch_max:.6f}, Mean: {pytorch_mean:.6f}, Std: {pytorch_std:.6f}")
        
        print(f"\nDifferences:")
        print(f"  Max absolute difference: {max_abs_diff:.10f}")
        print(f"  Mean absolute difference: {mean_abs_diff:.10f}")
        print(f"  Max relative difference: {max_rel_diff:.10f}")
        print(f"  Mean relative difference: {mean_rel_diff:.10f}")
        
        print(f"\nData Quality:")
        print(f"  CUDA NaN: {cuda_has_nan}, Inf: {cuda_has_inf}")
        print(f"  PyTorch NaN: {pytorch_has_nan}, Inf: {pytorch_has_inf}")
        
        print(f"\nTolerance Tests:")
        print(f"  Close (rtol=1e-5, atol=1e-5): {'✓' if close_atol_1e5 else '✗'}")
        print(f"  Close (rtol=1e-6, atol=1e-6): {'✓' if close_atol_1e6 else '✗'}")
        print(f"  Close (rtol=1e-7, atol=1e-7): {'✓' if close_atol_1e7 else '✗'}")
        
        # Show some sample values for debugging
        print(f"\nSample Values (first 5):")
        for i in range(min(5, len(result_cuda))):
            print(f"  [{i}] CUDA: {result_cuda[i].item():.10f}, PyTorch: {result_pytorch[i].item():.10f}, Diff: {abs_diff[i].item():.10f}")
    
    return {
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'cuda_time': cuda_time,
        'pytorch_time': pytorch_time,
        'close_1e5': close_atol_1e5,
        'close_1e6': close_atol_1e6,
        'close_1e7': close_atol_1e7,
        'has_nan_inf': cuda_has_nan or cuda_has_inf or pytorch_has_nan or pytorch_has_inf
    }

def test_random_data(device="cuda"):
    """Test with random data - basic functionality"""
    N, grid_size, C = 1024, 128, 8
    
    planes = torch.rand(3, C, grid_size, grid_size, device=device)
    lines = torch.rand(3, C, grid_size, 1, device=device)
    xyz_sampled = torch.rand(N, 3, device=device) * 2 - 1
    
    # Prepare coordinates using TensoRF's matMode and vecMode mapping
    # matMode = [[0,1], [0,2], [1,2]] and vecMode = [2, 1, 0]
    matMode = [[0,1], [0,2], [1,2]]
    vecMode = [2, 1, 0]
    
    coordinate_plane = torch.stack([
        xyz_sampled[..., matMode[0]],  # [0,1] for XY plane
        xyz_sampled[..., matMode[1]],  # [0,2] for XZ plane
        xyz_sampled[..., matMode[2]]   # [1,2] for YZ plane
    ], dim=0).detach()
    coordinate_line = torch.stack([
        xyz_sampled[..., vecMode[0]],  # Z coordinate
        xyz_sampled[..., vecMode[1]],  # Y coordinate
        xyz_sampled[..., vecMode[2]]   # X coordinate
    ], dim=0).detach()
    coordinate_plane2 = torch.stack([
        torch.stack((xyz_sampled[..., matMode[i][0]], xyz_sampled[..., matMode[i][1]]), dim=-1)
        for i in range(3)
    ], dim=0)
    
    return compare_implementations(planes, lines, coordinate_plane2, coordinate_line, "Random Data")

def test_edge_values(device="cuda"):
    """Test with edge case values"""
    N, grid_size, C = 256, 64, 4
    
    test_cases = {
        "zeros": (torch.zeros(3, C, grid_size, grid_size, device=device),
                  torch.zeros(3, C, grid_size, 1, device=device)),
        "ones": (torch.ones(3, C, grid_size, grid_size, device=device),
                 torch.ones(3, C, grid_size, 1, device=device)),
        "small_values": (torch.rand(3, C, grid_size, grid_size, device=device) * 1e-6,
                        torch.rand(3, C, grid_size, 1, device=device) * 1e-6),
        "large_values": (torch.rand(3, C, grid_size, grid_size, device=device) * 1e3,
                        torch.rand(3, C, grid_size, 1, device=device) * 1e3),
        "mixed_signs": (torch.randn(3, C, grid_size, grid_size, device=device),
                       torch.randn(3, C, grid_size, 1, device=device)),
    }
    
    coord_cases = {
        "normal": torch.rand(N, 3, device=device) * 2 - 1,
        "boundary": torch.ones(N, 3, device=device),
        "negative_boundary": -torch.ones(N, 3, device=device),
        "beyond_boundary": torch.rand(N, 3, device=device) * 4 - 2,  # [-2, 2]
        "zero_coords": torch.zeros(N, 3, device=device),
    }
    
    results = []
    for data_name, (planes, lines) in test_cases.items():
        for coord_name, xyz_sampled in coord_cases.items():
            # Use correct TensoRF coordinate mapping
            matMode = [[0,1], [0,2], [1,2]]
            vecMode = [2, 1, 0]
            
            coordinate_plane = torch.stack([
                xyz_sampled[..., matMode[0]],  # [0,1] for XY plane
                xyz_sampled[..., matMode[1]],  # [0,2] for XZ plane
                xyz_sampled[..., matMode[2]]   # [1,2] for YZ plane
            ], dim=0).detach()
            coordinate_line = torch.stack([
                xyz_sampled[..., vecMode[0]],  # Z coordinate
                xyz_sampled[..., vecMode[1]],  # Y coordinate
                xyz_sampled[..., vecMode[2]]   # X coordinate
            ], dim=0).detach()
            coordinate_plane2 = torch.stack([
                torch.stack((xyz_sampled[..., matMode[i][0]], xyz_sampled[..., matMode[i][1]]), dim=-1)
                for i in range(3)
            ], dim=0)
            
            test_name = f"{data_name} + {coord_name}_coords"
            result = compare_implementations(planes, lines, coordinate_plane2, coordinate_line, test_name, verbose=False)
            results.append((test_name, result))
            
            # Print summary for each test
            status = "✓" if result['close_1e5'] and not result['has_nan_inf'] else "✗"
            print(f"{test_name:30} {status} Max diff: {result['max_abs_diff']:.2e}, Rel diff: {result['max_rel_diff']:.2e}")
    
    return results

def test_different_sizes(device="cuda"):
    """Test with different tensor sizes"""
    test_configs = [
        (64, 32, 2),     # Small
        (256, 64, 4),    # Medium
        (1024, 128, 8),  # Large
        (2048, 64, 16),  # Many channels
        (512, 256, 4),   # High resolution
    ]
    
    results = []
    for N, grid_size, C in test_configs:
        planes = torch.rand(3, C, grid_size, grid_size, device=device)
        lines = torch.rand(3, C, grid_size, 1, device=device)
        xyz_sampled = torch.rand(N, 3, device=device) * 2 - 1
        
        # Use correct TensoRF coordinate mapping
        matMode = [[0,1], [0,2], [1,2]]
        vecMode = [2, 1, 0]
        
        coordinate_plane = torch.stack([
            xyz_sampled[..., matMode[0]],  # [0,1] for XY plane
            xyz_sampled[..., matMode[1]],  # [0,2] for XZ plane
            xyz_sampled[..., matMode[2]]   # [1,2] for YZ plane
        ], dim=0).detach()
        coordinate_line = torch.stack([
            xyz_sampled[..., vecMode[0]],  # Z coordinate
            xyz_sampled[..., vecMode[1]],  # Y coordinate
            xyz_sampled[..., vecMode[2]]   # X coordinate
        ], dim=0).detach()
        coordinate_plane2 = torch.stack([
            torch.stack((xyz_sampled[..., matMode[i][0]], xyz_sampled[..., matMode[i][1]]), dim=-1)
            for i in range(3)
        ], dim=0)
        
        test_name = f"Size N={N}, grid={grid_size}, C={C}"
        result = compare_implementations(planes, lines, coordinate_plane2, coordinate_line, test_name, verbose=False)
        results.append((test_name, result))
        
        status = "✓" if result['close_1e5'] and not result['has_nan_inf'] else "✗"
        speedup = f"{result['pytorch_time']/result['cuda_time']:.1f}x" if result['cuda_time'] > 0 else "N/A"
        print(f"{test_name:25} {status} Max diff: {result['max_abs_diff']:.2e}, Speedup: {speedup}")
    
    return results

def print_summary(all_results):
    """Print overall summary"""
    print(f"\n" + "="*60)
    print(f"NUMERICAL AGREEMENT SUMMARY")
    print(f"="*60)
    
    total_tests = len(all_results)
    passed_1e5 = sum(1 for _, result in all_results if result['close_1e5'])
    passed_1e6 = sum(1 for _, result in all_results if result['close_1e6'])
    passed_1e7 = sum(1 for _, result in all_results if result['close_1e7'])
    
    max_abs_diffs = [result['max_abs_diff'] for _, result in all_results]
    max_rel_diffs = [result['max_rel_diff'] for _, result in all_results]
    
    print(f"Total tests: {total_tests}")
    print(f"Passed (rtol=1e-5, atol=1e-5): {passed_1e5}/{total_tests} ({100*passed_1e5/total_tests:.1f}%)")
    print(f"Passed (rtol=1e-6, atol=1e-6): {passed_1e6}/{total_tests} ({100*passed_1e6/total_tests:.1f}%)")
    print(f"Passed (rtol=1e-7, atol=1e-7): {passed_1e7}/{total_tests} ({100*passed_1e7/total_tests:.1f}%)")
    
    print(f"\nWorst case differences:")
    print(f"  Maximum absolute difference: {max(max_abs_diffs):.2e}")
    print(f"  Maximum relative difference: {max(max_rel_diffs):.2e}")
    print(f"  Median absolute difference: {np.median(max_abs_diffs):.2e}")
    print(f"  Median relative difference: {np.median(max_rel_diffs):.2e}")
    
    # Identify problematic tests
    problematic = [(name, result) for name, result in all_results if not result['close_1e5']]
    if problematic:
        print(f"\nProblematic tests (not close at 1e-5 tolerance):")
        for name, result in problematic:
            print(f"  {name}: max_abs_diff={result['max_abs_diff']:.2e}, max_rel_diff={result['max_rel_diff']:.2e}")
    else:
        print(f"\n✓ All tests passed at 1e-5 tolerance!")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA device not available. Tests will not run.")
        exit(1)
    
    print("CUDA Fused Kernel vs PyTorch Numerical Agreement Test")
    print("="*60)
    
    all_results = []
    
    # Test 1: Random data
    result = test_random_data(device)
    all_results.append(("Random Data", result))
    
    # Test 2: Edge values
    print(f"\n=== Edge Values Test ===")
    edge_results = test_edge_values(device)
    all_results.extend(edge_results)
    
    # Test 3: Different sizes
    print(f"\n=== Different Sizes Test ===")
    size_results = test_different_sizes(device)
    all_results.extend(size_results)
    
    # Summary
    print_summary(all_results)
