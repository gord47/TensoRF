#!/usr/bin/env python3
"""
Quick test to verify the density_shift fix works
"""

import torch
from models.tensoRF import TensorVMSplit

print('🔧 VERIFYING DENSITY_SHIFT FIX')
print('=' * 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Test parameters
aabb = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]], device=device)
reso_cur = [128, 128, 128]

# Create model with the fixed density_shift = -2
tensorf = TensorVMSplit(
    aabb, reso_cur, device,
    density_n_comp=[16, 16, 16], 
    appearance_n_comp=[48, 48, 48], 
    app_dim=27, 
    shadingMode='MLP_PE', 
    alphaMask_thres=1e-4,  # Updated threshold from config
    density_shift=-2,      # FIXED VALUE
    distance_scale=25,
    pos_pe=6, view_pe=6, fea_pe=6, 
    featureC=128, step_ratio=2.0, 
    fea2denseAct='softplus'
)

print(f'Model created with density_shift = {tensorf.density_shift}')

# Test updateAlphaMask with small grid
print('\n🧪 Testing updateAlphaMask...')
try:
    result_aabb = tensorf.updateAlphaMask(gridSize=(16, 16, 16))
    print('✅ updateAlphaMask completed successfully!')
    print(f'Result AABB shape: {result_aabb.shape}')
    print(f'Result AABB: {result_aabb}')
    
    # Quick alpha test
    alpha, dense_xyz = tensorf.getDenseAlpha((16, 16, 16))
    above_threshold = (alpha > tensorf.alphaMask_thres).sum().item()
    total_voxels = alpha.numel()
    percentage = above_threshold / total_voxels * 100
    
    print(f'\n📊 VERIFICATION RESULTS:')
    print(f'   Alpha range: {alpha.min().item():.6f} to {alpha.max().item():.6f}')
    print(f'   Above threshold: {above_threshold}/{total_voxels} voxels')
    print(f'   Percentage: {percentage:.2f}%')
    
    if above_threshold > 0:
        print('🎉 SUCCESS: Fix verified! Alpha values are now above threshold.')
        print('💡 Ready for training without crashes.')
    else:
        print('⚠️ WARNING: Still no voxels above threshold.')
        
except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 50)
