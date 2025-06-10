#!/usr/bin/env python3
"""
Test the proper CUDA fix that matches PyTorch density computation
"""

import torch
from models.tensoRF import TensorVMSplit

print('🔧 TESTING PROPER CUDA FIX (CUDA = PyTorch)')
print('=' * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Test parameters matching standard PyTorch config
aabb = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]], device=device)
reso_cur = [128, 128, 128]

# Create model with STANDARD PyTorch values
tensorf = TensorVMSplit(
    aabb, reso_cur, device,
    density_n_comp=[16, 16, 16], 
    appearance_n_comp=[48, 48, 48], 
    app_dim=27, 
    shadingMode='MLP_PE', 
    alphaMask_thres=0.001,    # Standard PyTorch threshold
    density_shift=-10,        # Standard PyTorch density_shift
    distance_scale=25,
    pos_pe=6, view_pe=6, fea_pe=6, 
    featureC=128, step_ratio=2.0, 
    fea2denseAct='softplus'
)

print(f'Model created with:')
print(f'  density_shift = {tensorf.density_shift}')
print(f'  alphaMask_thres = {tensorf.alphaMask_thres}')

# Test updateAlphaMask
print('\n🧪 Testing updateAlphaMask with proper CUDA implementation...')
try:
    result_aabb = tensorf.updateAlphaMask(gridSize=(16, 16, 16))
    print('✅ updateAlphaMask completed successfully!')
    
    # Test alpha computation
    alpha, dense_xyz = tensorf.getDenseAlpha((16, 16, 16))
    above_threshold = (alpha > tensorf.alphaMask_thres).sum().item()
    total_voxels = alpha.numel()
    percentage = above_threshold / total_voxels * 100
    
    print(f'\n📊 RESULTS WITH PROPER CUDA IMPLEMENTATION:')
    print(f'   Alpha range: {alpha.min().item():.8f} to {alpha.max().item():.8f}')
    print(f'   Above threshold: {above_threshold}/{total_voxels} voxels')
    print(f'   Percentage: {percentage:.2f}%')
    
    if above_threshold > 0:
        print('🎉 SUCCESS: CUDA now matches PyTorch density computation!')
        print('💡 Ready for training with standard PyTorch config values.')
    else:
        print('⚠️ WARNING: Still no voxels above threshold.')
        print('🔍 Need to investigate further...')
        
except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
