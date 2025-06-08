"""
Test Script for CUDA Environment Fixes
This script validates that the CUDA environment fixes work properly
"""

import sys
import os
import torch

def test_cuda_fixes():
    """Test the CUDA environment fixes"""
    print("CUDA Environment Fix Test")
    print("=" * 50)
    
    # Basic CUDA availability check
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.get_device_name()}")
    
    print()
    
    # Apply environment fixes
    print("Applying CUDA environment fixes...")
    try:
        from cuda_renderer.fix_cuda_environment import CUDAEnvironmentFixer
        
        fixer = CUDAEnvironmentFixer()
        print(f"PyTorch CUDA version: {fixer.pytorch_cuda_version}")
        print(f"System CUDA version: {fixer.system_cuda_version}")
        
        # Apply fixes
        success = fixer.apply_all_fixes()
        
        if success:
            print("✓ All fixes applied successfully!")
        else:
            print("✗ Some fixes failed")
        
    except Exception as e:
        print(f"✗ Failed to apply fixes: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test CUDA extension loading
    print("Testing CUDA extension loading...")
    try:
        from cuda_renderer.ray_renderer_fixed import get_cuda_extension
        
        extension = get_cuda_extension()
        
        if extension is not None:
            print("✓ CUDA extension loaded successfully!")
            
            # Test the extension with a simple tensor
            if torch.cuda.is_available():
                test_tensor = torch.ones(10, device='cuda', dtype=torch.float32)
                print(f"✓ Created test CUDA tensor: {test_tensor.shape}")
                
                # Check if the extension has our expected functions
                if hasattr(extension, 'render_rays_forward'):
                    print("✓ Extension has render_rays_forward function")
                else:
                    print("⚠ Extension missing render_rays_forward function")
                
                return True
            else:
                print("⚠ CUDA not available for testing")
                return False
        else:
            print("✗ CUDA extension failed to load")
            return False
            
    except Exception as e:
        print(f"✗ Failed to load CUDA extension: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cuda_fixes()
    
    if success:
        print("\n🎉 All tests passed! Your CUDA environment is ready.")
        print("You can now use the CUDA ray renderer.")
    else:
        print("\n⚠ Some tests failed. Check the error messages above.")
        print("You may need to install CUDA runtime libraries or fix DLL issues.")
    
    # Give user instructions
    print("\nNext steps:")
    print("1. If the test passed, you can now run your TensorRF training")
    print("2. If it failed, try running as administrator or check CUDA installation")
    print("3. Make sure your PyTorch version matches your CUDA version")
