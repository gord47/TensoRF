"""
CUDA Environment Fix Tool
Resolves CUDA version mismatches and DLL issues for PyTorch CUDA extensions
"""

import os
import sys
import shutil
import subprocess
import torch
from pathlib import Path
import ctypes
from ctypes import wintypes
import winreg

class CUDAEnvironmentFixer:
    def __init__(self):
        self.cuda_path = os.environ.get('CUDA_PATH', '')
        self.pytorch_cuda_version = torch.version.cuda
        self.system_cuda_version = self.get_system_cuda_version()
        
    def get_system_cuda_version(self):
        """Get system CUDA version from nvcc"""
        try:
            if self.cuda_path:
                nvcc_path = os.path.join(self.cuda_path, 'bin', 'nvcc.exe')
                if os.path.exists(nvcc_path):
                    result = subprocess.run([nvcc_path, '--version'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        # Parse version from output like "release 12.4, V12.4.99"
                        for line in result.stdout.split('\n'):
                            if 'release' in line:
                                version = line.split('release ')[1].split(',')[0]
                                return version.strip()
        except Exception as e:
            print(f"Failed to get system CUDA version: {e}")
        return None
    
    def find_cuda_dlls(self):
        """Find CUDA DLLs in system"""
        dll_locations = {}
        
        # Common CUDA DLL locations
        search_paths = [
            os.path.join(self.cuda_path, 'bin') if self.cuda_path else None,
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin',
            r'C:\Windows\System32',
            os.path.dirname(torch.__file__) + r'\lib',
        ]
        
        # Add PyTorch DLL path
        torch_dll_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_dll_path):
            search_paths.append(torch_dll_path)
        
        # CUDA DLLs we need
        cuda_dlls = [
            'cudart64_11.dll', 'cudart64_12.dll',
            'cublas64_11.dll', 'cublas64_12.dll',
            'curand64_10.dll', 'cusparse64_11.dll',
            'cufft64_10.dll', 'cusolver64_11.dll'
        ]
        
        for path in search_paths:
            if path and os.path.exists(path):
                for dll in cuda_dlls:
                    dll_path = os.path.join(path, dll)
                    if os.path.exists(dll_path):
                        if dll not in dll_locations:
                            dll_locations[dll] = []
                        dll_locations[dll].append(dll_path)
        
        return dll_locations
    
    def create_dll_symlinks(self):
        """Create symlinks/copies for missing CUDA DLLs"""
        print("=== Creating CUDA DLL Compatibility Links ===")
        
        dll_locations = self.find_cuda_dlls()
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        
        # Ensure torch lib directory exists
        os.makedirs(torch_lib_path, exist_ok=True)
        
        # Create compatibility mappings
        dll_mappings = {
            'cudart64_11.dll': ['cudart64_12.dll'],  # Map CUDA 11 runtime to CUDA 12
            'cublas64_11.dll': ['cublas64_12.dll'],   # Map CUDA 11 cublas to CUDA 12
        }
        
        for target_dll, source_candidates in dll_mappings.items():
            target_path = os.path.join(torch_lib_path, target_dll)
            
            # Skip if target already exists
            if os.path.exists(target_path):
                print(f"✓ {target_dll} already exists")
                continue
            
            # Find a source DLL to copy from
            source_path = None
            for candidate in source_candidates:
                if candidate in dll_locations and dll_locations[candidate]:
                    source_path = dll_locations[candidate][0]
                    break
            
            if source_path:
                try:
                    # Try to create symlink first, fallback to copy
                    try:
                        os.symlink(source_path, target_path)
                        print(f"✓ Created symlink: {target_dll} -> {source_path}")
                    except OSError:
                        # Symlink failed, try copy
                        shutil.copy2(source_path, target_path)
                        print(f"✓ Copied: {target_dll} from {source_path}")
                except Exception as e:
                    print(f"✗ Failed to create {target_dll}: {e}")
            else:
                print(f"✗ No source found for {target_dll}")
    
    def setup_environment_variables(self):
        """Setup environment variables for CUDA compilation"""
        print("=== Setting up Environment Variables ===")
        
        # Add PyTorch lib path to PATH
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        current_path = os.environ.get('PATH', '')
        
        if torch_lib_path not in current_path:
            os.environ['PATH'] = torch_lib_path + ';' + current_path
            print(f"✓ Added to PATH: {torch_lib_path}")
        
        # Set CUDA compilation flags for compatibility
        cuda_flags = [
            '-std=c++14',  # Ensure C++14 compatibility
            '-Xcompiler', '/MD',  # Use dynamic runtime
        ]
        
        # Force CUDA architecture for A10G (compute capability 8.6)
        cuda_flags.extend(['-gencode', 'arch=compute_86,code=sm_86'])
        
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
        os.environ['FORCE_CUDA'] = '1'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging
        
        # Set build directory to avoid permission issues
        build_dir = os.path.join(os.getcwd(), 'cuda_build_temp')
        os.makedirs(build_dir, exist_ok=True)
        os.environ['TORCH_EXTENSIONS_DIR'] = build_dir
        
        print(f"✓ Set TORCH_EXTENSIONS_DIR: {build_dir}")
        print(f"✓ Set TORCH_CUDA_ARCH_LIST: 8.6")
    
    def clean_build_cache(self):
        """Clean PyTorch extension build cache"""
        print("=== Cleaning Build Cache ===")
        
        # Default PyTorch extensions directory
        default_cache = os.path.expanduser('~/.cache/torch_extensions')
        if os.path.exists(default_cache):
            try:
                shutil.rmtree(default_cache)
                print(f"✓ Cleaned default cache: {default_cache}")
            except Exception as e:
                print(f"⚠ Could not clean default cache: {e}")
        
        # Custom build directories
        custom_dirs = [
            'cuda_build_temp',
            'jit_build',
            'build',
            os.path.join('cuda_renderer', 'build')
        ]
        
        for dir_name in custom_dirs:
            if os.path.exists(dir_name):
                try:
                    shutil.rmtree(dir_name)
                    print(f"✓ Cleaned: {dir_name}")
                except Exception as e:
                    print(f"⚠ Could not clean {dir_name}: {e}")
    
    def test_cuda_extension_build(self):
        """Test building a simple CUDA extension after fixes"""
        print("=== Testing CUDA Extension Build ===")
        
        try:
            from torch.utils.cpp_extension import load_inline
            
            # Simple CUDA kernel code
            cuda_source = '''
            __global__ void test_kernel(float* data, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    data[idx] = data[idx] * 2.0f;
                }
            }
            
            torch::Tensor test_cuda_forward(torch::Tensor input) {
                auto output = input.clone();
                int n = input.numel();
                
                dim3 block(256);
                dim3 grid((n + block.x - 1) / block.x);
                
                test_kernel<<<grid, block>>>(output.data_ptr<float>(), n);
                cudaDeviceSynchronize();
                
                return output;
            }
            '''
            
            cpp_source = '''
            torch::Tensor test_cuda_forward(torch::Tensor input);
            '''
            
            # Load the extension
            extension = load_inline(
                name='test_cuda_fixed',
                cpp_sources=[cpp_source],
                cuda_sources=[cuda_source],
                functions=['test_cuda_forward'],
                verbose=True,
                with_cuda=True,
                build_directory='cuda_test_build'
            )
            
            # Test it
            if torch.cuda.is_available():
                test_tensor = torch.ones(100, dtype=torch.float32, device='cuda')
                result = extension.test_cuda_forward(test_tensor)
                
                if torch.allclose(result, test_tensor * 2):
                    print("✓ CUDA extension test PASSED!")
                    return True
                else:
                    print("✗ CUDA extension test FAILED - incorrect results")
            else:
                print("✗ CUDA not available for testing")
            
        except Exception as e:
            print(f"✗ CUDA extension test FAILED: {e}")
            return False
        
        return False
    
    def apply_all_fixes(self):
        """Apply all fixes"""
        print("CUDA Environment Fixer")
        print("=" * 50)
        print(f"PyTorch CUDA Version: {self.pytorch_cuda_version}")
        print(f"System CUDA Version: {self.system_cuda_version}")
        print()
        
        self.clean_build_cache()
        self.create_dll_symlinks()
        self.setup_environment_variables()
        
        # Test the fixes
        if self.test_cuda_extension_build():
            print("\n🎉 All fixes applied successfully!")
            print("You can now try building your CUDA extension again.")
        else:
            print("\n⚠ Some issues may remain. Check the error messages above.")
        
        return True

def main():
    """Main function"""
    if sys.platform != "win32":
        print("This tool is designed for Windows systems only.")
        return
    
    fixer = CUDAEnvironmentFixer()
    fixer.apply_all_fixes()

if __name__ == "__main__":
    main()
