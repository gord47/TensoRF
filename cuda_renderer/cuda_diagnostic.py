"""
CUDA Extension Diagnostic Tool
Helps diagnose and fix CUDA extension loading issues on Windows
"""

import os
import sys
import subprocess
import torch
import platform
from pathlib import Path

def print_system_info():
    """Print system and environment information"""
    print("=== System Information ===")
    print(f"OS: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    print()

def check_cuda_installation():
    """Check CUDA installation"""
    print("=== CUDA Installation ===")
    
    # Check CUDA_PATH environment variable
    cuda_path = os.environ.get('CUDA_PATH')
    print(f"CUDA_PATH: {cuda_path}")
    
    if cuda_path and os.path.exists(cuda_path):
        bin_path = os.path.join(cuda_path, 'bin')
        lib_path = os.path.join(cuda_path, 'lib', 'x64')
        print(f"CUDA bin exists: {os.path.exists(bin_path)}")
        print(f"CUDA lib exists: {os.path.exists(lib_path)}")
        
        # Check for nvcc
        nvcc_path = os.path.join(bin_path, 'nvcc.exe')
        print(f"nvcc.exe exists: {os.path.exists(nvcc_path)}")
        
        if os.path.exists(nvcc_path):
            try:
                result = subprocess.run([nvcc_path, '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"nvcc version: {result.stdout.strip()}")
                else:
                    print(f"nvcc error: {result.stderr}")
            except Exception as e:
                print(f"Failed to run nvcc: {e}")
    
    # Check for common CUDA DLLs
    cuda_dlls = [
        'cudart64_11.dll', 'cudart64_12.dll',
        'cublas64_11.dll', 'cublas64_12.dll', 
        'curand64_10.dll', 'cusparse64_11.dll'
    ]
    
    print("\nCUDA DLL Check:")
    for dll in cuda_dlls:
        try:
            import ctypes
            ctypes.CDLL(dll)
            print(f"  {dll}: Found")
        except OSError:
            print(f"  {dll}: Not found")
    print()

def check_visual_studio():
    """Check Visual Studio installation"""
    print("=== Visual Studio Check ===")
    
    # Check for cl.exe (MSVC compiler)
    try:
        result = subprocess.run(['cl'], capture_output=True, text=True, timeout=5)
        print(f"cl.exe available: {result.returncode == 0}")
        if result.stderr:
            # cl.exe typically outputs version info to stderr
            lines = result.stderr.strip().split('\n')
            if lines:
                print(f"MSVC version: {lines[0]}")
    except Exception as e:
        print(f"cl.exe not found: {e}")
    
    # Check for MSBuild
    try:
        result = subprocess.run(['msbuild', '/version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"MSBuild available: True")
        else:
            print(f"MSBuild available: False")
    except Exception as e:
        print(f"MSBuild check failed: {e}")
    
    # Check environment variables
    vs_vars = ['VCINSTALLDIR', 'VSINSTALLDIR', 'WindowsSdkDir']
    for var in vs_vars:
        value = os.environ.get(var)
        print(f"{var}: {value if value else 'Not set'}")
    print()

def check_ninja():
    """Check Ninja build system"""
    print("=== Ninja Build System ===")
    try:
        result = subprocess.run(['ninja', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"Ninja version: {result.stdout.strip()}")
        else:
            print("Ninja not available")
    except Exception as e:
        print(f"Ninja check failed: {e}")
    print()

def test_simple_cuda_extension():
    """Test compiling a simple CUDA extension"""
    print("=== Simple CUDA Extension Test ===")
    
    # Create a minimal CUDA extension test
    current_dir = Path(__file__).parent
    test_dir = current_dir / "test_cuda"
    test_dir.mkdir(exist_ok=True)
    
    # Simple CUDA kernel
    cuda_code = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

torch::Tensor test_cuda(torch::Tensor input) {
    auto output = input.clone();
    int n = input.numel();
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    test_kernel<<<grid, block>>>(output.data_ptr<float>(), n);
    cudaDeviceSynchronize();
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_cuda", &test_cuda, "Test CUDA function");
}
'''
    
    # C++ interface
    cpp_code = '''
#include <torch/extension.h>

torch::Tensor test_cuda(torch::Tensor input);

torch::Tensor test_forward(torch::Tensor input) {
    return test_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_forward", &test_forward, "Test forward");
}
'''
    
    # Write test files
    (test_dir / "test.cu").write_text(cuda_code)
    (test_dir / "test.cpp").write_text(cpp_code)
    
    try:
        from torch.utils.cpp_extension import load
        
        print("Attempting to compile simple CUDA extension...")
        extension = load(
            name="test_cuda_ext",
            sources=[str(test_dir / "test.cpp"), str(test_dir / "test.cu")],
            verbose=True,
            with_cuda=True,
            build_directory=str(test_dir / "build")
        )
        
        # Test the extension
        test_tensor = torch.ones(10, dtype=torch.float32, device='cuda')
        result = extension.test_forward(test_tensor)
        
        if torch.allclose(result, test_tensor * 2):
            print("✓ Simple CUDA extension test PASSED")
        else:
            print("✗ Simple CUDA extension test FAILED - incorrect results")
            
    except Exception as e:
        print(f"✗ Simple CUDA extension test FAILED: {e}")
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(test_dir)
    except:
        pass
    
    print()

def main():
    """Run all diagnostic checks"""
    print("CUDA Extension Diagnostic Tool")
    print("=" * 50)
    
    print_system_info()
    check_cuda_installation()
    
    if sys.platform == "win32":
        check_visual_studio()
    
    check_ninja()
    
    if torch.cuda.is_available():
        test_simple_cuda_extension()
    else:
        print("Skipping CUDA extension test - CUDA not available")
    
    print("Diagnostic complete!")

if __name__ == "__main__":
    main()
