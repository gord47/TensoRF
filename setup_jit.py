import os
import torch
from torch.utils.cpp_extension import load
import sys

def load_cuda_extension():
    """
    Load CUDA extension using JIT compilation
    This approach often works better on Windows than setup.py builds
    """
    
    # Set Windows-specific environment variables for ninja build system
    if sys.platform == "win32":
        os.environ["DISTUTILS_USE_SDK"] = "1"
        # Force ninja to use single-threaded compilation to avoid race conditions
        os.environ["MAX_JOBS"] = "1"
        # Set explicit ninja path
        ninja_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
        if os.path.exists(ninja_path):
            os.environ["NINJA"] = ninja_path
            os.environ["CMAKE_MAKE_PROGRAM"] = ninja_path
            # Also add to PATH to ensure it's found by all build tools
            ninja_dir = os.path.dirname(ninja_path)
            current_path = os.environ.get("PATH", "")
            if ninja_dir not in current_path:
                os.environ["PATH"] = ninja_dir + os.pathsep + current_path
            print(f"Using ninja from: {ninja_path}")
        else:
            print(f"Warning: Ninja not found at {ninja_path}")
        
        # Set build directory to avoid permission issues
        build_dir = os.path.join(os.getcwd(), "jit_build")
        os.makedirs(build_dir, exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = build_dir
        
        # Additional Windows build environment setup
        os.environ["CMAKE_GENERATOR"] = "Ninja"
        os.environ["USE_NINJA"] = "1"
    
    # Bypass CUDA version check for JIT compilation
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6"
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_dir = os.path.join(current_dir, 'cuda_renderer')
    
    # Source files
    sources = [
        os.path.join(cuda_dir, 'ray_renderer.cpp'),
        os.path.join(cuda_dir, 'ray_renderer_kernel.cu'),
    ]
    
    # Check if source files exist
    for src in sources:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}")
    
    # Simplified CUDA architectures to reduce compilation complexity
    cuda_arches = [
        '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx series, T4
        '-gencode=arch=compute_80,code=sm_80',  # A100, RTX 30xx series
        '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx series
    ]
    
    # Compile arguments - simplified for better compatibility
    extra_cflags = ['-O2']  # Use O2 instead of O3 for better stability
    extra_cuda_cflags = [
        '-O2',
        '--use_fast_math',
        '--expt-relaxed-constexpr',
        '-Xcompiler',
        '/wd4819'  # Suppress MSVC warning about character encoding
    ] + cuda_arches
    
    # Add platform-specific flags
    if sys.platform == "win32":
        extra_cflags.extend(['/std:c++14', '/bigobj'])  # Use C++14 for better compatibility
        extra_cuda_cflags.extend(['-std=c++14'])
    else:
        extra_cflags.append('-std=c++14')
        extra_cuda_cflags.append('-std=c++14')
    
    print("Compiling CUDA extension with JIT...")
    print(f"Sources: {sources}")
    print(f"Build directory: {os.environ.get('TORCH_EXTENSIONS_DIR', 'default')}")
    
    try:
        # Load extension with JIT compilation
        ray_renderer_cuda = load(
            name='ray_renderer_cuda_jit',
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=True,
            with_cuda=True,
            build_directory=os.environ.get('TORCH_EXTENSIONS_DIR', None)
        )
        
        print("CUDA extension compiled successfully!")
        return ray_renderer_cuda
        
    except Exception as e:
        print(f"Failed to compile CUDA extension: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure Visual Studio Build Tools are installed")
        print("2. Check that CUDA toolkit matches your GPU architecture")
        print("3. Try running from Developer Command Prompt")
        raise

if __name__ == "__main__":
    # Test compilation
    try:
        module = load_cuda_extension()
        print("✓ CUDA extension compilation test successful!")
        print(f"Available functions: {dir(module)}")
    except Exception as e:
        print(f"✗ CUDA extension compilation test failed: {e}")
        sys.exit(1)
