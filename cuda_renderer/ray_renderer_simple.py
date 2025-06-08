"""
Alternative CUDA Ray Renderer Loader with Enhanced Windows Support
This version focuses on resolving Windows DLL loading issues.
"""

import torch
import os
import sys
import warnings
from pathlib import Path

def setup_cuda_environment():
    """Setup CUDA environment for PyTorch extension loading"""
    
    if sys.platform == "win32":
        # Set environment variables for Windows
        os.environ["DISTUTILS_USE_SDK"] = "1"
        os.environ["MSSdk"] = "1"
         
        # Add CUDA paths to environment
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            bin_path = os.path.join(cuda_path, 'bin')
            lib_path = os.path.join(cuda_path, 'lib', 'x64')
            
            # Add to PATH if not already there
            current_path = os.environ.get('PATH', '')
            if bin_path not in current_path:
                os.environ['PATH'] = bin_path + os.pathsep + current_path
            if lib_path not in current_path:
                os.environ['PATH'] = lib_path + os.pathsep + os.environ['PATH']
        
        # Try to add DLL directories (Python 3.8+)
        if hasattr(os, 'add_dll_directory') and cuda_path:
            try:
                os.add_dll_directory(os.path.join(cuda_path, 'bin'))
                print(f"Added CUDA bin to DLL search path: {os.path.join(cuda_path, 'bin')}")
            except:
                pass


def load_cuda_extension_jit():
    """Load CUDA extension using JIT compilation with fallback options"""
    
    setup_cuda_environment()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define source files
    sources = [
        os.path.join(current_dir, "ray_renderer.cpp"),
        os.path.join(current_dir, "ray_renderer_kernel.cu"),
    ]
    
    # Check if source files exist
    for src in sources:
        if not os.path.exists(src):
            raise FileNotFoundError(f"CUDA source file not found: {src}")
    
    # Compilation flags
    if sys.platform == "win32":
        extra_cflags = ['/O2', '/std:c++17']
        extra_cuda_cflags = ['-O3', '--use_fast_math', '--expt-relaxed-constexpr']
    else:
        extra_cflags = ['-O3', '-std=c++17']
        extra_cuda_cflags = ['-O3', '--use_fast_math', '--expt-relaxed-constexpr']
    
    # Add CUDA architectures
    cuda_arches = [
        '-gencode=arch=compute_70,code=sm_70',
        '-gencode=arch=compute_75,code=sm_75', 
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_86,code=sm_86',
    ]
    extra_cuda_cflags.extend(cuda_arches)
    
    print("Compiling CUDA extension...")
    
    try:
        from torch.utils.cpp_extension import load
        return load(
            name="ray_renderer_cuda_simple",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=True,
            with_cuda=True,
            build_directory=os.path.join(current_dir, "build_simple")
        )
    except Exception as e:
        print(f"JIT compilation failed: {e}")
        return None


def try_precompiled_extension():
    """Try to load a precompiled extension if available"""
    try:
        import ray_renderer_cuda_precompiled
        print("Loaded precompiled CUDA extension")
        return ray_renderer_cuda_precompiled
    except ImportError:
        return None


def get_cuda_extension():
    """Get CUDA extension with multiple fallback strategies"""
    
    # Strategy 1: Try precompiled extension
    ext = try_precompiled_extension()
    if ext is not None:
        return ext
    
    # Strategy 2: Try JIT compilation
    ext = load_cuda_extension_jit()
    if ext is not None:
        return ext
    
    # Strategy 3: Return None (will fallback to PyTorch)
    warnings.warn("Could not load CUDA extension, falling back to PyTorch implementation")
    return None


# Try to load the extension
cuda_extension = None
cuda_available = False

try:
    cuda_extension = get_cuda_extension()
    if cuda_extension is not None:
        cuda_available = True
        print("CUDA ray renderer loaded successfully!")
    else:
        print("CUDA ray renderer not available")
except Exception as e:
    print(f"Failed to load CUDA ray renderer: {e}")
    cuda_extension = None
    cuda_available = False
