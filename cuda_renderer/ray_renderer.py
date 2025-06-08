import torch
from torch.utils.cpp_extension import load
import os
import sys

# Import Windows DLL helper for resolving loading issues
try:
    from .windows_dll_helper import setup_windows_cuda_environment, check_cuda_dependencies
    
    # Setup Windows environment before loading CUDA extension
    if sys.platform == "win32":
        print("Setting up Windows CUDA environment...")
        setup_windows_cuda_environment()
        check_cuda_dependencies()
except ImportError:
    print("Windows DLL helper not available")

def load_fused_ray_renderer():
    """Load CUDA extension with proper error handling and Windows support"""
    
    # Set Windows-specific environment variable
    if sys.platform == "win32":
        os.environ["DISTUTILS_USE_SDK"] = "1"
    
    # Get the directory of this file
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
    
    # Define CUDA architectures
    cuda_arches = [
        '-gencode=arch=compute_70,code=sm_70',  # V100, Titan V
        '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx series, T4
        '-gencode=arch=compute_80,code=sm_80',  # A100, RTX 30xx series
        '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx series
    ]
    
    # Add newer architectures if available
    try:
        cuda_version = torch.version.cuda
        if cuda_version and float(cuda_version) >= 11.8:
            cuda_arches.append('-gencode=arch=compute_89,code=sm_89')  # RTX 40xx series
    except:
        pass
    
    # Platform-specific compile flags
    extra_cflags = ['-O3']
    extra_cuda_cflags = [
        '-O3', 
        '--use_fast_math', 
        '--expt-relaxed-constexpr'
    ] + cuda_arches
    
    if sys.platform == "win32":
        extra_cflags.append('/std:c++17')
    else:
        extra_cflags.append('-std=c++17')
    
    print("Loading CUDA ray renderer extension...")
    
    try:
        # Load the CUDA extension
        return load(
            name="fused_ray_renderer",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=True,
            with_cuda=True
        )
    except Exception as e:
        print(f"Failed to load CUDA extension: {e}")
        raise

# Try to load the extension
try:
    fused_ray_renderer = load_fused_ray_renderer()
    print("CUDA ray renderer loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load CUDA ray renderer: {e}")
    fused_ray_renderer = None
