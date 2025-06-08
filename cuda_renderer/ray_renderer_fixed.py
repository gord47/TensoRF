"""
PyTorch JIT CUDA Extension Loader for Ray Renderer - Fixed Version
Compiles the CUDA extension on-demand with environment fixes and proper error handling
"""

import os
import sys
import torch
import warnings
from pathlib import Path

# Suppress compilation warnings
warnings.filterwarnings("ignore", category=UserWarning)

def apply_environment_fixes():
    """Apply CUDA environment fixes for compilation"""
    try:
        # Import and apply fixes
        from .fix_cuda_environment import CUDAEnvironmentFixer
        fixer = CUDAEnvironmentFixer()
        
        # Apply essential fixes silently
        fixer.create_dll_symlinks()
        fixer.setup_environment_variables()
        
        print("✓ Applied CUDA environment fixes")
        return True
    except Exception as e:
        print(f"⚠ Could not apply environment fixes: {e}")
        return False

def load_cuda_extension():
    """
    Load the CUDA extension using PyTorch's JIT compilation.
    Returns the compiled extension or None if compilation fails.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, cannot load CUDA extension")
        return None
    
    # Apply environment fixes first
    apply_environment_fixes()
    
    try:
        from torch.utils.cpp_extension import load
        
        # Get current directory
        current_dir = Path(__file__).parent
        
        # Source files
        cpp_file = current_dir / 'ray_renderer.cpp'
        cuda_file = current_dir / 'ray_renderer_kernel.cu'
        
        # Verify files exist
        if not cpp_file.exists():
            print(f"Error: C++ source file not found: {cpp_file}")
            return None
        if not cuda_file.exists():
            print(f"Error: CUDA source file not found: {cuda_file}")
            return None
        
        print("Compiling CUDA extension (this may take a while on first run)...")
        print(f"C++ source: {cpp_file}")
        print(f"CUDA source: {cuda_file}")
        
        # Setup environment for Windows
        if sys.platform == "win32":
            # Force Visual Studio build tools
            os.environ['DISTUTILS_USE_SDK'] = '1'
            os.environ['MSSdk'] = '1'
            
            # Add ninja to PATH if available
            ninja_paths = [
                r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja',
                r'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja',
                r'C:\tools\ninja'
            ]
            
            for ninja_path in ninja_paths:
                if os.path.exists(ninja_path):
                    current_path = os.environ.get('PATH', '')
                    if ninja_path not in current_path:
                        os.environ['PATH'] = ninja_path + ';' + current_path
                        print(f"Added Ninja to PATH: {ninja_path}")
                    break
        
        # Set up build directory to avoid conflicts
        build_dir = current_dir / 'jit_build_fixed'
        build_dir.mkdir(exist_ok=True)
        
        # Clean any existing lock files
        lock_file = build_dir / 'lock'
        if lock_file.exists():
            try:
                lock_file.unlink()
                print("✓ Cleaned existing lock file")
            except:
                pass
        
        # Compilation flags
        extra_cflags = ['-O3']
        extra_cuda_cflags = [
            '-O3',
            '--use_fast_math',
            '-Xptxas=-v',
            '--generate-line-info'
        ]
        
        # Add architecture-specific flags for A10G (compute capability 8.6)
        extra_cuda_cflags.extend([
            '-gencode', 'arch=compute_86,code=sm_86'
        ])
        
        if sys.platform == "win32":
            extra_cflags.extend(['/std:c++14', '/MD'])
            extra_cuda_cflags.extend(['-std=c++14', '-Xcompiler', '/MD'])
        else:
            extra_cflags.extend(['-std=c++14'])
            extra_cuda_cflags.extend(['-std=c++14'])
        
        # Load the extension with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                extension = load(
                    name=f'ray_renderer_cuda_fixed_v{attempt}',  # Unique name for each attempt
                    sources=[str(cpp_file), str(cuda_file)],
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    verbose=True,
                    build_directory=str(build_dir),
                    with_cuda=True
                )
                
                print("✓ CUDA extension compiled successfully!")
                return extension
                
            except Exception as e:
                print(f"Compilation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # Clean build directory and try again
                    import shutil
                    if build_dir.exists():
                        try:
                            shutil.rmtree(build_dir)
                            print("✓ Cleaned build directory")
                        except PermissionError:
                            print("⚠ Permission denied cleaning build directory - continuing anyway")
                        except Exception as clean_e:
                            print(f"⚠ Could not clean build directory: {clean_e}")
                        try:
                            build_dir.mkdir(exist_ok=True)
                        except Exception as mkdir_e:
                            print(f"⚠ Could not recreate build directory: {mkdir_e}")
                    print("Retrying compilation...")
                else:
                    raise e
        
    except Exception as e:
        print(f"Failed to compile CUDA extension: {e}")
        import traceback
        traceback.print_exc()
        return None

# Global extension instance with proper singleton pattern
_cuda_extension = None
_initialization_attempted = False

def get_cuda_extension():
    """Get the compiled CUDA extension, compiling if necessary"""
    global _cuda_extension, _initialization_attempted
    
    if _initialization_attempted:
        # Return cached result (could be None if failed)
        return _cuda_extension
    
    # First time - attempt to load
    _initialization_attempted = True
    try:
        _cuda_extension = load_cuda_extension()
        if _cuda_extension is not None:
            print("✓ CUDA extension loaded and cached successfully!")
        return _cuda_extension
    except Exception as e:
        print(f"Error during CUDA extension loading: {e}")
        _cuda_extension = None
        return None

# Compatibility with existing code
fused_ray_renderer = None

def initialize():
    """Initialize the CUDA extension"""
    global fused_ray_renderer
    
    # If already successfully initialized, return True
    if fused_ray_renderer is not None:
        return True
    
    try:
        extension = get_cuda_extension()
        if extension is not None:
            # Check if the extension has the forward method
            if hasattr(extension, 'forward'):
                fused_ray_renderer = extension
                print("✓ CUDA ray renderer initialized successfully!")
                return True
            else:
                print("Error: CUDA extension missing 'forward' method")
                return False
        else:
            print("Error: CUDA extension compilation failed")
            return False
    except Exception as e:
        print(f"Failed to initialize CUDA ray renderer: {e}")
        return False
