"""
Windows DLL Helper for CUDA Extensions
Helps resolve DLL loading issues on Windows by adding necessary paths and checking dependencies.
"""

import os
import sys
import ctypes
from pathlib import Path
import warnings


def find_cuda_dlls():
    """Find CUDA DLL paths on Windows"""
    cuda_paths = []
    
    # Check CUDA_PATH environment variable
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        cuda_paths.append(Path(cuda_path))
    
    # Check common CUDA installation paths
    common_paths = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
        Path(r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA"),
    ]
    
    for base_path in common_paths:
        if base_path.exists():
            # Find version directories
            for version_dir in base_path.iterdir():
                if version_dir.is_dir() and version_dir.name.startswith('v'):
                    cuda_paths.append(version_dir)
    
    # Find bin directories with DLLs
    dll_paths = []
    for cuda_path in cuda_paths:
        bin_path = cuda_path / "bin"
        if bin_path.exists():
            dll_paths.append(str(bin_path))
    
    return dll_paths


def find_visual_studio_dlls():
    """Find Visual Studio runtime DLL paths"""
    vs_paths = []
    
    # Check for Visual Studio Build Tools or Visual Studio
    program_files = [
        Path(r"C:\Program Files\Microsoft Visual Studio"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019"),
    ]
    
    for base_path in program_files:
        if base_path.exists():
            # Look for VC redistributables and tools
            for subdir in base_path.rglob("**/VC/Redist/**/x64/**"):
                if subdir.is_dir():
                    vs_paths.append(str(subdir))
            
            for subdir in base_path.rglob("**/VC/Tools/**/bin/Hostx64/x64"):
                if subdir.is_dir():
                    vs_paths.append(str(subdir))
    
    return vs_paths


def add_dll_directory(path):
    """Add DLL directory to Windows DLL search path"""
    if not os.path.exists(path):
        return False
    
    try:
        if hasattr(os, 'add_dll_directory'):
            # Python 3.8+
            os.add_dll_directory(path)
            return True
        else:
            # Fallback for older Python versions
            if path not in os.environ.get('PATH', ''):
                os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
            return True
    except Exception as e:
        warnings.warn(f"Failed to add DLL directory {path}: {e}")
        return False


def setup_windows_cuda_environment():
    """Setup Windows environment for CUDA extensions"""
    if sys.platform != "win32":
        return True
    
    print("Setting up Windows CUDA environment...")
    
    # Find and add CUDA DLL paths
    cuda_dll_paths = find_cuda_dlls()
    cuda_found = False
    
    for path in cuda_dll_paths:
        if add_dll_directory(path):
            print(f"Added CUDA DLL path: {path}")
            cuda_found = True
    
    if not cuda_found:
        print("Warning: No CUDA DLL paths found")
    
    # Find and add Visual Studio DLL paths
    vs_dll_paths = find_visual_studio_dlls()
    vs_found = False
    
    for path in vs_dll_paths:
        if add_dll_directory(path):
            print(f"Added VS DLL path: {path}")
            vs_found = True
    
    if not vs_found:
        print("Warning: No Visual Studio DLL paths found")
    
    # Set additional environment variables
    os.environ["DISTUTILS_USE_SDK"] = "1"
    
    # Try to load common CUDA libraries to verify they're available
    try:
        ctypes.CDLL("cudart64_11.dll")
        print("CUDA Runtime DLL found")
    except OSError:
        try:
            ctypes.CDLL("cudart64_12.dll")
            print("CUDA Runtime DLL (v12) found")
        except OSError:
            print("Warning: CUDA Runtime DLL not found")
    
    try:
        ctypes.CDLL("cublas64_11.dll")
        print("CUDA BLAS DLL found")
    except OSError:
        try:
            ctypes.CDLL("cublas64_12.dll")
            print("CUDA BLAS DLL (v12) found")
        except OSError:
            print("Warning: CUDA BLAS DLL not found")
    
    return cuda_found or vs_found


def check_cuda_dependencies():
    """Check if required CUDA dependencies are available"""
    if sys.platform != "win32":
        return True
    
    required_dlls = [
        "cudart64_11.dll",
        "cudart64_12.dll", 
        "cublas64_11.dll",
        "cublas64_12.dll",
        "curand64_10.dll",
        "cusparse64_11.dll",
        "cusparse64_12.dll"
    ]
    
    found_dlls = []
    for dll in required_dlls:
        try:
            ctypes.CDLL(dll)
            found_dlls.append(dll)
        except OSError:
            continue
    
    if found_dlls:
        print(f"Found CUDA DLLs: {found_dlls}")
        return True
    else:
        print("No CUDA DLLs found")
        return False


if __name__ == "__main__":
    # Test the helper functions
    print("Testing Windows CUDA environment setup...")
    setup_windows_cuda_environment()
    check_cuda_dependencies()
