"""
Windows CUDA DLL Fix
Helps resolve CUDA runtime DLL loading issues on Windows
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import ctypes
from ctypes import wintypes

def find_cuda_dlls():
    """Find all available CUDA runtime DLLs"""
    print("=== Finding CUDA DLLs ===")
    
    # Common CUDA installation paths
    cuda_paths = [
        os.environ.get('CUDA_PATH'),
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA',
        r'C:\ProgramData\NVIDIA Corporation\CUDA Samples',
    ]
    
    # Search for DLLs
    found_dlls = {}
    
    for cuda_path in cuda_paths:
        if not cuda_path or not os.path.exists(cuda_path):
            continue
            
        print(f"Searching in: {cuda_path}")
        
        # Search recursively for CUDA DLLs
        for root, dirs, files in os.walk(cuda_path):
            for file in files:
                if file.startswith('cudart64_') and file.endswith('.dll'):
                    full_path = os.path.join(root, file)
                    found_dlls[file] = full_path
                    print(f"  Found: {file} -> {full_path}")
    
    # Also check system PATH
    print("\nSearching in system PATH...")
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    for path_dir in path_dirs:
        if os.path.exists(path_dir):
            try:
                files = os.listdir(path_dir)
                for file in files:
                    if file.startswith('cudart64_') and file.endswith('.dll'):
                        full_path = os.path.join(path_dir, file)
                        found_dlls[file] = full_path
                        print(f"  Found in PATH: {file} -> {full_path}")
            except (PermissionError, OSError):
                continue
    
    return found_dlls

def create_dll_symlinks():
    """Create symlinks to help with DLL compatibility"""
    print("\n=== Creating DLL Compatibility Links ===")
    
    found_dlls = find_cuda_dlls()
    
    if not found_dlls:
        print("No CUDA runtime DLLs found!")
        return False
    
    # Find the highest version DLL
    dll_versions = {}
    for dll_name in found_dlls:
        if 'cudart64_' in dll_name:
            try:
                version = dll_name.replace('cudart64_', '').replace('.dll', '')
                dll_versions[version] = found_dlls[dll_name]
            except:
                continue
    
    if not dll_versions:
        print("No valid CUDA runtime DLLs found!")
        return False
    
    # Get the latest version
    latest_version = max(dll_versions.keys(), key=lambda x: tuple(map(int, x.split('.'))))
    latest_dll = dll_versions[latest_version]
    
    print(f"Latest CUDA runtime DLL: cudart64_{latest_version}.dll")
    print(f"Path: {latest_dll}")
    
    # Create symlinks for missing versions
    needed_dlls = ['cudart64_11.dll', 'cudart64_12.dll']
    cuda_bin = os.path.dirname(latest_dll)
    
    for needed_dll in needed_dlls:
        if needed_dll not in found_dlls:
            target_path = os.path.join(cuda_bin, needed_dll)
            print(f"Creating link: {needed_dll} -> {os.path.basename(latest_dll)}")
            
            try:
                # Try to create a hard link first (doesn't require admin rights)
                os.link(latest_dll, target_path)
                print(f"  ✓ Hard link created: {target_path}")
            except OSError:
                try:
                    # Fall back to copy
                    shutil.copy2(latest_dll, target_path)
                    print(f"  ✓ Copy created: {target_path}")
                except Exception as e:
                    print(f"  ✗ Failed to create {needed_dll}: {e}")
    
    return True

def add_cuda_to_path():
    """Add CUDA bin directory to PATH if not already there"""
    print("\n=== Updating PATH ===")
    
    cuda_path = os.environ.get('CUDA_PATH')
    if not cuda_path:
        print("CUDA_PATH not set")
        return False
    
    cuda_bin = os.path.join(cuda_path, 'bin')
    if not os.path.exists(cuda_bin):
        print(f"CUDA bin directory not found: {cuda_bin}")
        return False
    
    current_path = os.environ.get('PATH', '')
    if cuda_bin not in current_path:
        print(f"Adding {cuda_bin} to PATH")
        os.environ['PATH'] = cuda_bin + os.pathsep + current_path
    else:
        print("CUDA bin already in PATH")
    
    return True

def test_dll_loading():
    """Test if CUDA DLLs can be loaded"""
    print("\n=== Testing DLL Loading ===")
    
    dlls_to_test = [
        'cudart64_11.dll',
        'cudart64_12.dll',
        'cublas64_11.dll',
        'curand64_10.dll'
    ]
    
    success_count = 0
    for dll in dlls_to_test:
        try:
            ctypes.CDLL(dll)
            print(f"  ✓ {dll}: Loaded successfully")
            success_count += 1
        except OSError as e:
            print(f"  ✗ {dll}: Failed to load - {e}")
    
    print(f"\nLoaded {success_count}/{len(dlls_to_test)} DLLs successfully")
    return success_count == len(dlls_to_test)

def fix_jit_compilation():
    """Fix JIT compilation environment"""
    print("\n=== Fixing JIT Compilation Environment ===")
    
    # Set environment variables for JIT compilation
    env_vars = {
        'TORCH_CUDA_ARCH_LIST': '7.5;8.0;8.6',  # Common architectures
        'CUDA_LAUNCH_BLOCKING': '1',  # For debugging
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"Set {var}={value}")
    
    # Create a simplified build directory
    project_root = Path(__file__).parent.parent
    build_dir = project_root / "jit_build_fixed"
    build_dir.mkdir(exist_ok=True)
    
    print(f"Created build directory: {build_dir}")
    
    return str(build_dir)

def main():
    """Main fix function"""
    print("CUDA DLL Fix Tool")
    print("=" * 50)
    
    if sys.platform != "win32":
        print("This tool is for Windows only!")
        return
    
    # Step 1: Find existing CUDA DLLs
    found_dlls = find_cuda_dlls()
    
    # Step 2: Create compatibility links
    create_dll_symlinks()
    
    # Step 3: Update PATH
    add_cuda_to_path()
    
    # Step 4: Test DLL loading
    dll_success = test_dll_loading()
    
    # Step 5: Fix JIT environment
    build_dir = fix_jit_compilation()
    
    print("\n=== Summary ===")
    if dll_success:
        print("✓ CUDA DLL loading fixed!")
        print("✓ Ready to compile CUDA extensions")
    else:
        print("✗ Some CUDA DLLs still cannot be loaded")
        print("  You may need to:")
        print("  1. Install CUDA 11.3 runtime")
        print("  2. Or update PyTorch to match your CUDA 12.4 installation")
    
    print(f"\nNext steps:")
    print(f"1. Try running the CUDA extension compilation again")
    print(f"2. Use build directory: {build_dir}")
    print(f"3. If still failing, consider updating PyTorch:")
    print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")

if __name__ == "__main__":
    main()
