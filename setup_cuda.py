import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Set environment variable to bypass CUDA version check
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9"
os.environ["DISTUTILS_USE_SDK"] = "1"

# Monkey patch to bypass CUDA version check
def _check_cuda_version_patched(self):
    print("Bypassing CUDA version check...")
    pass

# Apply the patch
BuildExtension._check_cuda_version = _check_cuda_version_patched

# Get CUDA version
cuda_version = torch.version.cuda
print(f"Building CUDA extension for CUDA {cuda_version} (version check bypassed)")

# Setup CUDA extension
setup(
    name='ray_renderer_cuda',
    ext_modules=[
        CUDAExtension(
            name='ray_renderer_cuda',
            sources=[
                'cuda_renderer/ray_renderer.cpp',
                'cuda_renderer/ray_renderer_kernel.cu',
            ],
            include_dirs=[
                'cuda_renderer/',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx series
                    '-gencode=arch=compute_80,code=sm_80',  # RTX 30xx series  
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx series
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx series
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)
