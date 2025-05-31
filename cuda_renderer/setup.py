from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name="cuda_renderer",
    ext_modules=[
        CUDAExtension(
            name="cuda_renderer",
            sources=[
                "csrc/cuda_renderer.cpp",
                "csrc/cuda_renderer_kernel.cu",
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
