from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='grid_sample_cuda',
    version='0.1',
    ext_modules=[
        CUDAExtension('grid_sample_cuda', [
            'grid_sample.cpp',
            'grid_sample_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)