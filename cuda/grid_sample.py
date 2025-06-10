import os
import torch
from torch.utils.cpp_extension import load

_src_dir = os.path.dirname(os.path.abspath(__file__))

if os.name == 'nt':
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    os.add_dll_directory(torch_lib_dir)
    os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin")

grid_sample_cuda = load(
    name='grid_sample_cuda',
    sources=[
        os.path.join(_src_dir, 'grid_sample.cpp'),
        os.path.join(_src_dir, 'grid_sample_kernel.cu'),
    ],
    verbose=True
)

forward = grid_sample_cuda.forward