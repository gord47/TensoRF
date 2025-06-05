import os
from torch.utils.cpp_extension import load

_src_dir = os.path.dirname(os.path.abspath(__file__))
'''
with open(os.path.join(_src_dir, "fused_plane_line.cu"), "r") as f:
    cuda_code = f.read()

fused_plane_line = load_inline(
    name="fused_plane_line",
    cpp_sources="",
    cuda_sources=cuda_code,
    functions=["fused_plane_line_forward"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)
'''
if os.name == 'nt':
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    os.add_dll_directory(torch_lib_dir)
    os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin")

fused_plane_line = load(
    name="fused_plane_line",
    sources=[
        os.path.join(_src_dir, "fused_plane_line.cpp"),
        os.path.join(_src_dir, "fused_plane_line.cu"),
    ],
    verbose=True
)
# __all__ = ["fused_plane_line