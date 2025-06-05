import os
from torch.utils.cpp_extension import load

_src_dir = os.path.dirname(__file__)
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
fused_plane_line = load(
    name="fused_plane_line",
    sources=[
        os.path.join(src_dir, "fused_plane_line.cpp"),
        os.path.join(src_dir, "fused_plane_line.cu"),
    ],
    verbose=True
)
# __all__ = ["fused_plane_line"]