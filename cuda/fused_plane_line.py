import os
import torch
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
        os.path.join(_src_dir, "fused_plane_line_single.cu"),
    ],
    verbose=True
)
forward = fused_plane_line.forward
forward_single = fused_plane_line.forward_single

def forward_split(planes, lines, coords_plane, coords_line):
    """
    Process plane-line computation with tensors of different sizes by handling each axis separately.
    
    Args:
        planes: List of 3 tensors of shape [1, C, H_i, W_i] for each axis
        lines: List of 3 tensors of shape [1, C, L_i, 1] for each axis
        coords_plane: Tensor of shape [3, N, 2] containing plane coordinates
        coords_line: Tensor of shape [3, N] containing line coordinates
        
    Returns:
        Tensor of shape [N] containing the output features
    """
    device = planes[0].device
    N = coords_plane.size(1)
    
    # Initialize output tensor
    output = torch.zeros(N, device=device)
    
    # Process each axis separately
    for i in range(3):
        plane_i = planes[i]
        line_i = lines[i]
        print(plane_i.shape, line_i.shape)
        coord_plane_i = coords_plane[i]  # [N, 2]
        coord_line_i = coords_line[i]    # [N]
        
        # Call the CUDA kernel for this component
        forward_single(plane_i, line_i, coord_plane_i, coord_line_i, output)
    
    return output