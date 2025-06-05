import torch
import cuda.fused_plane_line as fpl

# Dummy tensors
planes = torch.rand(3, 2, 4, 4).cuda()
lines = torch.rand(3, 2, 8).cuda()
coord_plane = torch.rand(3, 10, 2).cuda()
coord_line = torch.rand(3, 10).cuda()

output = fpl.forward(planes, lines, coord_plane, coord_line)
print(output[0].shape)