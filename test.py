import torch
import torch.nn.functional as F
import cuda.grid_sample as grid_sample_cuda

# Parameters
device = "cuda"
N, C, H, W = 2, 3, 128, 128
out_H, out_W = 64, 64

# Create tensors with gradient tracking
input_pt = torch.randn(N, C, H, W, device=device, requires_grad=True)
grid_pt = torch.rand(N, out_H, out_W, 2, device=device) * 2 - 1
grid_pt.requires_grad = True

input_cu = input_pt.detach().clone().requires_grad_()
grid_cu = grid_pt.detach().clone().requires_grad_()

# Forward pass
output_pt = F.grid_sample(input_pt, grid_pt, align_corners=True)
output_cu = grid_sample_cuda.forward(input_cu, grid_cu, True)

# Backward pass
grad_output = torch.randn_like(output_pt)
output_pt.backward(grad_output)
output_cu.backward(grad_output)

# Compare
print("Forward max diff:", torch.max(torch.abs(output_pt - output_cu)).item())
print("Input grad max diff:", torch.max(torch.abs(input_pt.grad - input_cu.grad)).item())
print("Grid grad max diff:", torch.max(torch.abs(grid_pt.grad - grid_cu.grad)).item())

# Check for NaNs
assert not torch.isnan(output_cu).any(), "Custom output contains NaNs"
assert not torch.isnan(input_cu.grad).any(), "Custom input grad contains NaNs"
assert not torch.isnan(grid_cu.grad).any(), "Custom grid grad contains NaNs"