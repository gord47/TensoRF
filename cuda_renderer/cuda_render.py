from cuda_renderer import OctreeRenderCUDA
import torch
import torch.cuda.nvtx as nvtx

def OctreeRender_CUDA(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
    """
    CUDA-accelerated version of OctreeRender_trilinear_fast
    """
    nvtx.range_push("OctreeRender_CUDA")
    
    # Create CUDA renderer
    cuda_renderer = OctreeRenderCUDA(distance_scale=tensorf.distance_scale)
    
    # Process rays in chunks for memory efficiency
    rgbs, depth_maps = [], []
    N_rays_all = rays.shape[0]
    
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        nvtx.range_push(f"Process chunk {chunk_idx}")
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        
        # Call CUDA renderer
        nvtx.range_push("CUDA Render Forward")
        rgb_map, depth_map = cuda_renderer(
            rays_chunk, tensorf, N_samples=N_samples, 
            ndc_ray=ndc_ray, white_bg=white_bg
        )
        nvtx.range_pop()  # CUDA Render Forward
        
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        nvtx.range_pop()  # Process chunk
    
    nvtx.range_pop()  # OctreeRender_CUDA
    
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None
