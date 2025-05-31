import torch
import cuda_renderer

class OctreeRenderCUDA(torch.nn.Module):
    def __init__(self, distance_scale=1.0):
        super().__init__()
        self.distance_scale = distance_scale

    def forward(self, rays, tensorf, N_samples=-1, ndc_ray=False, white_bg=True):
        """
        Octree Render CUDA implementation
        
        Args:
            rays: ray origins and directions, shape [N_rays, 6]
            tensorf: TensorRF model containing density and appearance grids
            N_samples: number of samples per ray, if -1 will be automatically calculated
            ndc_ray: whether rays are in NDC space
            white_bg: whether to use white background
            
        Returns:
            rgb_map: rendered RGB, shape [N_rays, 3]
            depth_map: rendered depth, shape [N_rays, 1]
        """
        # Extract tensorf components
        density_plane = tensorf.density_plane
        density_line = tensorf.density_line
        app_plane = tensorf.app_plane
        app_line = tensorf.app_line
        basis_mat = tensorf.basis_mat.weight
        aabb = tensorf.aabb
        
        if N_samples <= 0:
            N_samples = tensorf.cal_n_samples(rays)
        
        # Call CUDA function
        results = cuda_renderer.octree_render_forward(
            rays, density_plane, density_line, app_plane, app_line,
            basis_mat, aabb, N_samples, ndc_ray, white_bg, self.distance_scale
        )
        
        # Process results
        rgb_map = results[..., :3]
        depth_map = results[..., 3]
        
        return rgb_map, depth_map
