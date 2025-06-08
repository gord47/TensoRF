import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
import torch.cuda.nvtx as nvtx

# Import CUDA renderer - REQUIRED FOR TRAINING (no fallbacks)
try:
    from cuda_renderer import CudaRayRenderer
    cuda_renderer = CudaRayRenderer()
    print("CUDA Ray Renderer imported successfully!")
    CUDA_RENDERER_AVAILABLE = True
except Exception as e:
    print(f"CRITICAL ERROR: CUDA Ray Renderer failed to load: {e}")
    print("Training requires CUDA acceleration and will not work without it.")
    CUDA_RENDERER_AVAILABLE = False
    cuda_renderer = None
    # Note: We don't raise here to allow import, but functions will fail when called


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
    """
    CUDA-only version of the ray renderer (no fallback).
    
    This function requires CUDA acceleration and will fail if CUDA is not available.
    Use OctreeRender_trilinear_fast_cuda for the same functionality with better error messages.
    """
    nvtx.range_push("OctreeRender_trilinear_fast")
    
    if not CUDA_RENDERER_AVAILABLE:
        raise RuntimeError("CUDA Ray Renderer is not available. Training requires CUDA acceleration.")
    
    if not cuda_renderer.cuda_available:
        raise RuntimeError("CUDA Ray Renderer failed to initialize. Training requires CUDA acceleration.")
        
    if ndc_ray:
        raise NotImplementedError("NDC rays are not yet supported in CUDA implementation")
    
    try:
        nvtx.range_push("CUDA Ray Rendering")
        # Use CUDA renderer for all rays at once (no chunking needed)
        rgb_maps, depth_maps = cuda_renderer.render_rays(
            rays=rays.to(device), 
            tensorf_model=tensorf,
            chunk=chunk,
            N_samples=N_samples,
            white_bg=white_bg,
            is_train=is_train
        )
        nvtx.range_pop()
        nvtx.range_pop()
        return rgb_maps, None, depth_maps, None, None
    except Exception as e:
        nvtx.range_pop()
        nvtx.range_pop()
        raise RuntimeError(f"CUDA rendering failed: {e}")

def OctreeRender_trilinear_fast_cuda(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
    """
    CUDA-accelerated version of OctreeRender_trilinear_fast
    
    This function forces CUDA acceleration and will raise an error if CUDA is not available.
    Use this for performance-critical applications where CUDA acceleration is required.
    """
    nvtx.range_push("OctreeRender_trilinear_fast_cuda")
    
    if not CUDA_RENDERER_AVAILABLE:
        raise RuntimeError("CUDA Ray Renderer is not available. Please install the CUDA extension.")
    
    if not cuda_renderer.cuda_available:
        raise RuntimeError("CUDA Ray Renderer failed to initialize.")
        
    if ndc_ray:
        raise NotImplementedError("NDC rays are not yet supported in CUDA implementation")
    
    try:
        nvtx.range_push("CUDA Ray Rendering")
        # Use CUDA renderer for all rays at once (no chunking needed)
        rgb_maps, depth_maps = cuda_renderer.render_rays(
            rays=rays.to(device), 
            tensorf_model=tensorf,
            chunk=chunk,
            N_samples=N_samples,
            white_bg=white_bg,
            is_train=is_train
        )
        nvtx.range_pop()
        nvtx.range_pop()
        return rgb_maps, None, depth_maps, None, None
    except Exception as e:
        nvtx.range_pop()
        nvtx.range_pop()
        raise RuntimeError(f"CUDA rendering failed: {e}")

@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):
        nvtx.range_push(f"Eval idx {idx}")
        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])
        nvtx.range_push("Render rays")
        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        nvtx.range_pop()
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                nvtx.range_push("Compute metrics")
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)
                nvtx.range_pop()

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)
        nvtx.range_pop()

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

