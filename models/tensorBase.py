import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time


def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb



class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device=device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio


        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]


        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])


    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox


    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize
        print(f"[DEBUG] getDenseAlpha called with gridSize: {gridSize}")

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        print(f"[DEBUG] Samples shape: {samples.shape}")
        
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples
        print(f"[DEBUG] Dense xyz shape: {dense_xyz.shape}")
        print(f"[DEBUG] Dense xyz range - min: {dense_xyz.min().item():.6f}, max: {dense_xyz.max().item():.6f}")

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[...,0])
        print(f"[DEBUG] Initial alpha shape: {alpha.shape}")
        
        try:
            for i in range(gridSize[0]):
                print(f"[DEBUG] Processing slice {i}/{gridSize[0]}")
                xyz_slice = dense_xyz[i].view(-1,3)
                print(f"[DEBUG] xyz_slice shape: {xyz_slice.shape}, stepSize: {self.stepSize}")
                
                alpha_slice = self.compute_alpha(xyz_slice, self.stepSize)
                print(f"[DEBUG] alpha_slice shape: {alpha_slice.shape}, stats - min: {alpha_slice.min().item():.6f}, max: {alpha_slice.max().item():.6f}")
                
                alpha[i] = alpha_slice.view((gridSize[1], gridSize[2]))
                
            print(f"[DEBUG] Final alpha stats - min: {alpha.min().item():.6f}, max: {alpha.max().item():.6f}, mean: {alpha.mean().item():.6f}")
            return alpha, dense_xyz
            
        except Exception as e:
            print(f"[ERROR] getDenseAlpha failed with error: {e}")
            print(f"[ERROR] Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):
        print(f"[DEBUG] updateAlphaMask called with gridSize={gridSize}")
        print(f"[DEBUG] Current alphaMask_thres: {self.alphaMask_thres}")
        print(f"[DEBUG] Current device: {self.device}")
        print(f"[DEBUG] Current aabb: {self.aabb}")

        try:
            alpha, dense_xyz = self.getDenseAlpha(gridSize)
            print(f"[DEBUG] getDenseAlpha completed - alpha shape: {alpha.shape}, dense_xyz shape: {dense_xyz.shape}")
            print(f"[DEBUG] Alpha stats - min: {alpha.min().item():.6f}, max: {alpha.max().item():.6f}, mean: {alpha.mean().item():.6f}")
            print(f"[DEBUG] Alpha values > 0: {(alpha > 0).sum().item()}/{alpha.numel()}")
            print(f"[DEBUG] Alpha values > alphaMask_thres ({self.alphaMask_thres}): {(alpha > self.alphaMask_thres).sum().item()}")
            
            dense_xyz = dense_xyz.transpose(0,2).contiguous()
            alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
            total_voxels = gridSize[0] * gridSize[1] * gridSize[2]
            print(f"[DEBUG] After transpose and clamp - alpha shape: {alpha.shape}")

            ks = 3
            alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
            print(f"[DEBUG] After max_pool3d - alpha shape: {alpha.shape}")
            print(f"[DEBUG] Alpha stats after pooling - min: {alpha.min().item():.6f}, max: {alpha.max().item():.6f}")
            
            # Apply thresholding and debug statistics before/after
            pre_threshold_stats = {
                'min': alpha.min().item(),
                'max': alpha.max().item(), 
                'mean': alpha.mean().item(),
                'above_thres': (alpha >= self.alphaMask_thres).sum().item(),
                'total': alpha.numel()
            }
            print(f"[DEBUG] Pre-threshold alpha stats: {pre_threshold_stats}")
            
            alpha[alpha>=self.alphaMask_thres] = 1
            alpha[alpha<self.alphaMask_thres] = 0
            
            post_threshold_stats = {
                'active_voxels': (alpha > 0.5).sum().item(),
                'total_voxels': alpha.numel(),
                'percentage': (alpha > 0.5).sum().item() / alpha.numel() * 100
            }
            print(f"[DEBUG] Post-threshold alpha stats: {post_threshold_stats}")

            # Check for empty alpha mask before proceeding
            if post_threshold_stats['active_voxels'] == 0:
                print(f"[ERROR] CRITICAL: No active voxels after thresholding!")
                print(f"[ERROR] alphaMask_thres: {self.alphaMask_thres}")
                print(f"[ERROR] This suggests alpha values are all below threshold")
                print(f"[ERROR] Recommended: Lower alphaMask_thres or check density computation")
                
                # Create a minimal alpha mask to prevent complete failure
                print(f"[FALLBACK] Creating minimal alpha mask to prevent crash")
                center_idx = [s//2 for s in gridSize[::-1]]  # gridSize is reversed for alpha
                alpha[center_idx[0]-1:center_idx[0]+2, 
                      center_idx[1]-1:center_idx[1]+2, 
                      center_idx[2]-1:center_idx[2]+2] = 1.0
                print(f"[FALLBACK] Added {(alpha > 0.5).sum().item()} central voxels")

            self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)
            print(f"[DEBUG] AlphaGridMask created successfully")

            # Get valid coordinates from thresholded alpha
            # Fix tensor shape mismatch: dense_xyz is [H,W,D,3] but alpha is [D,W,H]
            # We need to transpose alpha to match dense_xyz dimensions
            print(f"[DEBUG] Valid xyz extraction:")
            print(f"[DEBUG]   - dense_xyz shape: {dense_xyz.shape}")
            print(f"[DEBUG]   - alpha shape: {alpha.shape}")
            
            # Transpose alpha to match dense_xyz indexing: [D,W,H] -> [H,W,D]
            alpha_for_indexing = alpha.permute(2, 1, 0)
            print(f"[DEBUG]   - alpha_for_indexing shape: {alpha_for_indexing.shape}")
            print(f"[DEBUG]   - alpha_for_indexing > 0.5 count: {(alpha_for_indexing > 0.5).sum().item()}")
            
            # Now extract valid coordinates with properly shaped mask
            valid_xyz = dense_xyz[alpha_for_indexing > 0.5]
            print(f"[DEBUG]   - valid_xyz shape: {valid_xyz.shape}")
            print(f"[DEBUG]   - valid_xyz numel: {valid_xyz.numel()}")
            
            # Double-check for empty tensor before amin/amax operations
            if valid_xyz.numel() == 0:
                print(f"[ERROR] CRITICAL: valid_xyz is empty after extraction!")
                print(f"[ERROR] This should not happen after fallback - possible tensor indexing issue")
                print(f"[ERROR] Alpha stats: min={alpha.min().item():.6f}, max={alpha.max().item():.6f}")
                print(f"[ERROR] Active voxels in alpha: {(alpha > 0.5).sum().item()}")
                print(f"[ERROR] Active voxels in alpha_for_indexing: {(alpha_for_indexing > 0.5).sum().item()}")
                print(f"[ERROR] Falling back to original aabb to prevent crash")
                return self.aabb
            
            # Additional safety check for tensor dimensions
            if len(valid_xyz.shape) != 2 or valid_xyz.shape[1] != 3:
                print(f"[ERROR] CRITICAL: valid_xyz has wrong shape: {valid_xyz.shape}")
                print(f"[ERROR] Expected: [N, 3] where N > 0")
                print(f"[ERROR] Falling back to original aabb to prevent crash")
                return self.aabb
                
            print(f"[DEBUG] Computing bounding box from {valid_xyz.shape[0]} valid points")
            xyz_min = valid_xyz.amin(0)
            xyz_max = valid_xyz.amax(0)
            print(f"[DEBUG] xyz_min: {xyz_min}, xyz_max: {xyz_max}")

            new_aabb = torch.stack((xyz_min, xyz_max))
            print(f"[DEBUG] new_aabb shape: {new_aabb.shape}")

            total = torch.sum(alpha)
            print(f"[DEBUG] bbox: {xyz_min, xyz_max} alpha rest %{total/total_voxels*100:.2f}")
            return new_aabb
            
        except Exception as e:
            print(f"[ERROR] updateAlphaMask failed with error: {e}")
            print(f"[ERROR] Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            # Return original aabb to prevent crash
            return self.aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False): 
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):
        print(f"[ALPHA_DEBUG] compute_alpha called with xyz_locs shape: {xyz_locs.shape}, length: {length}")
        print(f"[ALPHA_DEBUG] Input xyz_locs range - min: {xyz_locs.min().item():.6f}, max: {xyz_locs.max().item():.6f}")

        if self.alphaMask is not None:
            try:
                alphas = self.alphaMask.sample_alpha(xyz_locs)
                alpha_mask = alphas > 0
                print(f"[ALPHA_DEBUG] AlphaMask sampling - alphas shape: {alphas.shape}, active points: {alpha_mask.sum().item()}/{alpha_mask.numel()}")
                print(f"[ALPHA_DEBUG] AlphaMask alphas range - min: {alphas.min().item():.6f}, max: {alphas.max().item():.6f}")
            except Exception as e:
                print(f"[ALPHA_ERROR] AlphaMask sampling failed: {e}")
                alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            print(f"[ALPHA_DEBUG] No AlphaMask, using all points: {alpha_mask.sum().item()}")

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        print(f"[ALPHA_DEBUG] Initial sigma shape: {sigma.shape}")

        if alpha_mask.any():
            try:
                xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
                print(f"[ALPHA_DEBUG] Normalized coordinates shape: {xyz_sampled.shape}")
                print(f"[ALPHA_DEBUG] Normalized coords range - min: {xyz_sampled.min().item():.6f}, max: {xyz_sampled.max().item():.6f}")
                
                sigma_feature = self.compute_densityfeature(xyz_sampled)
                print(f"[ALPHA_DEBUG] Density features shape: {sigma_feature.shape}")
                print(f"[ALPHA_DEBUG] Density features stats - min: {sigma_feature.min().item():.6f}, max: {sigma_feature.max().item():.6f}, mean: {sigma_feature.mean().item():.6f}")
                
                validsigma = self.feature2density(sigma_feature)
                print(f"[ALPHA_DEBUG] Valid sigma shape: {validsigma.shape}")
                print(f"[ALPHA_DEBUG] Valid sigma stats - min: {validsigma.min().item():.6f}, max: {validsigma.max().item():.6f}, mean: {validsigma.mean().item():.6f}")
                
                sigma[alpha_mask] = validsigma
                
            except Exception as e:
                print(f"[ALPHA_ERROR] Density computation failed: {e}")
                print(f"[ALPHA_ERROR] Exception type: {type(e)}")
                import traceback
                traceback.print_exc()
                # Set sigma to zero to prevent crashes
                sigma.fill_(0.0)

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])
        print(f"[ALPHA_DEBUG] Final alpha stats - min: {alpha.min().item():.6f}, max: {alpha.max().item():.6f}, mean: {alpha.mean().item():.6f}")
        print(f"[ALPHA_DEBUG] Alpha values > 0: {(alpha > 0).sum().item()}/{alpha.numel()}")
        print(f"[ALPHA_DEBUG] Alpha values > 1e-6: {(alpha > 1e-6).sum().item()}")
        print(f"[ALPHA_DEBUG] Alpha values > 1e-4: {(alpha > 1e-4).sum().item()}")
        print(f"[ALPHA_DEBUG] Alpha values > 1e-3: {(alpha > 1e-3).sum().item()}")

        return alpha


    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid


        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

