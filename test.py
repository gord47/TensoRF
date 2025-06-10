import torch
from models.tensoRF import TensorVMSplit
from opt import config_parser

# Test the updateAlphaMask fix
print('Testing updateAlphaMask tensor indexing fix...')

# Create a simple test configuration
class MockArgs:
    def __init__(self):
        self.datadir = './data/nerf_synthetic/lego'
        self.dataset_name = 'blender'
        self.white_bkgd = True
        self.half_res = False
        self.testskip = 8
        self.n_lamb_sigma = [16, 16, 16]
        self.n_lamb_sh = [48, 48, 48]
        self.shadingMode = 'MLP_PE'
        self.fea2denseAct = 'softplus'
        self.distance_scale = 25
        self.density_shift = -10
        self.rayMarch_weight_thres = 0.0001
        self.alpha_mask_thre = 0.001
        self.rm_weight_mask_thre = 0.0001
        self.batch_size = 4096
        self.vis = False
        self.N_vis = 5
        self.vis_every = 10000
        self.n_iters = 30000
        self.weight_distortion = 0.01
        self.weight_rgbper = 0.1
        self.lr_init = 0.02
        self.lr_basis = 1e-3
        self.lr_decay_iters = -1
        self.lr_decay_target_ratio = 0.1
        self.lr_upsample_reset = True
        self.L1_weight_inital = 8e-5
        self.L1_weight_rest = 4e-5
        self.rm_weight_mask_thre = 0.0001
        self.ckpt = None
        self.render_only = 0
        self.render_test = 0
        self.render_train = 0
        self.render_path = 0
        self.export_mesh = 0
        self.use_cuda_renderer = True

args = MockArgs()

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

aabb = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]], device=device)
reso_cur = [128, 128, 128]

tensorf = TensorVMSplit(aabb, reso_cur, device,
                       density_n_comp=args.n_lamb_sigma, 
                       appearance_n_comp=args.n_lamb_sh, 
                       app_dim=27, 
                       shadingMode=args.shadingMode, 
                       alphaMask_thres=args.alpha_mask_thre,
                       density_shift=args.density_shift, 
                       distance_scale=args.distance_scale,
                       pos_pe=6, view_pe=6, fea_pe=6, 
                       featureC=128, step_ratio=2.0, 
                       fea2denseAct=args.fea2denseAct)

print('Model initialized successfully')

# Test updateAlphaMask with a small grid
print('Testing updateAlphaMask...')
try:
    result = tensorf.updateAlphaMask(gridSize=(16, 16, 16))
    print(f'updateAlphaMask completed successfully!')
    print(f'Result shape: {result.shape}')
    print(f'Result: {result}')
except Exception as e:
    print(f'Error in updateAlphaMask: {e}')
    import traceback
    traceback.print_exc()