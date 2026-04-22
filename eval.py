import argparse
from copy import deepcopy
import logging
from data.CNS_data_utils import FNODatasetSingle, FNODatasetMultistep
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils.metrics import *
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from einops import rearrange
from models.diff_afno_sit import SiT_models
import math
from torchvision.utils import make_grid
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = get_logger(__name__)

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x

@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Testing Loop                                #
#################################################################################
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    #* 替换为新的exp的name
    parser.add_argument("--exp-name", type=str, default="fourierflow_v2_0421-17:53")
    parser.add_argument("--flnm", type=str, default="2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5")
    parser.add_argument("--logging-dir", type=str, default="exps/logs")
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--ckpt-step", type=int, default=135000)
    parser.add_argument("--test-subset", type=str, default="")

    # model
    parser.add_argument("--model", type=str,default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=3)
    parser.add_argument("--spa-depth", type=int, default=6)
    parser.add_argument("--tem-depth", type=int, default=3)
    parser.add_argument("--afno-depth", type=int, default=6)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[128,256], default=128)
    parser.add_argument("--reduced-resolution", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--mixer", type=str, default="wavelet")
    parser.add_argument("--wavelet-level", type=int, default=2)
    parser.add_argument("--wavelet-wave", type=str, default="db4")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    args.logging_dir = os.path.join(args.logging_dir,args.exp_name)
        
    return args

def main(args):    
    os.makedirs(args.logging_dir, exist_ok=True)
    logger = create_logger(args.logging_dir)
    logger.info(f"Experiment directory created at {args.logging_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
 
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create model:
    latent_size = 512 // args.reduced_resolution

    z_dims = [256]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        spa_depth=args.spa_depth,
        tem_depth=args.tem_depth,
        afno_depth=args.afno_depth,
        mixer=args.mixer,
        wavelet_level=args.wavelet_level,
        wavelet_wave=args.wavelet_wave,
        **block_kwargs
    )
    ckpt_name = str(args.ckpt_step).zfill(7) +'.pt'
    ckpt = torch.load(
        f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
        map_location='cpu',
        )["model"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in ckpt.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    # model.load_state_dict(ckpt['model'])
    model.load_state_dict(new_state_dict)

    model = model.to(device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # flnm = '2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'
    flnm = args.flnm
    base_path='data/'
    reduce_resolution = args.reduced_resolution
    reduced_batch = 1

    if args.test_subset:
        logger.info(f"Loading test subset from {args.test_subset}...")
        import h5py
        with h5py.File(args.test_subset, 'r') as f:
            # Reconstruct the dataset object minimally
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, f):
                    # structure in subset: [B, T, H, W] -> [B, H, W, T, C]
                    d = np.array(f['density'])
                    p = np.array(f['pressure'])
                    vx = np.array(f['Vx'])
                    vy = np.array(f['Vy'])
                    self.data = np.stack([d, p, vx, vy], axis=-1) # [B, T, H, W, C]
                    self.data = np.transpose(self.data, (0, 2, 3, 1, 4)) # [B, H, W, T, C]
                    
                    x = np.array(f['x-coordinate'])
                    y = np.array(f['y-coordinate'])
                    X, Y = np.meshgrid(x, y, indexing='ij')
                    self.grid = torch.stack((torch.tensor(X), torch.tensor(Y)), axis=-1)
                    
                    self.mean = np.array(f['norm_mean'])
                    self.std = np.array(f['norm_std'])
                    self.eps = 1e-8
                    
                    # Normalize
                    self.data = (self.data - self.mean) / (self.std + self.eps)
                    self.data = torch.tensor(self.data, dtype=torch.float32)

                def __len__(self): return len(self.data)
                def __getitem__(self, idx):
                    # Same logic as Multistep dataset
                    k = 4
                    return self.data[idx,...,k:k*2,:], self.grid, self.data[idx,...,0:k,:]

            test_dataset = SimpleDataset(f)
            
            class SimpleNormalizer:
                def __init__(self, m, s): self.mean, self.std, self.eps = m, s, 1e-8
                def decode(self, x): return (x * (self.std + self.eps)) + self.mean
            
            normalizer = SimpleNormalizer(test_dataset.mean, test_dataset.std)
    else:
        train_dataset, test_dataset,normalizer = FNODatasetMultistep.get_train_test_datasets(
                                        flnm,
                                        reduced_resolution=reduce_resolution,
                                        reduced_resolution_t=1,
                                        reduced_batch=reduced_batch,
                                        initial_step=0,
                                        saved_folder=base_path,
                                        if_eval_plot=True
                                    )
    local_batch_size = 8
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f'==== {next(iter(test_dataloader))[0].mean().item():.6f} ====')
    
    model.eval()  # important! This enables embedding dropout for classifier-free guidance
               
    from samplers import euler_sampler
    _err_RMSE_avg = 0
    _err_nRMSE_avg = 0
    _err_max_avg = 0
    with torch.no_grad():
        # test_iter = iter(test_dataloader)
        # target_test, grid_test, raw_image_test = next(test_iter)
        for target_test, grid_test, raw_image_test in test_dataloader:
            raw_image_test = rearrange(raw_image_test, "B H W T C -> B T C H W").to(device)
            target_test = rearrange(target_test, "B H W T C -> B T C H W").to(device)
            sample_input = torch.randn_like(target_test, device=device)
            samples = euler_sampler(
                model, 
                sample_input, 
                raw_image_test,
                num_steps=3, 
                cfg_scale=4.0,
                guidance_low=0.,
                guidance_high=1.,
                path_type=args.path_type,
                heun=False,
            ).to(torch.float32)
            Lx, Ly, Lz = 1., 1., 1.
            _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
            = metric_func(samples, target_test, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)
            _err_RMSE_avg += _err_RMSE.item()
            _err_nRMSE_avg += _err_nRMSE.item()
            _err_max_avg += _err_Max.item()
        _err_RMSE_avg /= len(test_dataloader)
        _err_nRMSE_avg /= len(test_dataloader)
        _err_max_avg /= len(test_dataloader)
        
        logger.info(f'RMSE: {_err_RMSE_avg:.4f}, nRMSE: {_err_nRMSE_avg:.4f}, Max:{_err_max_avg:.4f}')
    vis_dir = os.path.join(args.output_dir, 'visualizations', args.exp_name)
    os.makedirs(vis_dir, exist_ok=True)
    samples = rearrange(samples, "B T C H W -> B H W T C")
    target_test = rearrange(target_test, "B T C H W -> B H W T C")
    samples = normalizer.decode(samples.cpu())
    target_test = normalizer.decode(target_test.cpu())

    # only visualize Vx (ch2) and Vy (ch3)
    vis_channels = [2, 3]
    ch_names = ["Vx", "Vy"]
    T = samples.size(-2)

    for i in range(min(3, samples.size(0))):
        # layout: rows = [Pred Vx, GT Vx, Pred Vy, GT Vy], cols = timesteps
        n_rows = len(vis_channels) * 2  # 4
        fig, axes = plt.subplots(n_rows, T, figsize=(T * 3.5, n_rows * 3.2))

        # row order: Pred Vx, GT Vx, Pred Vy, GT Vy
        row_configs = []
        for ch_name, ch_idx in zip(ch_names, vis_channels):
            row_configs.append(("Prediction", ch_name, ch_idx, samples))
            row_configs.append(("Ground Truth", ch_name, ch_idx, target_test))

        for row, (src_label, ch_name, ch_idx, data) in enumerate(row_configs):
            for k in range(T):
                ax = axes[row, k]
                img = data[i, :, :, k, ch_idx].numpy()
                ch_data = data[i, :, :, :, ch_idx].numpy()
                vmin, vmax = ch_data.min(), ch_data.max()
                ax.imshow(img, cmap='coolwarm', vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f't = {k+1}', fontsize=9)
                # colored border: orange for Prediction, teal for Ground Truth
                color = '#E87722' if src_label == 'Prediction' else '#1A936F'
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            # row label on left
            axes[row, 0].set_ylabel(
                f'{src_label}\n{ch_name}', fontsize=9,
                rotation=90, labelpad=8, va='center',
                color='#E87722' if src_label == 'Prediction' else '#1A936F',
                fontweight='bold'
            )

        fig.suptitle(
            f'Sample {i+1}   |   RMSE = {_err_RMSE_avg:.4f}   nRMSE = {_err_nRMSE_avg:.4f}',
            fontsize=10, y=1.01
        )
        plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.3)
        fig_path = os.path.join(vis_dir, f'sample_{i+1}.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {fig_path}")
        plt.close(fig)

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
