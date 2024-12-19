import json
import os
from abc import *

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import ControlNetModel
from diffusers.utils import is_compiled_module
from PIL import Image

from synctweedies.method_configs.case_config import * 
from synctweedies.model.base_model import BaseModel
from synctweedies.renderer.mesh.voronoi import voronoi_solve
from synctweedies.renderer.panorama.Equirec2Perspec import Equirectangular
from synctweedies.renderer.panorama.utils import *
from synctweedies.utils.image_utils import *
from synctweedies.utils.mesh_utils import *

from synctweedies.model.panorama_sd_model import PanoramaSDModel


class PanoramaSDModel_0_0_1(PanoramaSDModel):
    def __init__(self, config):
        super().__init__(config)
    
    def forward_ft(
        self, 
        canonical_input,   # torch.Size([1, 4, 2048, 4096])
        index, 
        xy_map_stack=None, # torch.Size([8, 64, 64, 2])
        **kwargs,
    ):
        
        xy = xy_map_stack[index].to(canonical_input)
        # instance_out = remap_torch(canonical_input, xy[..., 0], xy[..., 1]) # torch.Size([1, 4, 64, 64])
        instance_out = None
        return instance_out
    
    def inverse_ft(
        self, 
        screen_input, 
        index, 
        orig_canonical_input, 
        xy_map_stack=None, 
        **kwargs,
    ):
        xy = xy_map_stack[index]
        pano_height, pano_width = (
            orig_canonical_input.shape[-2],
            orig_canonical_input.shape[-1],
        )
        # canonical_out, mask = wrapper_perspective_to_pano_torch(
        #     screen_input, xy, pano_height, pano_width
        # )
        return canonical_out, mask

    def forward_ft(
        self, 
        canonical_input,   # torch.Size([1, 4, 2048, 4096])
        index, 
        xy_map_stack=None, # torch.Size([8, 64, 64, 2])
        **kwargs,
    ):
        
        xy = xy_map_stack[index].to(canonical_input)
        instance_out = remap_torch(canonical_input, xy[..., 0], xy[..., 1]) # torch.Size([1, 4, 64, 64])
        return instance_out
    

    def inverse_ft(
        self, 
        screen_input, 
        index, 
        orig_canonical_input, 
        xy_map_stack=None, 
        **kwargs,
    ):
        xy = xy_map_stack[index]
        pano_height, pano_width = (
            orig_canonical_input.shape[-2],
            orig_canonical_input.shape[-1],
        )
        canonical_out, mask = wrapper_perspective_to_pano_torch(
            screen_input, xy, pano_height, pano_width
        )
        return canonical_out, mask
    
    @torch.no_grad()
    def cond_noise_upsample(self, src_noise):
        B, C, H, W = src_noise.shape
        up_factor = 2 ** self.up_level if self.up_level is not None else 2 ** 3
        upscaled_means = F.interpolate(src_noise, scale_factor=(up_factor, up_factor), mode='nearest')

        up_H = up_factor * H
        up_W = up_factor * W

        # 1) Unconditionally sample a discrete Nk x Nk Gaussian sample
        raw_rand = torch.randn(B, C, up_H, up_W, device=src_noise.device)

        # 2) Remove its mean from it
        Z_mean = raw_rand.unfold(2, up_factor, up_factor).unfold(3, up_factor, up_factor).mean((4, 5))
        Z_mean = F.interpolate(Z_mean, scale_factor=up_factor, mode='nearest')
        mean_removed_rand = raw_rand - Z_mean

        # 3) Add the pixel value to it
        up_noise = upscaled_means / up_factor + mean_removed_rand
        return up_noise # sqrt(N_k)* W(A_k): sub-pixel noise scaled with sqrt(N_k). So, ~ N(0, 1)

    @torch.no_grad()
    def discrete_warping(self, src_noise, hw=(512//8, 512//8)):
        B, C, H, W = src_noise.shape
        h, w = hw

        # ERP pixel coordinate (normalized to [-1, 1])
        erp_grid = make_coord((H, W), flatten=False).to(self.device) # (H, W, 2)
        
        # Pers. pixel index (range: [1, hw])
        pers_idx = torch.arange(1, h*w+1, device=self.device, dtype=torch.float32).view(1, h, w) # (1, h, w)

        # flatten ERP noise
        erp_up_noise_flat = src_noise.reshape(B*C, -1)
        # flatten count value
        ones_flat = torch.ones_like(erp_up_noise_flat[:1])

        pers_noises = []
        erp2pers_idxs = []

        for theta, phi in self.views:

            # Warp ERP pixel grid into Pers. coordinate
            erp2pers_grid, valid_mask = gridy2x_erp2pers(gridy=erp_grid,
                HWy=(H, W), HWx=hw, THETA=theta, PHI=phi, FOVy=360, FOVx=90)
            erp2pers_grid = erp2pers_grid.view(H, W, 2)
            valid_mask = valid_mask.view(1, H, W)
            
            # Find which index of Pers. is mapped to ERP pixel
            erp2pers_idx = F.grid_sample(
                pers_idx.unsqueeze(0),
                erp2pers_grid.unsqueeze(0).flip(-1),
                mode="nearest", align_corners=False)[0] # (1, H, W)
            erp2pers_idx *= valid_mask # non-mapped ERP pixel has 0 index
            erp2pers_idx = erp2pers_idx.to(torch.int64) # index's dtype must be dtype

            # Get warped Pers. pixel noise
            ind_flat = erp2pers_idx.view(1, -1)
            fin_v_val = torch.zeros(B*C, h*w+1, device=self.device) \
                .scatter_add_(1, index=ind_flat.repeat(B*C, 1), src=erp_up_noise_flat)[..., 1:]
            fin_v_num = torch.zeros(1, h*w+1, device=self.device) \
                .scatter_add_(1, index=ind_flat, src=ones_flat)[..., 1:]
            assert fin_v_num.min() != 0, ValueError(f"{theta},{phi} has 0 fin_v_num.")

            # Normalize fin_v_val to scale Variance of Pers. pixel noise
            final_values = fin_v_val / torch.sqrt(fin_v_num)

            pers_noises.append(final_values.reshape(B, C, h, w).float())
            erp2pers_idxs.append(erp2pers_idx.reshape((H, W)))

        return pers_noises, erp2pers_idxs