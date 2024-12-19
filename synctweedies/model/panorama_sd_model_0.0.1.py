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
