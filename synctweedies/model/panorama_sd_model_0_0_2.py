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

from synctweedies.model.panorama_sd_model_0_0_0 import PanoramaSDModel_0_0_0

from diffusers.models.vae import Decoder as VAEDecoder
from typing import Optional

'''
1) Redefine Decoder::forward to process input step-by-step (step i as an additional input)
2) Redifine VAE::decode to process multiple latents
3) Fuse latents in each steps
'''
class PanoramaSDModel_0_0_2(PanoramaSDModel_0_0_0):
    def __init__(self, config):
        super().__init__(config)
    
    def inverse_mapping(
        self,
        x_ts,
        xy_map_stack=None,
        mask_stack=None,
        decoding=False,
        voronoi=None,
        num_channels=None,
        canonical_h=None,
        canonical_w=None,
        **kwargs,
    ):
        if voronoi is None:
            voronoi = self.config.voronoi

        x_ts = x_ts.to(self.device)
        if num_channels is None:
            num_channels = 3 if self.config.average_rgb else 4
        if canonical_h is None:
            canonical_h = (
                self.config.canonical_rgb_h
                if self.config.average_rgb
                else self.config.canonical_latent_h
            )
        if canonical_w is None:
            canonical_w = (
                self.config.canonical_rgb_w
                if self.config.average_rgb
                else self.config.canonical_latent_w
            )

        pano = torch.zeros(1, num_channels, canonical_h, canonical_w).to(x_ts)
        pano_count = torch.zeros(1, 1, canonical_h, canonical_w, device=self.device)
        if decoding:
            decoded_x_ts = self.decode_latents_with_fusing(x_ts, xy_map_stack)
            x_ts = torch.cat(decoded_x_ts, 0).float()
            assert x_ts.shape == (
                self.num_views,
                3,
                self.config.instance_rgb_size,
                self.config.instance_rgb_size,
            ), f"decoded_x_ts: {x_ts.shape}"

        for i in range(self.num_views):
            z_i, mask_i = self.inverse_ft(x_ts[i : i + 1], i, pano, xy_map_stack)

            if voronoi:
                # Voronoi filling
                tmp = voronoi_solve(z_i[0].permute(1, 2, 0), mask_i[0, 0])
                z_i = tmp.permute(2, 0, 1).unsqueeze(0)
                mask_i = mask_stack[i].unsqueeze(0).unsqueeze(0)
                z_i = z_i * mask_i
            pano = pano + z_i
            pano_count = pano_count + mask_i

        z_t = pano / (pano_count + 1e-8)

        return z_t

    @torch.no_grad()
    def decode_latents_with_fusing(self, x_ts, xy_map_stack):
        latents = []
        custom_decoder = CustomDecoder(self.model.vae.decoder).to(self.device)
        post_quant_conv = self.model.vae.post_quant_conv.to(self.device)
    
        # Preprocess
        for i in range(self.num_views):
            latents_i = x_ts[i: i + 1].half()
            latents_i = 1 / 0.18215 * latents_i
            latents.append(latents_i)

        # Decoding
        for i in range(self.num_views):
            latents[i] = post_quant_conv(latents[i])

        for i in range(self.num_views):
            latents[i] = custom_decoder(latents[i], "pre")

        for i in range(len(custom_decoder.up_blocks)):
            for j in range(self.num_views):
                latents[j] = custom_decoder(latents[j], "up", i)
            # latents[i]: (1, 128, 512, 512)
            pano_latent = torch.zeros(1, *latents[0].shape[1:]).to(self.device)
            pano_latent_count = torch.zeros(1, 1, *latents[0].shape[2:], device=self.device)
            for j in range(self.num_views):
                z_j, mask_j = self.inverse_ft(latents[j], j, pano_latent, xy_map_stack)

                pano_latent = pano_latent + z_j
                pano_latent_count = pano_latent_count + mask_j
            
            z_t = pano_latent / (pano_latent_count + 1e-8)
            latents = [
                self.forward_ft(z_t, j, xy_map_stack)
                for j in range(self.num_views)
            ]

        for i in range(self.num_views):
            latents[i] = custom_decoder(latents[i], "post")

        # Post process
        for i in range(self.num_views):
            latents[i] = (latents[i] / 2 + 0.5).clamp(0, 1)

        return latents

class CustomDecoder(VAEDecoder):

    def __init__(self, vae_decoder):
        super().__init__()
        attributes = ["layers_per_block", "conv_in", "up_blocks", "mid_block", "conv_norm_out", "conv_act", "conv_out", "gradient_checkpointing"]
        for attr_name in attributes:
            setattr(self, attr_name, getattr(vae_decoder, attr_name))

    # modified forward of VAEDecoder::forward with partial steps (nograd only)
    def forward(
        self,
        sample: torch.Tensor,
        phase: str,
        step_up: int = -1,
        latent_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward method of the `Decoder` class."""

        # pre-process phase
        if phase == "pre":
            sample = self.conv_in(sample)
            upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)
        
        # upsample phase
        elif phase == "up":
            assert step_up >= 0 and step_up < len(self.up_blocks)
            sample = self.up_blocks[step_up](sample, latent_embeds)

        # post-process phase
        elif phase == "post":
            if latent_embeds is None:
                sample = self.conv_norm_out(sample)
            else:
                sample = self.conv_norm_out(sample, latent_embeds)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)
        else:
            raise NotImplementedError("forward phases are pre|up|post")

        return sample