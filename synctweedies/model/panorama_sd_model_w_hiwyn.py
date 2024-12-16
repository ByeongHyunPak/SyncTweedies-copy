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


class PanoramaSDModelwithHIWYN(PanoramaSDModel):
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