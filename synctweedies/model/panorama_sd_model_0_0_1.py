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
from synctweedies.renderer.panorama.geometry import make_coord, gridy2x_erp2pers

class PanoramaSDModel_0_0_1(PanoramaSDModel_0_0_0):
    def __init__(self, config):
        super().__init__(config)
        self.up_level = 3
    
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

        for theta in self.theta_list:
            for phi in self.phi_list:

                # Warp ERP pixel grid into Pers. coordinate
                erp2pers_grid, valid_mask = gridy2x_erp2pers(gridy=erp_grid,
                    HWy=(H, W), HWx=hw, THETA=theta, PHI=phi, FOVy=360, FOVx=self.config.FOV)
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
    
    @torch.no_grad()
    def __call__(
        self,
        depth_data_path=None,
        prompt=None,
        negative_prompt=None,
    ):
        """
        Process reverse diffusion steps

        (v0.0.1) line 272-283
        - Conditional noise upsampling zT
        - Discrete warping xts from zT
        """

        if negative_prompt is None:
            negative_prompt = self.config.negative_prompt
            self.config.negative_prompt = negative_prompt

        prompt = f"Best quality, extremely detailed {self.config.prompt}"
        self.config.prompt = prompt
        

        log_opt = vars(self.config)
        config_path = os.path.join(self.output_dir, "pano_run_config.yaml")
        with open(config_path, "w") as f:
            json.dump(log_opt, f, indent=4)

        num_timesteps = self.model.scheduler.config.num_train_timesteps
        ref_attention_end = self.config.ref_attention_end
        multiview_diffusion_end = self.config.mvd_end
        callback = None

        callback_steps = 1

        if self.config.model == "controlnet":
            if depth_data_path is None:
                depth_data_path = self.config.depth_data_path
            conditioning_images = self.get_depth_conditioning_images(depth_data_path)

            control_guidance_start = self.config.control_guidance_start
            control_guidance_end = self.config.control_guidance_end

            controlnet = (
                self.model.controlnet._orig_mod
                if is_compiled_module(self.model.controlnet)
                else self.model.controlnet
            )
            if not isinstance(control_guidance_start, list) and isinstance(
                control_guidance_end, list
            ):
                control_guidance_start = len(control_guidance_end) * [
                    control_guidance_start
                ]
            elif not isinstance(control_guidance_end, list) and isinstance(
                control_guidance_start, list
            ):
                control_guidance_end = len(control_guidance_start) * [
                    control_guidance_end
                ]
            elif not isinstance(control_guidance_start, list) and not isinstance(
                control_guidance_end, list
            ):
                mult = 1
                control_guidance_start, control_guidance_end = mult * [
                    control_guidance_start
                ], mult * [control_guidance_end]
            controlnet_conditioning_scale = self.config.conditioning_scale

            controlnet_keep = []
            timesteps = self.model.scheduler.timesteps
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(
                    keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
                )

        else:
            controlnet_conditioning_scale = None
            conditioning_images = None
            controlnet_keep = None

        num_images_per_prompt = 1
        if prompt is not None and isinstance(prompt, list):
            assert (
                len(prompt) == 1 and len(negative_prompt) == 1
            ), "Only implemented for 1 (negative) prompt"
        assert num_images_per_prompt == 1, "Only implemented for 1 image per-prompt"
        batch_size = self.num_views

        device = self.device
        guidance_scale = self.config.guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0

        guess_mode = False
        cross_attention_kwargs = None

        prompt_embeds = self.model._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )
        prompt_embeds_buffer = prompt_embeds
        if do_classifier_free_guidance:
            negative_prompt_embeds, prompt_embeds = torch.chunk(prompt_embeds, 2)
        else:
            negative_prompt_embeds = None
                
        # 5. Prepare timesteps
        num_inference_steps = self.config.steps
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.model.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.model.unet.config.in_channels
        generator = torch.manual_seed(self.config.seed)

        case_name = METHOD_MAP[str(self.config.case_num)]
        instance_denoising_cases = list(INSTANCE_DENOISING_XT.values()) + list(JOINT_DENOISING_XT.values())
        canonical_denoising_cases = list(CANONICAL_DENOISING_ZT.values()) + list(JOINT_DENOISING_ZT.values())

        get_latent = self.model.prepare_latents
        canonical_latent_h_param = self.config.canonical_latent_h * 8
        canonical_latent_w_param = self.config.canonical_latent_w * 8
        
        func_params = {
            "guidance_scale": guidance_scale,
            "positive_prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "group_metas": self.group_metas,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "guess_mode": guess_mode,
            "ref_attention_end": ref_attention_end,
            "num_timesteps": num_timesteps,
            "cross_attention_kwargs": cross_attention_kwargs,
            "conditioning_images": conditioning_images,
            "controlnet_keep": controlnet_keep,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "generator": generator,
            "predicted_variance": None,
            "cos_weighted": True,
            "prompt_embeds": prompt_embeds_buffer, # TO SOLVE: noise_preds_stack = noise_pred_dict["uncond"] + guidance_scale * (noise_pred_dict["text"] - noise_pred_dict["uncond"]); KeyError: 'text'
        }
        
        if case_name in instance_denoising_cases:
            if self.config.initialize_xt_from_zt:
                print("[*] Although instance denoising, initialize xT from zT")
                # zT = get_latent(
                #     1,
                #     num_channels_latents,
                #     canonical_latent_h_param,
                #     canonical_latent_w_param,
                #     prompt_embeds.dtype,
                #     device,
                #     generator,
                # )
                
                # xts = self.forward_mapping(zT, 
                #                            xy_map_stack=self.latent_xy_map_stack,
                #                            encoding=False)
                # zts = None
                # del zT

                # version 0.0.1: Conditional noise upsampling & Discrete warping
                zT = get_latent(
                    1,
                    num_channels_latents,
                    canonical_latent_h_param // (2**self.up_level),
                    canonical_latent_w_param // (2**self.up_level),
                    prompt_embeds.dtype,
                    device,
                    generator,
                )
                zT = self.cond_noise_upsample(zT)

                xts, _ = self.discrete_warping(zT, (512//8, 512//8))
                xts = torch.cat(xts, dim=0)

                zts = None
                del zT

            else:
                xts = get_latent(
                    batch_size,
                    num_channels_latents,
                    self.config.instance_latent_size * 8,
                    self.config.instance_latent_size * 8,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    None,
                )  # (B, 4, 96, 96)
                zts = None
        else:
            zts = get_latent(
                1,
                num_channels_latents,
                self.config.canonical_latent_h * 8,
                self.config.canonical_latent_w * 8,
                prompt_embeds.dtype,
                device,
                generator,
            )
            xts = None

        input_params = {"zts": zts, "xts": xts}

        mapping_dict = {
            "encoding": self.config.average_rgb,
            "decoding": self.config.average_rgb,
        }
        if self.config.average_rgb:
            mapping_dict["xy_map_stack"] = self.rgb_xy_map_stack
            mapping_dict["mask_stack"] = self.rgb_mask_stack
        else:
            mapping_dict["xy_map_stack"] = self.latent_xy_map_stack
            mapping_dict["mask_stack"] = self.latent_mask_stack

        eta = 0.0
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.model.scheduler.order
        )
        
        func_params.update(mapping_dict)

        alphas = self.model.scheduler.alphas_cumprod ** (0.5)
        sigmas = (1 - self.model.scheduler.alphas_cumprod) ** (0.5)
        with self.model.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                func_params["cur_index"] = i
                # self.model.set_up_coefficients(t, self.config.sampling_method) # TO SOLVE: AttributeError: 'SyncTweediesSD' object has no attribute 'set_up_coefficients'

                if t > (1 - multiview_diffusion_end) * num_timesteps:
                    out_params = self.one_step_process(
                        input_params=input_params,
                        timestep=t,
                        alphas=alphas,
                        sigmas=sigmas,
                        case_name=case_name,
                        **func_params,
                    )

                    if case_name in instance_denoising_cases:
                        input_params["xts"] = out_params["x_t_1"]
                        input_params["zts"] = None
                        log_x_prevs = out_params["x_t_1"]
                        log_x0s = out_params["x0s"]
                    elif case_name in canonical_denoising_cases:
                        input_params["xts"] = None
                        input_params["zts"] = out_params["z_t_1"]
                        log_x_prevs = self.forward_mapping(
                            out_params["z_t_1"], **mapping_dict
                        )
                        log_x0s = self.forward_mapping(
                            out_params["z0s"], **mapping_dict
                        )
                else:
                    if case_name in instance_denoising_cases:
                        latents = out_params["x_t_1"]

                    elif case_name in canonical_denoising_cases:
                        if out_params.get("x_t_1") is None:
                            assert (
                                out_params.get("z_t_1") is not None
                            ), f"{out_params['z_t_1']}"
                            latents = self.forward_mapping(
                                out_params["z_t_1"], **mapping_dict
                            )
                        else:
                            latents = out_params["x_t_1"]

                    noise_pred = self.compute_noise_preds(
                        latents, t, **func_params
                    )
                    step_results = self.model.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=True
                    )

                    pred_original_sample = step_results["pred_original_sample"]
                    latents = step_results["prev_sample"]
                    out_params = dict()
                    out_params["x_t_1"] = latents
                    out_params["x0s"] = pred_original_sample

                    log_x_prevs = out_params["x_t_1"]
                    log_x0s = out_params["x0s"]

                if (i + 1) % self.config.log_interval == 0:
                    self.intermediate_dir = self.output_dir / f"intermediate/{i}"
                    self.intermediate_dir.mkdir(exist_ok=True, parents=True)

                    log_x_prev_img = self.instance_latents_to_pano_image(log_x_prevs)
                    log_x0_img = self.instance_latents_to_pano_image(log_x0s)

                    log_img = merge_images([log_x_prev_img, log_x0_img])
                    log_img.save(f"{self.intermediate_dir}/i={i}_t={t}.png")

                    for view_idx, log_x0 in enumerate(log_x0s[:10]):
                        decoded = self.decode_latents(log_x0.unsqueeze(0)).float()
                        TF.to_pil_image(decoded[0].cpu()).save(
                            f"{self.intermediate_dir}/i={i}_v={view_idx}_view.png"
                        )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.model.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

            if case_name in instance_denoising_cases:
                final_out = self.forward_mapping(
                    self.inverse_mapping(out_params["x_t_1"], **mapping_dict),
                    **mapping_dict,
                )
            elif case_name in canonical_denoising_cases:
                if multiview_diffusion_end == 1:
                    final_out = self.forward_mapping(
                        out_params["z_t_1"], **mapping_dict
                    )
                else:
                    final_out = self.forward_mapping(
                        self.inverse_mapping(out_params["x_t_1"], **mapping_dict),
                        **mapping_dict,
                    )

            self.result_dir = f"{self.output_dir}/results"
            os.makedirs(self.result_dir, exist_ok=True)
            
            final_img = self.instance_latents_to_pano_image(final_out)
            final_img.save(f"{self.result_dir}/final.png")

            equ = Equirectangular(f"{self.result_dir}/final.png")
            for i, theta in enumerate(np.random.randint(0, 360, 10)):
                pers_img = equ.GetPerspective(60, theta, 0, 512, 512)[..., [2, 1, 0]]
                pers_img = Image.fromarray(pers_img)
                pers_img.save(
                    f"{self.result_dir}/final_pers_sample_{i}_theta={theta}.png"
                )