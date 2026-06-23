import numpy as np
from tqdm import tqdm
import torch
import time
from collections import defaultdict
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from lvdm.common import noise_like
import os
import torchvision
import sys
from pathlib import Path
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch.nn.functional as F
import logging


class ProfilingTimer:
    """Lightweight GPU-aware profiling timer for denoising steps."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = defaultdict(list)
        self._start_events = {}

    def start(self, name):
        if not self.enabled:
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._start_events[name] = (start, end)

    def stop(self, name):
        if not self.enabled or name not in self._start_events:
            return
        start, end = self._start_events.pop(name)
        end.record()
        torch.cuda.synchronize()
        self.timings[name].append(start.elapsed_time(end))  # milliseconds

    def summary(self):
        lines = ["\n=== Denoising Profiling Summary ==="]
        for name, times in sorted(self.timings.items()):
            total = sum(times)
            avg = total / len(times)
            lines.append(f"  {name}: {total:.1f}ms total, {avg:.1f}ms avg, {len(times)} calls")
        lines.append("=" * 40)
        return "\n".join(lines)
logging.getLogger().setLevel(logging.ERROR)  # Only show ERROR messages
logging.disable(logging.INFO)
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)
import sys
import os

# Add the Grounded-SAM-2 directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Grounded-SAM-2'))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from rebuttals.masks_quality_test import create_eroded_mask, create_dilated_mask, create_noisy_mask
from PIL import Image, ImageDraw, ImageFont
import math
import torch.nn as nn
from .visualization import VisualizationHelper
import cv2
import matplotlib.pyplot as plt

class DDIMSampler(object):
    """
    Perform DDIM sampling using a diffusion model.
    """
    def __init__(self, model, schedule="linear", use_self_attention=False, experiment_condition=None, **kwargs):
        super().__init__()
        self.model = model # DDIM model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0
        self.use_self_attention = use_self_attention
        self.vis_helper = VisualizationHelper()

        # Initialize models only if needed
        self.sam2_model = None
        self.sam2_predictor = None
        self.processor = None
        self.grounding_model = None

        # Flag to control model initialization
        self.models_initialized = False

        # Profiling timer (set enabled=True to measure bottlenecks)
        self.profiler = ProfilingTimer(enabled=False)

        # Segmentation caching: reuse masks when IOU is high
        self._cached_text_inputs = None
        self._cached_text_target = None
        self._cached_mask = None
        self._mask_iou_skip_threshold = 0.85  # Skip segmentation if mask IOU > threshold

        # ConceptAttention-based masking (lightweight alternative to SAM2+GDINO)
        self._concept_attn_extractor = None
        self._concept_token_indices = None
        self._concept_attn_target = None
        self._concept_mask_strategy = "relative_threshold"
        self._concept_mask_threshold = 0.3
        self._concept_mask_topk_ratio = 0.2
        self._concept_mask_std_scale = 1.0
        self._concept_mask_max_components = 1

        # Initialize based on experiment condition
        if experiment_condition == "concept_attention":
            self.initialize_concept_attention()
        elif not use_self_attention:
            self.initialize_segmentation_models()
            

    def register_buffer(self, name, attr):
        """
        Register a buffer (tensor) to the class, ensuring its on the CUDA device if necessary.
        """
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        """
        Create the DDIM sampling schedule.
        ddim_num_steps: int, number of DDIM steps
        ddim_discretize: str, method to discretize the diffusion steps
        ddim_eta: float, noise scale factor
        verbose: bool, whether to print progress
        """
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
        self.use_scale = self.model.use_scale

        if self.use_scale:
            self.register_buffer('scale_arr', to_torch(self.model.scale_arr))
            ddim_scale_arr = self.scale_arr.cpu()[self.ddim_timesteps]
            self.register_buffer('ddim_scale_arr', ddim_scale_arr)
            ddim_scale_arr = np.asarray([self.scale_arr.cpu()[0]] + self.scale_arr.cpu()[self.ddim_timesteps[:-1]].tolist())
            self.register_buffer('ddim_scale_arr_prev', ddim_scale_arr)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S, # number of steps
               batch_size, # batch size
               shape, # shape of the data
               conditioning=None, # conditioning information for guided sampling
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               latents_dir=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        ## schedule creation
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=schedule_verbose)
        
        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W) # frames added
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        
        # Perform the actual sampling
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose,
                                                    latents_dir=latents_dir,
                                                    **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,
                      cond_tau=1., target_size=None, start_timesteps=None, latents_dir=None,
                      **kwargs):
        """
        cond: dict, conditioning information
        shape: tuple, shape of the generated data
        x_T: tensor, initial latent state
        ddim_use_original_steps: bool, whether to use the original DDPM steps
        other arguments: see sample method
        """
        device = self.model.betas.device        
        b = shape[0] # batch size
        if x_T is None:
            img = torch.randn(shape, device=device) # [1,4,16,40,64] -> [bath_size, C, T, H, W]
        else:
            img = x_T
        
        if timesteps is None: # True
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps: # enable customized sampling schedules
            # compare the ratio of the provided timesteps to the original timesteps, and ensure it not exceeds 1
            # which prevents selecting more timesteps than available
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}

        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        init_x0 = False

        for i, step in enumerate(iterator):
            if i == 0 and latents_dir is not None:
                torch.save(img, f"{latents_dir}/{i}.pt")
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long) # [1]
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      x0=x0,
                                      **kwargs)
            # the img is the intermediate state of the latent variable x at each timestep
            # pred_x0 is the model's prediction of the original data at each timestep
            img, pred_x0 = outs 
            
        if latents_dir is not None:
            torch.save(img, f"{latents_dir}/{total_steps}.pt")

        return img, intermediates

    @torch.no_grad()
    def fifo_onestep(self, cond, shape, latents=None, timesteps=None, indices=None,
                     unconditional_guidance_scale=1., unconditional_conditioning=None,
                     cond_image=None, target=None, use_self_attention=False,
                     davis_masks=None, experiment_condition="baseline",
                     cond_original=None, mixing_strength=0.3, **kwargs):
        device = self.model.betas.device
        b, _, f, _, _ = shape
        ts = torch.Tensor(timesteps.copy()).to(device=device, dtype=torch.long)

        # Enable attention map capture for concept_attention mode
        use_concept_attn = (experiment_condition == "concept_attention" and
                            self._concept_attn_extractor is not None)

        self.profiler.start("unet_forward")
        if use_concept_attn:
            with self._concept_attn_extractor.capture():
                noise_pred = self.unet(latents, cond, ts,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        **kwargs)
        else:
            noise_pred = self.unet(latents, cond, ts,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,
                                    **kwargs)
        self.profiler.stop("unet_forward")

        self.profiler.start("ddim_step")
        latents, pred_x0 = self.ddim_step(latents, noise_pred, indices, cond_image, target, ts,
                                        use_self_attention=use_self_attention,
                                        davis_masks=davis_masks,
                                        experiment_condition=experiment_condition)
        self.profiler.stop("ddim_step")

        return latents, pred_x0

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      uc_type=None, conditional_guidance_scale_temporal=None, **kwargs):
        """
        It is used for performing a single DDIM sampling step.
        x: current state of the latent variable x
        c: conditioning information
        t: timestep
        index: index of the timestep
        """
        b, *_, device = *x.shape, x.device
        if x.dim() == 5: #[batch_size, C, T, H, W]
            is_video = True
        else:
            is_video = False
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, **kwargs) # unet denoiser
        else:
            # with unconditional condition
            if isinstance(c, torch.Tensor):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            elif isinstance(c, dict):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            else:
                raise NotImplementedError
            # text cfg
            if uc_type is None:
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            else:
                if uc_type == 'cfg_original':
                    e_t = e_t + unconditional_guidance_scale * (e_t - e_t_uncond)
                elif uc_type == 'cfg_ours':
                    e_t = e_t + unconditional_guidance_scale * (e_t_uncond - e_t)
                else:
                    raise NotImplementedError
            # temporal guidance
            if conditional_guidance_scale_temporal is not None:
                e_t_temporal = self.model.apply_model(x, t, c, **kwargs)
                e_t_image = self.model.apply_model(x, t, c, no_temporal_attn=True, **kwargs)
                e_t = e_t + conditional_guidance_scale_temporal * (e_t_temporal - e_t_image)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)

        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        
        if self.use_scale:
            scale_arr = self.model.scale_arr if use_original_steps else self.ddim_scale_arr
            scale_t = torch.full(size, scale_arr[index], device=device)
            scale_arr_prev = self.model.scale_arr_prev if use_original_steps else self.ddim_scale_arr_prev
            scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
            pred_x0 /= scale_t 
            x_prev = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
        else:
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0
    
    @torch.no_grad()
    def unet(self, x, c, t, unconditional_guidance_scale=1.,
             unconditional_conditioning=None, **kwargs):

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, **kwargs) # unet denoiser
        else:
            # Batch conditioned and unconditioned into a single forward pass
            x_combined = torch.cat([x, x], dim=0)
            t_combined = torch.cat([t, t], dim=0)

            # Merge conditioning dicts along batch dimension
            if isinstance(c, dict):
                c_combined = {}
                for key in c:
                    if key == 'fps':
                        # fps is a 1D tensor [B] that broadcasts with emb — just keep as-is
                        # since both cond and uncond use the same fps value
                        c_combined[key] = c[key]
                    elif isinstance(c[key], list):
                        c_combined[key] = [torch.cat([c_val, uc_val], dim=0)
                                           for c_val, uc_val in zip(c[key], unconditional_conditioning[key])]
                    elif isinstance(c[key], torch.Tensor):
                        c_combined[key] = torch.cat([c[key], unconditional_conditioning[key]], dim=0)
                    else:
                        c_combined[key] = c[key]
            else:
                c_combined = [torch.cat([c_val, uc_val], dim=0)
                              for c_val, uc_val in zip(c, unconditional_conditioning)]

            e_t_combined = self.model.apply_model(x_combined, t_combined, c_combined, **kwargs)
            e_t, e_t_uncond = e_t_combined.chunk(2, dim=0)

            # text cfg
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        return e_t

    @torch.no_grad()
    def ddim_step(self, sample, noise_pred, indices, cond_image, target, ts, gamma=0.5, use_self_attention=False, davis_masks=None, experiment_condition="baseline"):
        """Modified DDIM step to support both attention mechanisms and DAVIS masks"""
        b, _, f, *_, device = *sample.shape, sample.device

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        
        size = (b, 1, 1, 1, 1)
        
        x_prevs = []
        pred_x0s = []

        pre_masks = None
        prev_frame = None

        # Initialize momentum if not already done
        if not hasattr(self, 'momentum'):
            self.momentum = torch.zeros_like(sample)
            self.beta = 0.9  # Momentum decay rate

        # Create visualization directory if it doesn't exist
        vis_dir = "visualizations/denoising"
        os.makedirs(vis_dir, exist_ok=True)
        cond_dir = "visualizations/conditioning"
        os.makedirs(cond_dir, exist_ok=True)

        for i, index in enumerate(indices):
            x = sample[:, :, [i]]
            e_t = noise_pred[:, :, [i]]
            timestep = ts[i]
            a_t = torch.full(size, alphas[index], device=device)
            a_prev = torch.full(size, alphas_prev[index], device=device)
            sigma_t = torch.full(size, sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)

            # Current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # Direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
             # x_prev uses momentum-corrected pred_x0 (clean denoising trajectory)
            
            noise = sigma_t * noise_like(x.shape, device)
            

            # Calculate motion gradient if we have a previous frame
            if prev_frame is not None:
                motion_gradient = pred_x0 - prev_frame
                motion_gradient = motion_gradient + 0.05 * dir_xt
                mg = motion_gradient
                if mg.dim() == 4:
                    mg = mg.unsqueeze(2)
                self.momentum[:, :, [i]] = (
                    self.beta * self.momentum[:, :, [i-1]] +
                    (1 - self.beta) * mg
                )
                correction_strength = 0.1 * (1.0 - timestep / 1000.0)
                pred_x0 = pred_x0 + correction_strength * self.momentum[:, :, [i]]
                
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

           

            # Apply conditioning AFTER x_prev — injection only affects prev_frame
            # so it propagates through momentum into future frames
            if timestep <= 300:
                if davis_masks is not None and davis_masks.shape[2] > i:
                    mask = davis_masks[:, :, i, :, :]
                    mask = mask.unsqueeze(0)
                    mask = mask.expand(-1, pred_x0.shape[1], -1, -1, -1)

                    cond_img_local = cond_image
                    if cond_img_local is None:
                        cond_img_local = torch.zeros_like(pred_x0)
                    else:
                        if cond_img_local.shape[1] != pred_x0.shape[1] and cond_img_local.shape[1] == 3:
                            alpha_channel = torch.ones_like(cond_img_local[:, :1])
                            cond_img_local = torch.cat([cond_img_local, alpha_channel], dim=1)
                        if cond_img_local.dim() == 4 and pred_x0.dim() == 5:
                            cond_img_local = cond_img_local.unsqueeze(2)

                    if mask.sum() != 0:
                        injection_strength = gamma
                        mask_float = (mask.to(pred_x0.device) > 0.5).float()
                        blended = (1 - injection_strength) * pred_x0 + injection_strength * cond_img_local
                        pred_x0 = mask_float * blended + (1 - mask_float) * pred_x0
                else:
                    pred_x0, attention = self.apply_cond_img(
                        pred_x0,
                        cond_image,
                        target,
                        i,
                        pre_masks if not use_self_attention else getattr(self, 'previous_attention', None),
                        experiment_condition,
                        use_self_attention=use_self_attention,
                    )

                    if use_self_attention:
                        self.previous_attention = attention
                    else:
                        pre_masks = attention

            # Save injected pred_x0 as prev_frame → carries injection into momentum
            if pred_x0.dim() == 4:
                pred_x0 = pred_x0.unsqueeze(2)
            elif pred_x0.dim() == 5 and pred_x0.shape[2] != 1:
                pred_x0 = pred_x0[:, :, [0]]
            prev_frame = pred_x0.detach()

            x_prevs.append(x_prev)
            pred_x0s.append(pred_x0)

        x_prev = torch.cat(x_prevs, dim=2)
        pred_x0 = torch.cat(pred_x0s, dim=2)

        return x_prev, pred_x0

    @torch.no_grad()
    def ddim_step_composable(self, sample, noise_pred_orig, noise_pred_cond, indices, ts,
                              davis_masks=None, experiment_condition="baseline",
                              target=None, mixing_strength=0.3):
        """
        Composable diffusion: compute x_prev from both original and conditioned noise predictions,
        then blend in the masked region. This produces clean semantic mixing.

        Args:
            noise_pred_orig: noise prediction from original prompt only
            noise_pred_cond: noise prediction from original + conditioned prompt
            mixing_strength: 0.0 = pure original, 1.0 = pure conditioned (0.3 = sweet spot)
        """
        b, _, f, *_, device = *sample.shape, sample.device

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        size = (b, 1, 1, 1, 1)

        x_prevs = []
        pred_x0s = []

        for i, index in enumerate(indices):
            x = sample[:, :, [i]]
            e_t_orig = noise_pred_orig[:, :, [i]]
            e_t_cond = noise_pred_cond[:, :, [i]]
            timestep = ts[i]
            a_t = torch.full(size, alphas[index], device=device)
            a_prev = torch.full(size, alphas_prev[index], device=device)
            sigma_t = torch.full(size, sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index], device=device)

            # Compute x_prev from ORIGINAL prompt (clean scene)
            pred_x0_orig = (x - sqrt_one_minus_at * e_t_orig) / a_t.sqrt()
            dir_xt_orig = (1. - a_prev - sigma_t**2).sqrt() * e_t_orig
            noise = sigma_t * noise_like(x.shape, device)
            x_prev_orig = a_prev.sqrt() * pred_x0_orig + dir_xt_orig + noise

            # Compute x_prev from CONDITIONED prompt (concept-mixed scene)
            pred_x0_cond = (x - sqrt_one_minus_at * e_t_cond) / a_t.sqrt()
            dir_xt_cond = (1. - a_prev - sigma_t**2).sqrt() * e_t_cond
            x_prev_cond = a_prev.sqrt() * pred_x0_cond + dir_xt_cond + noise  # Same noise for consistency

            # Get mask for this frame
            mask_frame = None
            if davis_masks is not None and davis_masks.shape[2] > i:
                mask_frame = davis_masks[:, :, i, :, :].unsqueeze(0)  # [1, 1, 1, H, W]
            elif experiment_condition == "concept_attention" and self._concept_attn_extractor is not None:
                token_indices = self._find_target_token_indices(target)
                h, w = x.shape[3], x.shape[4]
                concept_mask = self._get_concept_mask(token_indices, h, w)
                if concept_mask is not None:
                    mask_frame = concept_mask.to(device).unsqueeze(2)

            # Blend x_prev: original outside mask, mix of original+conditioned inside mask
            if mask_frame is not None:
                mask_values = mask_frame.to(device).float().clamp(0.0, 1.0)
                if self._concept_mask_strategy != "soft":
                    mask_values = (mask_values > 0.5).float()
                mask_float = mask_values.expand_as(x_prev_orig)
                x_prev = (1 - mask_float) * x_prev_orig + mask_float * (
                    (1 - mixing_strength) * x_prev_orig + mixing_strength * x_prev_cond
                )
                pred_x0 = (1 - mask_float) * pred_x0_orig + mask_float * (
                    (1 - mixing_strength) * pred_x0_orig + mixing_strength * pred_x0_cond
                )
            else:
                # No mask available — use global blend (weaker)
                x_prev = (1 - mixing_strength) * x_prev_orig + mixing_strength * x_prev_cond
                pred_x0 = (1 - mixing_strength) * pred_x0_orig + mixing_strength * pred_x0_cond

            x_prevs.append(x_prev)
            pred_x0s.append(pred_x0)

        x_prev = torch.cat(x_prevs, dim=2)
        pred_x0 = torch.cat(pred_x0s, dim=2)

        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)

        def extract_into_tensor(a, t, x_shape):
            b, *_ = t.shape
            out = a.gather(-1, t)
            return out.reshape(b, *((1,) * (len(x_shape) - 1)))

        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec

    def visualize_sampling(self, pred_x0, noise, save_dir, step, is_manipulated=False):
        """Visualize the sampling process"""
        self.vis_helper.visualize_sampling(pred_x0, noise, save_dir, step, is_manipulated)

    def visualize_object_attention(self, pred_image, cond_image, attention_mask, attention_map, 
                                 labeled_regions, target_object, save_dir, step):
        """Visualize attention and region detection"""
        self.vis_helper.visualize_object_attention(
            pred_image, cond_image, attention_mask, attention_map,
            labeled_regions, target_object, save_dir, step
        )
    def visualize_mask_and_latent(self, mask, latent, timestep, frame_idx, save_dir):
        """Visualize the mask and latent during denoising process"""
        self.vis_helper.visualize_mask_and_latent(mask, latent, timestep, frame_idx, save_dir)

    def visualize_masks(self, masks, save_dir, step):
        """Visualize the segmentation masks"""
        self.vis_helper.visualize_masks(masks, save_dir, step)

    def setup_grounded_sam_paths(self):
        """Setup paths for Grounded SAM2 modules"""
        grounded_sam_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Grounded-SAM-2')

        if not os.path.exists(grounded_sam_path):
            raise RuntimeError(f"Grounded-SAM-2 directory not found at {grounded_sam_path}")

        # Add to Python path
        if str(grounded_sam_path) not in sys.path:
            sys.path.append(str(grounded_sam_path))
            
        return grounded_sam_path
    def initialize_concept_attention(self):
        """Initialize the ConceptAttention-based mask extractor (lightweight alternative to SAM2+GDINO)."""
        from lvdm.modules.attention import CrossAttentionMapExtractor
        if self._concept_attn_extractor is None:
            self._concept_attn_extractor = CrossAttentionMapExtractor()
            # No hooks needed — recording is injected directly into CrossAttention.efficient_forward
            self._concept_attn_extractor.register(self.model.model.diffusion_model)

    def set_prompt_for_concept_attention(self, prompt):
        """Store the full prompt text so we can find target token positions within it."""
        self._full_prompt = prompt

    def set_concept_mask_strategy(self, strategy="relative_threshold", threshold=0.3,
                                  topk_ratio=0.2, std_scale=1.0, max_components=1):
        """Configure how concept attention maps are converted into masks."""
        self._concept_mask_strategy = strategy
        self._concept_mask_threshold = threshold
        self._concept_mask_topk_ratio = topk_ratio
        self._concept_mask_std_scale = std_scale
        self._concept_mask_max_components = max_components

    def _get_concept_mask(self, token_indices, h, w):
        return self._concept_attn_extractor.get_concept_mask(
            token_indices, h, w,
            threshold=self._concept_mask_threshold,
            strategy=self._concept_mask_strategy,
            topk_ratio=self._concept_mask_topk_ratio,
            std_scale=self._concept_mask_std_scale,
            max_components=self._concept_mask_max_components,
        )

    def _find_target_token_indices(self, target, prompt_embeds=None):
        """Find token indices for the target concept word within the full prompt's token sequence."""
        if self._concept_attn_target == target and self._concept_token_indices is not None:
            return self._concept_token_indices

        import open_clip
        target_clean = target.rstrip(".")
        bos_id, eos_id = 49406, 49407

        # Tokenize the target word alone to get its token IDs
        target_tokens = open_clip.tokenize([target_clean])[0]
        target_ids = []
        for tid in target_tokens.tolist():
            if tid == bos_id or tid == eos_id:
                continue
            if tid == 0:  # padding
                break
            target_ids.append(tid)

        # Tokenize the full prompt to find where target_ids appear
        full_prompt = getattr(self, '_full_prompt', target_clean)
        prompt_tokens = open_clip.tokenize([full_prompt])[0]
        prompt_ids = prompt_tokens.tolist()

        # Search for target_ids subsequence within prompt_ids
        indices = []
        for start in range(len(prompt_ids) - len(target_ids) + 1):
            if prompt_ids[start:start + len(target_ids)] == target_ids:
                indices = list(range(start, start + len(target_ids)))
                break

        # Fallback: if exact match not found, search for individual token matches
        if not indices:
            for i, pid in enumerate(prompt_ids):
                if pid in target_ids and pid != bos_id and pid != eos_id:
                    indices.append(i)
            if not indices:
                # Last resort: assume position 1 onwards
                indices = list(range(1, 1 + len(target_ids)))

        self._concept_token_indices = indices
        self._concept_attn_target = target
        return indices

    def _apply_concept_attention(self, pred_x0, cond_image, target, step, pre_masks, blend_alpha=1.0):
        """
        Apply conditioning using cross-attention maps from the UNet (ConceptAttention).
        Much faster than SAM2+GDINO — uses attention maps already computed during denoising.
        """
        token_indices = self._find_target_token_indices(target)

        # pred_x0 may be [1, C, H, W] or [1, C, 1, H, W]
        if pred_x0.dim() == 5:
            h, w = pred_x0.shape[3], pred_x0.shape[4]
        else:
            h, w = pred_x0.shape[2], pred_x0.shape[3]

        mask = self._get_concept_mask(token_indices, h, w)

        if mask is None:
            return pred_x0, pre_masks

        # Visualize concept attention mask (save first 10 steps to avoid I/O overload)
        if step < 10:
            vis_dir = "visualizations/concept_attention_masks"
            os.makedirs(vis_dir, exist_ok=True)
            mask_vis = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_vis, mode='L').save(f"{vis_dir}/mask_frame_{step}.png")

        mask_2d = mask.squeeze(0).squeeze(0).to(pred_x0.device)  # (H, W)
        masks = mask_2d.unsqueeze(0)  # (1, H, W)
        return self._apply_mask_to_pred(pred_x0, masks, cond_image, step, blend_alpha=blend_alpha), pre_masks

    def apply_cond_img(self, pred_x0, cond_image, target, step, pre_masks, experiment_condition="baseline", use_self_attention=False, blend_alpha=1.0):
        """
        Apply conditioning image using either segmentation, concept attention, or self-attention.
        Args:
            pred_x0: predicted image
            cond_image: conditioning image
            target: text prompt for segmentation
            step: current step
            pre_masks: previous masks for temporal consistency
            experiment_condition: "segmentation", "concept_attention", "bounding_box", or "baseline"
            use_self_attention: whether to use self-attention instead of segmentation
            blend_alpha: blending strength (0=no conditioning, 1=full conditioning)
        """
        if experiment_condition == "concept_attention":
            return self._apply_concept_attention(pred_x0, cond_image, target, step, pre_masks, blend_alpha=blend_alpha)
        return self._apply_segmentation(pred_x0, cond_image, target, step, pre_masks, experiment_condition, blend_alpha=blend_alpha)


    def _apply_segmentation(self, pred_x0, cond_image, target, step, pre_masks, experiment_condition="baseline", blend_alpha=1.0):
        """Original segmentation-based approach with caching optimizations"""
        self.profiler.start("segmentation")
        original_masks = pre_masks
        if not target.endswith("."):
            target = target + "."

        # --- If we have a cached mask and previous masks, check IOU to skip segmentation ---
        if self._cached_mask is not None and pre_masks is not None:
            iou = self.calculate_iou(self._cached_mask, pre_masks)
            if isinstance(iou, (float, int)) and iou > self._mask_iou_skip_threshold:
                # Mask is stable — reuse cached mask, skip expensive segmentation
                masks = self._cached_mask if isinstance(self._cached_mask, torch.Tensor) else torch.from_numpy(self._cached_mask).float()
                self.profiler.stop("segmentation")
                return self._apply_mask_to_pred(pred_x0, masks, cond_image, step, blend_alpha=blend_alpha), original_masks

        # Convert tensor to PIL Image if needed
        if isinstance(pred_x0, torch.Tensor):
            image_np = pred_x0.cpu().numpy()
        if len(image_np.shape) == 5:
            image_np = image_np.squeeze(2).squeeze(0)

        frame = np.transpose(image_np, (1, 2, 0))

        if frame.shape[-1] != 3:
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            else:
                frame = frame[:, :, :3]

        # Scale to [0, 255] if in [0, 1]
        if np.floor(frame.max()) <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

        frame_pil = Image.fromarray(frame)

        # SAM2 image encoding (must run per-frame as image changes)
        self.sam2_predictor.set_image(np.array(frame_pil.convert("RGB")))

        # Cache text tokenization — only reprocess if target changed
        if self._cached_text_target != target:
            self._cached_text_inputs = self.processor(images=frame_pil, text=target, return_tensors="pt")
            self._cached_text_inputs = {k: (v.to("cuda", dtype=torch.float16) if v.dtype in [torch.float32, torch.float64] else
                        v.to("cuda", dtype=torch.long) if v.dtype in [torch.int32, torch.int64] else
                        v.to("cuda"))
                    for k, v in self._cached_text_inputs.items()
                    if isinstance(v, torch.Tensor)}
            self._cached_text_target = target
        else:
            # Re-run processor for new image but reuse text tokens where possible
            inputs = self.processor(images=frame_pil, text=target, return_tensors="pt")
            self._cached_text_inputs = {k: (v.to("cuda", dtype=torch.float16) if v.dtype in [torch.float32, torch.float64] else
                        v.to("cuda", dtype=torch.long) if v.dtype in [torch.int32, torch.int64] else
                        v.to("cuda"))
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)}

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = self.grounding_model(**self._cached_text_inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            self._cached_text_inputs['input_ids'],
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[frame_pil.size[::-1]]
        )

        input_boxes = results[0]["boxes"].cpu().numpy()
        if input_boxes.shape[0] == 0:
            ## Use the previous masks or cached mask
            if self._cached_mask is not None:
                masks = self._cached_mask if isinstance(self._cached_mask, torch.Tensor) else torch.from_numpy(self._cached_mask).float()
                self.profiler.stop("segmentation")
                return self._apply_mask_to_pred(pred_x0, masks, cond_image, step, blend_alpha=blend_alpha), original_masks
            if pre_masks is None:
                self.profiler.stop("segmentation")
                return pred_x0, None
        else:
            if experiment_condition == "bounding_box":
                masks = input_boxes
                if isinstance(input_boxes, torch.Tensor):
                    input_boxes_np = input_boxes.cpu().numpy()
                else:
                    input_boxes_np = np.asarray(input_boxes)
            else:
                # Get masks from SAM2
                masks, _, _ = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

            # Use IOU to decide whether to keep new mask or revert to previous
            if pre_masks is not None:
                iou = self.calculate_iou(masks, pre_masks)

            # Convert masks to tensor if they're numpy arrays
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks).float()

            # Cache the computed mask for future skip checks
            self._cached_mask = masks

        self.profiler.stop("segmentation")
        return self._apply_mask_to_pred(pred_x0, masks, cond_image, step, blend_alpha=blend_alpha), original_masks

    def _apply_mask_to_pred(self, pred_x0, masks, cond_image, step, blend_alpha=1.0):
        """Apply mask-based conditioning to pred_x0 via soft blending."""
        modified_pred_x0 = pred_x0.clone()

        for mask in masks:
            # if the mask majorly covers the image, skip
            if mask.sum() > 0.8 * mask.numel():
                continue
            # Expand mask to match channels
            try:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                if len(mask.shape) == 4:
                    mask = mask.expand(-1, pred_x0.shape[1], -1, -1)  # [1,C,H,W]
                else:
                    mask = mask.squeeze(0).expand(-1, pred_x0.shape[1], -1, -1)  # [1,C,H,W]
            except Exception as e:
                breakpoint()

            if cond_image is None:
                cond_image = torch.zeros_like(pred_x0[:, :, 0])
            elif cond_image.shape[1] != pred_x0.shape[1]:
                if cond_image.shape[1] == 3:
                    alpha_channel = torch.ones_like(cond_image[:, :1, :, :])
                    cond_image = torch.cat([cond_image, alpha_channel], dim=1)
                else:
                    raise ValueError(f"Conditional image must have 3 or 4 channels, got {cond_image.shape[1]}")

            # Blend cond_image into pred_x0 within the masked region. Hard mask
            # strategies are binary; the soft strategy keeps attention weights.
            mask_float = mask.to(pred_x0.device).float().clamp(0.0, 1.0)
            if self._concept_mask_strategy != "soft":
                mask_float = (mask_float > 0.5).float()
            blended = (1 - blend_alpha) * modified_pred_x0 + blend_alpha * cond_image
            modified_pred_x0 = mask_float * blended + (1 - mask_float) * modified_pred_x0
            # Save mask for x_prev regional nudge
            self._last_mask = mask_float[:, :1]  # [1, 1, H, W]

        return modified_pred_x0
    
    def calculate_iou(self, masks1, masks2):
        """Calculate Intersection over Union (IoU) between two sets of masks.
        
        Args:
            masks1 (numpy.ndarray or torch.Tensor): First set of masks [N,H,W]
            masks2 (numpy.ndarray or torch.Tensor): Second set of masks [N,H,W]
            
        Returns:
            float: Average IoU score across all mask pairs
        """
        # Convert to torch tensors if needed
        if isinstance(masks1, np.ndarray):
            masks1 = torch.from_numpy(masks1)
        if isinstance(masks2, np.ndarray):
            masks2 = torch.from_numpy(masks2)
        
        # Ensure masks are binary
        masks1 = masks1 > 0.5
        masks2 = masks2 > 0.5
        
        # Calculate IoU for each pair of masks
        ious = []
        for mask1, mask2 in zip(masks1, masks2):
            mask1 = mask1.to(masks2.device)
            intersection = torch.logical_and(mask1, mask2).sum().float()
            union = torch.logical_or(mask1, mask2).sum().float()
            
            # Handle edge case where union is 0
            if union == 0:
                if intersection == 0:  # Both masks are empty
                    iou = 1.0
                else:  # This shouldn't happen mathematically
                    iou = 0.0
            else:
                iou = intersection / union
            ious.append(iou)
        
        # Return average IoU
        return torch.tensor(ious).mean().item()
            
    def initialize_segmentation_models(self):
        """Initialize SAM2 and Grounding DINO for segmentation-based approach"""
        if self.models_initialized:
            return
            
        # Initialize SAM2 and Grounding DINO
        grounded_sam_path = self.setup_grounded_sam_paths()
        sam2_checkpoint = os.path.join(grounded_sam_path, 'checkpoints', 'sam2.1_hiera_large.pt')
        
        if not os.path.exists(sam2_checkpoint):
            raise RuntimeError(f"SAM2 checkpoint not found at {sam2_checkpoint}")
        
        # Initialize SAM2
        self.sam2_model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', sam2_checkpoint, device="cuda")
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Initialize Grounding DINO
        grounding_model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model_id,
            torch_dtype=torch.float16
        ).to("cuda").half()
        
        self.models_initialized = True
            
    @torch.no_grad()
    def ddim_inversion(self, frames, num_inference_steps, eta=1.0, latents_dir=None):
        """
        Perform DDIM inversion on input frames to obtain initial latents.
        
        Args:
            frames: Input frames tensor [B, C, T, H, W]
            num_inference_steps: Number of inference steps
            eta: DDIM eta parameter
            latents_dir: Optional directory to save intermediate latents
            
        Returns:
            latents: Inverted latents
        """
        # Ensure frames have correct dimensions
        if frames.dim() != 5:
            raise ValueError(f"Expected frames to have 5 dimensions [B, C, T, H, W], got {frames.dim()}")
        
        # Convert RGBA to RGB if needed
        if frames.shape[1] == 4:  # RGBA
            frames = frames[:, :3]  # Keep only RGB channels
        
        # Encode frames to latents
        latents = self.model.encode_first_stage_2DAE(frames)  # [B, C, 16, H, W]
        
        # Ensure latents have correct dimensions [B, C, T, H, W]
        if latents.dim() != 5:
            raise ValueError(f"Expected latents to have 5 dimensions [B, C, T, H, W], got {latents.dim()}")
        
        # Ensure latents have 4 channels
        if latents.shape[1] == 3:  # If encoder output has 3 channels
            zeros = torch.zeros_like(latents[:, :1])  # [B, 1, T, H, W]
            latents = torch.cat([latents, zeros], dim=1)  # [B, 4, T, H, W]
        
        # Initialize latents list
        latents_list = []
        
        # Main DDIM inversion loop
        for i in range(num_inference_steps):
            alpha = self.ddim_alphas[i]
            beta = 1 - alpha
            
            # Calculate frame index with proper offset
            frame_idx = max(0, i - (num_inference_steps - frames.shape[2]))
            
            # Get current frame's latents
            current_latents = latents[:,:,[frame_idx]]
            
            # Add noise with proper scaling
            noise = torch.randn_like(current_latents)
            new_latents = alpha**(0.5) * current_latents + beta**(0.5) * noise
            
            # Store intermediate results if needed
            if latents_dir is not None:
                torch.save(new_latents, f"{latents_dir}/step_{i}.pt")
            
            latents_list.append(new_latents)
        
        # Concatenate all latents
        latents = torch.cat(latents_list, dim=2)
        
        return latents
            