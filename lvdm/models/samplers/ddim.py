import numpy as np
from tqdm import tqdm
import torch
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from lvdm.common import noise_like
import os
import torchvision
import sys
from pathlib import Path
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch.nn.functional as F
import logging
logging.getLogger().setLevel(logging.ERROR)  # Only show ERROR messages
logging.disable(logging.INFO)
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import math
import torch.nn as nn
from .visualization import VisualizationHelper

class DDIMSampler(object):
    """
    Perform DDIM sampling using a diffusion model.
    """
    def __init__(self, model, schedule="linear", use_self_attention=False, **kwargs):
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
        
        # Initialize models based on self-attention flag
        if not use_self_attention:
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
            
        # Store the intermediate results during sampling
        # x_inter: it stores the intermediate states of the latent variable x at each timestep during the reverse diffusion process
        # pred_x0: it stores the model's prediction of the original data at each timestep, which is an estimate of the clean data corresponding to the latent variable x
        intermediates = {'x_inter': [img], 'pred_x0': [img]}

        # reversed(range(0, timesteps)): 9, 8, 7, ..., 0
        # np.flip is used when timesteps is an array specifying the exact timesteps to sample
        # if timesteps  = np.array([0, 2, 4, 6, 8, 10]), the reversed order using np.flip is 10, 8, 6, 4, 2, 0
        # when timesteps is an array, which means the model will perform denoising only at steps 0, 2, 4, 6, 8, 10
        # there are the points in the reverse diffusion process where the model will generate intermediate states leading to the final sample
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
                     davis_masks=None, no_sam=False, **kwargs):
        device = self.model.betas.device        
        b, _, f, _, _ = shape
        ts = torch.Tensor(timesteps.copy()).to(device=device, dtype=torch.long) # [16]
        noise_pred = self.unet(latents, cond, ts,
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                unconditional_conditioning=unconditional_conditioning,
                                **kwargs) # torch.Size([1, 4, 16, 40, 64])
        
        latents, pred_x0 = self.ddim_step(latents, noise_pred, indices, cond_image, target, ts, 
                                        use_self_attention=use_self_attention,
                                        davis_masks=davis_masks, no_sam=no_sam)

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
            e_t = self.model.apply_model(x, t, c, **kwargs)
            e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            
            # text cfg
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        return e_t

    @torch.no_grad()
    def ddim_step(self, sample, noise_pred, indices, cond_image, target, ts, gamma=0.5, use_self_attention=False, davis_masks=None, no_sam=False):
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


        # Initialize momentum if not already done
        if not hasattr(self, 'momentum'):
            self.momentum = torch.zeros_like(sample)
            self.beta = 0.9  # Momentum decay rate
            
        # Store previous frame for gradient calculation
        prev_frame = None

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
            
            # Calculate motion gradient if we have a previous frame
            if prev_frame is not None:
                motion_gradient = pred_x0 - prev_frame
                motion_gradient = motion_gradient + 0.05 * dir_xt
                self.momentum[:, :, [i]] = (
                    self.beta * self.momentum[:, :, [i-1]] + 
                    (1 - self.beta) * motion_gradient
                )
                correction_strength = 0.1 * (1.0 - timestep / 1000.0)
                pred_x0 = pred_x0 + correction_strength * self.momentum[:, :, [i]]
            
            prev_frame = pred_x0.detach()
            
            noise = sigma_t * noise_like(x.shape, device)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            
            # Apply conditioning using either DAVIS masks or attention/segmentation
            if timestep <= 300:
                if davis_masks is not None:
                    # Use DAVIS mask directly
                    mask = davis_masks[:, :, i, :, :]  # [H, W]
                    mask = mask.unsqueeze(0)  # [1, 1, H, W]
                    # Check if mask has any values of 1
                    print(f"Mask shape: {mask.shape}")
                    print(f"Mask min: {mask.min()}, max: {mask.max()}")
                    print(f"Number of 1s in mask: {(mask > 0.5).sum().item()}")
                  
                    mask = mask.expand(-1, pred_x0.shape[1], -1, -1, -1)  # [1, C, 1, 32, 32]
                    
                    # Apply the cond_image to the masked pred_x0 region
                    if cond_image is None:
                        cond_image = torch.zeros_like(pred_x0[:, :, 0])
                    elif cond_image.shape[1] != pred_x0.shape[1]:
                        if cond_image.shape[1] == 3:
                            alpha_channel = torch.ones_like(cond_image[:, :1, :, :])
                            cond_image = torch.cat([cond_image, alpha_channel], dim=1)
                        else:
                            raise ValueError(f"Conditional image must have 3 or 4 channels, got {cond_image.shape[1]}")
                    
                    # Apply enhancement factor
                    enhancement_factor = 1
                    
                    # Apply the mask with the properly sized conditioning image
                    if mask.sum() != 0:
                        pred_x0 = torch.where(
                            mask.to(pred_x0.device) > 0.5,
                            cond_image * enhancement_factor,
                            pred_x0
                        )   
                else:
                    # Use original attention/segmentation approach
                    print("Using sam approach")
                    pred_x0, attention = self.apply_cond_img(
                        pred_x0, 
                        cond_image, 
                        target, 
                        i, 
                        pre_masks if not use_self_attention else getattr(self, 'previous_attention', None),
                        use_self_attention=use_self_attention
                    )
                    
                    if use_self_attention:
                        self.previous_attention = attention
                    else:
                        pre_masks = attention
                    
                pred_x0 = (1-gamma) * pred_x0 + gamma * noise

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
        grounded_sam_path = "/ibex/user/zhant0g/code/VidStoryCraft/Grounded-SAM-2"

        if not os.path.exists(grounded_sam_path):
            raise RuntimeError(f"Grounded-SAM-2 directory not found at {grounded_sam_path}")

        # Add to Python path
        if str(grounded_sam_path) not in sys.path:
            sys.path.append(str(grounded_sam_path))
            
        return grounded_sam_path
    def apply_cond_img(self, pred_x0, cond_image, target, step, pre_masks, use_self_attention=False):
        """
        Apply conditioning image using either segmentation or self-attention
        Args:
            pred_x0: predicted image
            cond_image: conditioning image
            target: text prompt for segmentation
            step: current step
            pre_masks: previous masks for temporal consistency
            use_self_attention: whether to use self-attention instead of segmentation
        """
        return self._apply_segmentation(pred_x0, cond_image, target, step, pre_masks)


    def _apply_segmentation(self, pred_x0, cond_image, target, step, pre_masks):
        """Original segmentation-based approach"""
        if not target.endswith("."):
            target = target + "."
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
        
        # Rest of original segmentation code...
        self.sam2_predictor.set_image(np.array(frame_pil.convert("RGB")))
        
        inputs = self.processor(images=frame_pil, text=target, return_tensors="pt")
        inputs = {k: (v.to("cuda", dtype=torch.float16) if v.dtype in [torch.float32, torch.float64] else 
                    v.to("cuda", dtype=torch.long) if v.dtype in [torch.int32, torch.int64] else 
                    v.to("cuda"))
                for k, v in inputs.items() 
                if isinstance(v, torch.Tensor)}
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs['input_ids'],
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[frame_pil.size[::-1]]
        )
        
        input_boxes = results[0]["boxes"].cpu().numpy()
        if input_boxes.shape[0] == 0:
            ## Use the previous masks
            if pre_masks is None:
                return pred_x0, None
            else:
                masks = pre_masks
        else:
            # Get masks from SAM2
            masks, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            ## Add that if the new generated mask is too deviated from the previous mask, use the previous mask using IOU 
            if pre_masks is not None:
                iou = self.calculate_iou(masks, pre_masks)
                if iou < 0.5:
                    masks = pre_masks

            # Convert masks to tensor if they're numpy arrays
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks).float()  # Ensure float first
                
            
        # Create a copy of pred_x0 to modify
        modified_pred_x0 = pred_x0.clone()

        # For each mask in the batch
        for mask in masks:
            # if the mask majorly covers the image, use the original image
            if mask.sum() > 0.8 * mask.numel():
                modified_pred_x0 = pred_x0
                continue
            # Expand mask to match channels
            try:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                if len(mask.shape) == 4:
                    mask = mask.expand(-1, pred_x0.shape[1], -1, -1)  # [1,C,H,W]
                else:
                    print(mask.shape)
                    mask = mask.squeeze(0).expand(-1, pred_x0.shape[1], -1, -1)  # [1,C,H,W]
            except Exception as e:
                breakpoint()

            ## Apply the cond_image to the masked pred_x0 region
            if cond_image is None:
                # Create a black conditional image matching pred_x0's dimensions
                cond_image = torch.zeros_like(pred_x0[:, :, 0])  # Take first frame's dimensions
            elif cond_image.shape[1] != pred_x0.shape[1]:
                if cond_image.shape[1] == 3:
                    # Add Alpha Channel
                    alpha_channel = torch.ones_like(cond_image[:, :1, :, :])
                    cond_image = torch.cat([cond_image, alpha_channel], dim=1)
                else:
                    raise ValueError(f"Conditional image must have 3 or 4 channels, got {cond_image.shape[1]}")            
          
            # Apply enhancement factor
            enhancement_factor = 2
            # Apply the mask with the properly sized conditioning image
            modified_pred_x0 = torch.where(
                mask.to(pred_x0.device) > 0.5,
                cond_image * enhancement_factor,
                modified_pred_x0
            )
            
        return modified_pred_x0, masks
    
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
        sam2_checkpoint = os.path.join(grounded_sam_path, 'checkpoints/sam2.1_hiera_large.pt')
        
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
            