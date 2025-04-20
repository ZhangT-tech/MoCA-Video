import os, sys, glob, math
import numpy as np
from collections import OrderedDict
from decord import VideoReader, cpu
import cv2
import torch
import torchvision
import imageio
from tqdm import trange
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
import torch.fft as fft
import math
from utils.freeinit_utils import freq_mix_3d, get_freq_filter
import torchvision.transforms as transforms
from einops import rearrange, repeat
import csv
from PIL import Image
import torch.nn.functional as F

def prepare_latents(args, input_path, sampler, model=None, data=None):
    """
    Prepare latents for sampling, with support for DAVIS data and DDIM inversion.
    
    Args:
        args: Command line arguments
        input_path: Path to save latents
        sampler: DDIM sampler instance
        model: Diffusion model
        data: Optional DAVIS data tuple (frames, masks)
        
    Returns:
        latents: Prepared latents for sampling
    """
    if data is not None:
        # Handle DAVIS data
        frames, masks = data
        frames = frames.to("cuda")
        
        # Convert RGBA to RGB if needed
        if frames.shape[1] == 4:  # RGBA
            # Convert RGBA to RGB by removing alpha channel
            frames = frames[:, :3]  # Keep only RGB channels
        
        # Use DDIM inversion from the sampler
        latents = sampler.ddim_inversion(
            frames=frames,
            num_inference_steps=args.num_inference_steps,
            eta=args.eta,
            latents_dir=input_path
        )
    else:
        # Original latent preparation logic for non-DAVIS data
        latents_list = []
        video = torch.load(input_path+f"/{args.num_inference_steps}.pt")
        
        if args.lookahead_denoising:
            video = video.to("cuda")
            for i in range(args.video_length // 2):
                alpha = sampler.ddim_alphas[0]
                beta = 1 - alpha
                latents = alpha**(0.5) * video[:,:,[0]] + beta**(0.5) * torch.randn_like(video[:,:,[0]])
                latents_list.append(latents)
         
        for i in range(args.num_inference_steps): 
            alpha = sampler.ddim_alphas[i]
            beta = 1 - alpha
            frame_idx = max(0, i-(args.num_inference_steps - args.video_length)) 
            latents = (alpha)**(0.5) * video[:,:,[frame_idx]] + (1-alpha)**(0.5) * torch.randn_like(video[:,:,[frame_idx]])
            latents_list.append(latents)
            
        latents = torch.cat(latents_list, dim=2)
    
    return latents



def shift_latents(latents, masks=None):

    if masks is None:
        anchor_frame = latents[:, :, 0].clone().unsqueeze(2) # b,c,1,h,w
        
        latents[:, :, :-1] = latents[:, :, 1:].clone()

        new_noise = torch.randn_like(latents[:, :, -1]).unsqueeze(2)
        
        freq_filter = get_freq_filter(anchor_frame.shape, latents.device, "gaussian", 1, 0.25, 0.25)

        latents[:, :, -1] = freq_mix_3d(anchor_frame, new_noise, freq_filter).squeeze(2)
        
        return latents
    
    else:
        # shift latents
        latents[:,:,:-1] = latents[:,:,1:].clone()

        # add new noise to the last frame
        latents[:,:,-1] = torch.randn_like(latents[:,:,-1])

        # shift masks
        masks[:,:,:-1] = masks[:,:,1:].clone()
        ## create all zeros mask for the last frame
        masks[:,:,-1] = torch.zeros_like(masks[:,:,-1])
        return latents, masks

def batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, **kwargs):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    x_T = None
    batch_variants = []
    #batch_variants1, batch_variants2 = [], []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=True,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            **kwargs
                                            )
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage_2DAE(samples) # b,c,f,h,w
        batch_variants.append(batch_images)
    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1) # b,n,c,f,h,w
    return batch_variants

def base_ddim_sampling(model, cond, noise_shape, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, latents_dir=None, **kwargs):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type # used to improve the quality and diversity of generated samples with additional conditioning that is not directly rely on the input data
    batch_size = noise_shape[0]
    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq": # True, a neutral sample.
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        ## ===================== can add additional image embedding token =================================
        if hasattr(model, 'embedder'): # False 
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict): # True
            uc = {key:cond[key] for key in cond.keys()} # preserve the original condition keys-value pairs
            ## ===================== Check if it's working (Original not included) =================================
            ## Maybe somewhere else, they add them together cause this is simply the unconditional guidance
            # uc_emd = torch.cat([uc_emb, uc['c_crossattn'][0]], dim=1) # concatenate the cross attention key-value pairs
            uc.update({'c_crossattn': [uc_emb]}) # update the cross attention key-value pairs, overwrite or concatenate?
        else: # False
            uc = uc_emb
    else:
        uc = None
    
    x_T = None

    if ddim_sampler is not None:
        kwargs.update({"clean_cond": True})
        samples, _ = ddim_sampler.sample(S=ddim_steps,
                                        conditioning=cond, # the preserved conditional embeddings
                                        batch_size=noise_shape[0], # [batch_size, channels, frames, height, width] 
                                        shape=noise_shape[1:],
                                        verbose=True,
                                        unconditional_guidance_scale=cfg_scale,
                                        unconditional_conditioning=uc, # obtained last step
                                        eta=ddim_eta,
                                        temporal_length=noise_shape[2],
                                        conditional_guidance_scale_temporal=temporal_cfg_scale,
                                        x_T=x_T,
                                        latents_dir=latents_dir,
                                        **kwargs
                                        )
    ## reconstruct from latent to pixel space
    # samples: b,c,f,h,w
    batch_images = model.decode_first_stage_2DAE(samples) # b,c,f,H,W

    return batch_images, ddim_sampler, samples

def fifo_ddim_sampling(args, model, conditioning, noise_shape, ddim_sampler,\
                        cfg_scale=1.0, output_dir=None, latents_dir=None, save_frames=False, conditioned_image_path=None, targets=None, gamma=0.5, use_self_attention=False, davis_data=None, anchor_frame=None, **kwargs):
    batch_size = noise_shape[0]
    kwargs.update({"clean_cond": True})

    ## Handle concept removal case
    if conditioned_image_path == "empty":
        cond_image = None
    else:
        ## Obtain the conditioning image
        transform = transforms.Compose([
            transforms.Resize((args.height//8, args.width//8)),
            transforms.CenterCrop((args.height//8, args.width//8)),
            transforms.ToTensor(),
        ])
        cond_image = Image.open(conditioned_image_path).convert("RGBA")
        cond_image = transform(cond_image).unsqueeze(1).unsqueeze(0)
        
        cond_image = cond_image.to("cuda")

    ## Obtain the target
    target = targets

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
    
    cond = conditioning

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        prompts = batch_size * [""]
        uc_emb = model.get_learned_conditioning(prompts)
        
        uc = {key:cond[key] for key in cond.keys()}
        uc.update({'c_crossattn': [uc_emb]})
        
    else:
        uc = None
    
    # Latents preparation
    latents = prepare_latents(args, latents_dir, ddim_sampler, model, data=davis_data)
    num_frames_per_gpu = args.video_length 
    if args.save_frames:
        fifo_dir = os.path.join(output_dir, "fifo")
        os.makedirs(fifo_dir, exist_ok=True)

    fifo_video_frames = []

    timesteps = ddim_sampler.ddim_timesteps
    indices = np.arange(args.num_inference_steps) 
    if args.lookahead_denoising:
        timesteps = np.concatenate([np.full((args.video_length//2,), timesteps[0]), timesteps]) 
        indices = np.concatenate([np.full((args.video_length//2,), 0), indices]) 

    # Load DAVIS data if provided
    if davis_data is not None:
        frames, masks = davis_data
        frames = frames.to("cuda")
        masks = masks.to("cuda")
        current_frame_idx = 0
    # anchor_frame = frames[:, :, 0].clone().unsqueeze(2)

    for i in trange(args.new_video_length + args.num_inference_steps - args.video_length, desc="fifo sampling"):
        for rank in reversed(range(2 * args.num_partitions if args.lookahead_denoising else args.num_partitions)):
            start_idx = rank*(num_frames_per_gpu // 2) if args.lookahead_denoising else rank*num_frames_per_gpu
            midpoint_idx = start_idx + num_frames_per_gpu // 2 
            end_idx = start_idx + num_frames_per_gpu 
            
            t = timesteps[start_idx:end_idx] 
            idx = indices[start_idx:end_idx]            
            print(f"start_idx: {start_idx}, midpoint_idx: {midpoint_idx}, end_idx: {end_idx}")
            print(f"t: {t}, idx: {idx}")
            input_latents = latents[:,:,start_idx:end_idx].clone() 
            input_masks = masks[:,:,start_idx:end_idx].clone() if masks is not None else None
            ## Visualize the latents
            latents_dir = os.path.join("visualizations", "latents")
            os.makedirs(latents_dir, exist_ok=True)
            
            # Check if input_latents is not empty and has correct dimensions
            if input_latents.numel() > 0 and input_latents.dim() == 5:  # [B, C, T, H, W]
                # Ensure we have valid data to visualize
                if input_latents.shape[2] > 0:  # Check if we have frames
                    # Squeeze batch dimension and permute for visualization
                    vis_latents = input_latents.squeeze(0).permute(1, 0, 2, 3)  # [T, C, H, W]
                    try:
                        latents_grid = torchvision.utils.make_grid(
                            vis_latents,
                            nrow=vis_latents.shape[0],  # Use actual number of frames
                            normalize=True,
                            padding=2
                        )
                        torchvision.utils.save_image(latents_grid, os.path.join(latents_dir, "latents_grid_{}.png".format(i)))
                    except Exception as e:
                        print(f"Warning: Could not visualize latents: {str(e)}")
            else:
                print(f"Warning: Invalid latents shape for visualization: {input_latents.shape}")   

            # Use DAVIS masks if available
            if davis_data is not None:
                ## Only for the first 16 frames and others dont use masks
                if input_masks.shape[2] == input_latents.shape[2]:
                    current_masks = input_masks

                    ## Visualize the masks
                    masks_dir = os.path.join("visualizations", "masks")
                    os.makedirs(masks_dir, exist_ok=True)
                    masks_grid = torchvision.utils.make_grid(
                        current_masks.squeeze(0).permute(1, 0, 2, 3),
                        nrow=len(current_masks),
                        normalize=True,
                        padding=2)
                    torchvision.utils.save_image(masks_grid, os.path.join(masks_dir, "masks_grid_{}.png".format(i)))
                    
                    output_latents, _ = ddim_sampler.fifo_onestep(
                        cond=cond,
                        shape=noise_shape,
                        latents=input_latents,
                        timesteps=t,
                        indices=idx,
                        unconditional_guidance_scale=cfg_scale,
                        unconditional_conditioning=uc,
                        cond_image=cond_image,
                        target=target,
                        gamma=gamma,
                        use_self_attention=use_self_attention,
                        davis_masks=input_masks,  # Pass DAVIS masks
                        **kwargs
                    )
                else:
                    # For subsequent iterations, don't use masks
                    output_latents, _ = ddim_sampler.fifo_onestep(
                        cond=cond,
                        shape=noise_shape,
                        latents=input_latents,
                        timesteps=t,
                        indices=idx,
                        unconditional_guidance_scale=cfg_scale,
                        unconditional_conditioning=uc,
                        cond_image=cond_image,
                        target=target,
                        gamma=gamma,
                        use_self_attention=use_self_attention,
                        no_sam=True,
                        **kwargs
                    )
            else:
                output_latents, _ = ddim_sampler.fifo_onestep(
                    cond=cond,
                    shape=noise_shape,
                    latents=input_latents,
                    timesteps=t,
                    indices=idx,
                    unconditional_guidance_scale=cfg_scale,
                    unconditional_conditioning=uc,
                    cond_image=cond_image,
                    target=target,
                    gamma=gamma,
                    use_self_attention=use_self_attention,
                    **kwargs
                )

            if args.lookahead_denoising:
                latents[:,:,midpoint_idx:end_idx] = output_latents[:,:,-(num_frames_per_gpu//2):] 
            else:
                latents[:,:,start_idx:end_idx] = output_latents
            del output_latents
        

        # reconstruct from latent to pixel space
        first_frame_idx = args.video_length // 2 if args.lookahead_denoising else 0
        frame_tensor = model.decode_first_stage_2DAE(latents[:,:,[first_frame_idx]])
        image = tensor2image(frame_tensor)
        if save_frames:
            fifo_path = os.path.join(fifo_dir, f"{i}.png")
            image.save(fifo_path)
        fifo_video_frames.append(image)

        ## Shift the latents
        if masks is not None:
            latents, masks = shift_latents(latents, masks) 
        else:
            latents = shift_latents(latents) 

    return fifo_video_frames

def fifo_ddim_sampling_multiprompts(args, model, conditioning, noise_shape, ddim_sampler, multiprompts,
                                    cfg_scale=1.0, output_dir=None, latents_dir=None, save_frames=False, **kwargs):
    batch_size = noise_shape[0]
    kwargs.update({"clean_cond": True})

    prompt_lengths = np.array([int(i) for i in multiprompts[-1].split(',')]).cumsum()
    multiprompts_embed = [model.get_learned_conditioning(prompt) for prompt in multiprompts[:-1]]
    assert len(prompt_lengths) == len(multiprompts_embed)

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
    
    cond = conditioning
    ## construct unconditional guidance
    if cfg_scale != 1.0:
        prompts = batch_size * [""]
        #prompts = N * T * [""]  ## if is_imgbatch=True
        uc_emb = model.get_learned_conditioning(prompts)
        
        uc = {key:cond[key] for key in cond.keys()}
        uc.update({'c_crossattn': [uc_emb]})    
    else:
        uc = None

    latents = prepare_latents(args, latents_dir, ddim_sampler)

    num_frames_per_gpu = args.video_length
    fifo_dir = os.path.join(output_dir, "fifo")
    # os.makedirs(fifo_dir, exist_ok=True)

    fifo_video_frames = []

    timesteps = ddim_sampler.ddim_timesteps
    indices = np.arange(args.num_inference_steps) # n * f

    if args.lookahead_denoising:
        timesteps = np.concatenate([np.full((args.video_length//2,), timesteps[0]), timesteps])
        indices = np.concatenate([np.full((args.video_length//2,), 0), indices])
    
    j = 0
    for i in trange(prompt_lengths[-1] + args.num_inference_steps - args.video_length, desc="fifo sampling"):

        if i - (args.num_inference_steps - args.video_length) >= prompt_lengths[j]:
            j = j +1
        embed = multiprompts_embed[j]

        cond.update({'c_crossattn':[embed]})
        for rank in reversed(range(2 * args.num_partitions if args.lookahead_denoising else args.num_partitions)):
            start_idx = rank*(num_frames_per_gpu // 2) if args.lookahead_denoising else rank*num_frames_per_gpu
            midpoint_idx = start_idx + num_frames_per_gpu // 2
            end_idx = start_idx + num_frames_per_gpu

            t = timesteps[start_idx:end_idx]
            idx = indices[start_idx:end_idx]

            input_latents = latents[:,:,start_idx:end_idx].clone()
            output_latents, _ = ddim_sampler.fifo_onestep(
                                            cond=cond,
                                            shape=noise_shape,
                                            latents=input_latents,
                                            timesteps=t,
                                            indices=idx,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            **kwargs
                                            )
            if args.lookahead_denoising:
                latents[:,:,midpoint_idx:end_idx] = output_latents[:,:,-(num_frames_per_gpu//2):]
            else:
                latents[:,:,start_idx:end_idx] = output_latents
            del output_latents
        

        # reconstruct from latent to pixel space
        first_frame_idx = args.video_length // 2 if args.lookahead_denoising else 0
        frame_tensor = model.decode_first_stage_2DAE(latents[:,:,[first_frame_idx]]) # b,c,1,H,W
        image = tensor2image(frame_tensor)
        if save_frames:
            fifo_path = os.path.join(fifo_dir, f"{i}.png")
            image.save(fifo_path)
        fifo_video_frames.append(image)
            
        latents = shift_latents(latents)
    return fifo_video_frames

def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def get_dirlist(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        try:
            ## deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model
    load_checkpoint(model, ckpt, full_strict=True)
    print('>>> model checkpoint loaded.')
    return model


def load_prompts(prompt_file, prompt_index=None):
    with open(prompt_file, 'r') as f:
        # Use csv reader to properly handle quoted strings containing commas
        reader = csv.DictReader(f)
        
        if prompt_index is not None:
            # Skip to the desired row
            for i, row in enumerate(reader):
                if i == prompt_index:
                    return [{
                        "prompt": row["prompt"].strip(),
                        "conditioned_object": row["conditioned_object"].strip(),
                        "conditioned_image_path": row["conditioned_image_path"].strip(),
                        "conditioned_prompt": row["conditioned_prompt"].strip()+".",
                        "gamma": float(row["gamma"].strip())
                    }]
            raise ValueError(f"Prompt index {prompt_index} exceeds number of available prompts")
        
        # If no index specified, return all prompts (original behavior)
        prompt_list = []
        for row in reader:
            prompt_data = {
                "prompt": row["prompt"].strip(),
                "conditioned_object": row["conditioned_object"].strip(),
                "conditioned_image_path": row["conditioned_image_path"].strip(),
                "conditioned_prompt": row["conditioned_prompt"].strip()+".",
                "gamma": float(row["gamma"].strip())
            }
            prompt_list.append(prompt_data)
        return prompt_list


def load_video_batch(filepath_list, frame_stride, video_size=(256,256), video_frames=16):
    '''
    Notice about some special cases:
    1. video_frames=-1 means to take all the frames (with fs=1)
    2. when the total video frames is less than required, padding strategy will be used (repreated last frame)
    '''
    fps_list = []
    batch_tensor = []
    assert frame_stride > 0, "valid frame stride should be a positive interge!"
    for filepath in filepath_list:
        padding_num = 0
        vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames-1) // frame_stride + 1
        if video_frames < 0:
            ## all frames are collected: fs=1 is a must
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [frame_stride*i for i in range(query_frames)]

        ## [t,h,w,c] -> [c,t,h,w]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frame_tensor = (frame_tensor / 255. - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = torch.cat([frame_tensor, *([frame_tensor[:,-1:,:,:]]*padding_num)], dim=1)
            print(f'{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded.')
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps/frame_stride)
        fps_list.append(sample_fps)
    
    return torch.stack(batch_tensor, dim=0)

def load_image_batch(filepath_list, image_size=(256,256)):
    batch_tensor = []
    for filepath in filepath_list:
        _, filename = os.path.split(filepath)
        _, ext = os.path.splitext(filename)
        if ext == '.mp4':
            vidreader = VideoReader(filepath, ctx=cpu(0), width=image_size[1], height=image_size[0])
            frame = vidreader.get_batch([0])
            img_tensor = torch.tensor(frame.asnumpy()).squeeze(0).permute(2, 0, 1).float()
        elif ext == '.png' or ext == '.jpg':
            img = Image.open(filepath).convert("RGBA")
            rgb_img = np.array(img, np.float32)
            #bgr_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            #bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (image_size[1],image_size[0]), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        else:
            print(f'ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]')
            raise NotImplementedError
        img_tensor = (img_tensor / 255. - 0.5) * 2
        batch_tensor.append(img_tensor)
    return torch.stack(batch_tensor, dim=0)


def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [t, n*h, w, 3]
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})

def save_gif(batch_tensors, savedir, name):
    vid_tensor = torch.squeeze(batch_tensors) # c,f,h,w

    video = vid_tensor.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    video = video.permute(1, 0, 2, 3) # f,c,h,w

    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1) # f,h,w,c

    frames = video.chunk(video.shape[0], dim=0)
    frames = [frame.squeeze(0) for frame in frames]
    savepath = os.path.join(savedir, f"{name}.gif")

    imageio.mimsave(savepath, frames, duration=100)

def tensor2image(batch_tensors):
    img_tensor = torch.squeeze(batch_tensors) # c,h,w

    image = img_tensor.detach().cpu()
    image = torch.clamp(image.float(), -1., 1.)

    image = (image + 1.0) / 2.0
    image = (image * 255).to(torch.uint8).permute(1, 2, 0) # h,w,c
    image = image.numpy()
    image = Image.fromarray(image)
    return image


def load_davis_data(video_name, davis_root, frame_stride=1, video_size=(256,256), video_frames=16, sampling_strategy="first"):
    """
    Load frames and masks from DAVIS dataset.
    
    Args:
        video_name (str): Name of the video sequence
        davis_root (str): Root directory of DAVIS dataset
        frame_stride (int): Stride for frame sampling (used only for uniform sampling)
        video_size (tuple): Target size for frames (height, width)
        video_frames (int): Number of frames to load
        sampling_strategy (str): Strategy for selecting frames ("first", "random", or "uniform")
        
    Returns:
        tuple: (frames_tensor, masks_tensor)
    """
    # Construct paths
    frames_dir = os.path.join(davis_root, 'JPEGImages', '480p', video_name)
    masks_dir = os.path.join(davis_root, 'Annotations', '480p', video_name)
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    
    total_frames = len(frame_files)
    
    # Select frame indices based on strategy
    if sampling_strategy == "first":
        # Take first 16 frames
        frame_indices = list(range(min(video_frames, total_frames)))
    elif sampling_strategy == "random":
        # Randomly sample 16 frames
        frame_indices = np.random.choice(total_frames, size=min(video_frames, total_frames), replace=False)
        frame_indices = sorted(frame_indices)  # Sort to maintain temporal order
    elif sampling_strategy == "uniform":
        # Sample every nth frame to get 16 frames
        if total_frames <= video_frames:
            frame_indices = list(range(total_frames))
        else:
            # Calculate stride to get approximately 16 frames
            stride = max(1, total_frames // video_frames)
            frame_indices = list(range(0, total_frames, stride))[:video_frames]
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    # Load frames and masks
    frames = []
    masks = []

    for idx in frame_indices:
        # Load frame
        frame_path = os.path.join(frames_dir, frame_files[idx])
        # Load as RGB instead of RGBA
        frame = Image.open(frame_path).convert("RGBA")
        # Convert to numpy array in uint8 format
        frame = np.array(frame, dtype=np.uint8)
        
        # Only resize if dimensions don't match
        if frame.shape[:2] != (video_size[0]*8, video_size[1]*8):
            # Use Lanczos interpolation for highest quality
            frame = cv2.resize(frame, (video_size[1]*8, video_size[0]*8), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        frame_tensor = (frame_tensor / 255. - 0.5) * 2
        frames.append(frame_tensor)
        
        # Load mask using PIL instead of OpenCV
        mask_path = os.path.join(masks_dir, mask_files[idx])
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Only resize mask if dimensions don't match
        if mask.size != (video_size[1], video_size[0]):
            # Use nearest neighbor for binary masks
            mask = mask.resize((video_size[1], video_size[0]), 
                             Image.Resampling.NEAREST)
        
        mask = np.array(mask, dtype=np.uint8)
        masks.append(mask)
    
    # Convert to tensors
    frames = torch.stack(frames)  # [T, C, H, W]
    masks = torch.tensor(np.stack(masks)).unsqueeze(1).float()  # [T, 1, H, W]

    # Add batch dimension and rearrange to [B, C, T, H, W]
    frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
    masks = masks.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 1, T, H, W]

    # Ensure masks are binary (0 or 1)
    masks = (masks > 0).float()
    
    # Print tensor shapes for debugging
    print(f"Frames tensor shape: {frames.shape}")
    print(f"Masks tensor shape: {masks.shape}")
    
    # Visualize the sampled frames and masks
    vis_dir = os.path.join("visualizations", "davis_data", video_name)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save frames and masks
    for i in range(frames.shape[2]):  # T dimension
        # Save frame
        frame = frames[0, :, i]  # [C, H, W]
        frame = (frame + 1.0) / 2.0  # Convert back to [0, 1] range for saving
        torchvision.utils.save_image(frame, os.path.join(vis_dir, f"frame_{i:03d}.png"))
        
        # Save mask
        mask = masks[0, :, i]  # [1, H, W]
        torchvision.utils.save_image(mask, os.path.join(vis_dir, f"mask_{i:03d}.png"))
    
    # Print tensor shapes and value ranges
    print(f"\nDAVIS Data Visualization for {video_name}:")
    print(f"Frames tensor shape: {frames.shape}")
    print(f"Frames value range: [{frames.min():.3f}, {frames.max():.3f}]")
    print(f"Masks tensor shape: {masks.shape}")
    print(f"Masks value range: [{masks.min():.3f}, {masks.max():.3f}]")
    print(f"Visualizations saved to: {vis_dir}")
    
    return frames, masks