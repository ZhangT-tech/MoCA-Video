import os, sys, glob
import numpy as np
from collections import OrderedDict
from decord import VideoReader, cpu
import cv2
import torch
import torchvision
import imageio
from torch.cuda.amp import autocast
from tqdm import trange, tqdm
from diffusers import DDPMScheduler, DDIMScheduler
from accelerate import Accelerator
accelerator = Accelerator()
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from utils.freeinit_utils import freq_mix_3d, get_freq_filter
def print_gpu_usage():
    print(f"GPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Max Reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    print("=" * 50)

def prepare_latents(args, latents_dir):
    latents_list = []
    # Load pre-saved latents from the directory
    video = torch.load(latents_dir + f"/{args.num_inference_steps}.pt")
    print(f"Loaded video latents with shape: {video.shape}") # torch.Size([3, 480, 49, 720])

    # Handle lookahead denoising
    if args.lookahead_denoising:
        for i in range(args.video_length // 2):
            # Generate noisy frames for lookahead denoising
            noise = torch.randn_like(video[:, :, [0]])  # Random noise
            latents = 0.5**(0.5) * video[:, :, [0]] + (1 - 0.5)**(0.5) * noise  # Fixed scaling
            latents_list.append(latents)

    # Prepare latents for each inference step
    for i in range(args.num_inference_steps):
        # Simulate the noise schedule (linear approximation)
        alpha = 1.0 - (i / args.num_inference_steps)  # Linear noise schedule
        beta = 1.0 - alpha
        frame_idx = max(0, i - (args.num_inference_steps - args.video_length))
        
        # Add noise to the current frame
        noise = torch.randn_like(video[:, :, [frame_idx]])  # Random noise
        latents = (alpha)**0.5 * video[:, :, [frame_idx]] + (beta)**0.5 * noise
        latents_list.append(latents)

    # Concatenate all latents along the temporal dimension
    latents = torch.cat(latents_list, dim=2)  # Shape: [batch, channels, frames, height, width]
    print(f"Final latents shape: {latents.shape}") # torch.Size([3, 480, 72, 720])

    return latents

def shift_latents(latents):
    anchor_frame = latents[:, :, 0].clone().unsqueeze(2) # b,c,1,h,w
    latents[:, :, :-1] = latents[:, :, 1:].clone()
    new_noise = torch.randn_like(latents[:, :, -1]).unsqueeze(2)
    freq_filter = get_freq_filter(anchor_frame.shape, latents.device, "gaussian", 1, 0.25, 0.25)
    latents[:, :, -1] = freq_mix_3d(anchor_frame, new_noise, freq_filter).squeeze(2)
    return latents

@torch.no_grad()
def fifo_sampling_cogvideo(args, model, conditioning, cfg_scale=1.0, output_dir=None,
                           latents_dir=None, save_frames=False, save_mid_steps=False,
                           save_mid_steps_every=10, **kwargs):

    model = accelerator.prepare(model)
    device = torch.device("cuda")
    batch_size = 1  # Fixed to 1 since CogVideo generates videos one prompt at a time.
    kwargs.update({"clean_cond": True})

    # Prepare initial latents
    latents = prepare_latents(args, latents_dir).to(torch.float16).permute(0, 2, 1, 3).contiguous()  # Initialize noisy latents
    latents = torch.nn.functional.interpolate(
        latents, size=(latents.shape[2] // 4, latents.shape[3] // 4), mode="bilinear"
    )

    num_frames_per_partition = args.video_length // args.num_partitions

    if save_frames:
        fifo_dir = os.path.join(output_dir, "fifo")
        os.makedirs(fifo_dir, exist_ok=True)

    fifo_video_frames = []

    scheduler_name = "ddim"
    scheduler = (DDPMScheduler if scheduler_name == 'ddpm' else DDIMScheduler).from_pretrained(
        "THUDM/CogVideoX-5b", subfolder="scheduler"
    )
    model.scheduler = scheduler 
    print(f"After initializing scheduler:")
    
    # Set up timesteps
    model.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps
    print(f"Set up timesteps")
    num_frames_per_gpu = args.video_length
    
    if args.lookahead_denoising:
        # Prepare timesteps with lookahead denoising logic
        timesteps = np.concatenate([
            np.full((args.video_length // 2,), timesteps[0].cpu().numpy()),
            timesteps.cpu().numpy()
        ]).reshape(-1)  # Ensure 1D array

    # Total number of iterations as per Videocrafter
    total_iterations = args.new_video_length + args.num_inference_steps - args.video_length # 148
    for i in trange(total_iterations, desc="FIFO Sampling"):
        for rank in reversed(range(2 * args.num_partitions if args.lookahead_denoising else args.num_partitions)):
            start_idx = rank*(num_frames_per_gpu // 2) if args.lookahead_denoising else rank*num_frames_per_gpu
            midpoint_idx = start_idx + num_frames_per_gpu // 2
            end_idx = start_idx + num_frames_per_gpu
            print(f"the start index is {start_idx} and end index is {end_idx}")

            t = timesteps[start_idx:end_idx]
            # Extract partition latents
            input_latents = latents[:, start_idx:end_idx, :, :].clone() # latents shape: torch.Size([3, 480, 72, 720])
            # AMP for mixed precision
            with autocast():
                # Scale latents
                model_input = scheduler.scale_model_input(input_latents, t).unsqueeze(0)

                # Prepare timestep and encoder states
                timestep = torch.tensor([t], dtype=torch.long, device=device).reshape(-1)
                encoder_states = torch.cat([prompt for prompt in conditioning["prompts"]], dim=0).to(torch.float16)

                # Predict noise
                noise_pred = model.transformer(
                    hidden_states=model_input, # torch.Size([1, 3, 16, 120, 180])
                    encoder_hidden_states=encoder_states, # torch.Size([1, 226, 4096])
                    timestep=timestep, # torch.Size([16])
                    image_rotary_emb=None,
                    return_dict=False,
                )[0] #  torch.Size([1, 3, 256, 120, 180])
   
                # Perform guidance
                # breakpoint()
                # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=1) # torch.Size([1, 2, 256, 120, 180]), torch.Size([1, 1, 256, 120, 180])
                # noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond) # torch.Size([1, 2, 256, 120, 180])

                # Update latents
                # Move noise_pred and input_latents to GPU
                timestep = timestep.cpu()
                print(f"noise_pred device: {noise_pred.device}")
                print(f"timestep device: {timestep.device}")
                print(f"input_latents device: {input_latents.device}")
                next_latents = input_latents.clone().detach()
                for idx in range(timestep.shape[0]): 
                    next_latents = scheduler.step(noise_pred[:, :, (idx * 16):((idx+1) * 16), :, :], timestep[idx], next_latents)[0]

            # Update latents based on lookahead denoising or standard
            
            if args.lookahead_denoising:
                latents[:, midpoint_idx:end_idx] = next_latents[:, :,  -(num_frames_per_partition * 2):].squeeze(0)
            else:
                latents[:, start_idx:end_idx] = next_latents

            del input_latents, model_input, next_latents, noise_pred
            torch.cuda.empty_cache()
            print("Clear cache")



        # Decode the current denoised frame (optional visualization)
        if save_frames:
            frame_tensor = model.vae.decode(latents[:, :, [0]])  # Decode first frame in latents
            image = tensor2image(frame_tensor)
            image.save(f"{fifo_dir}/{i:03d}.png")
            fifo_video_frames.append(image)

        # Shift latents for the next timestep
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


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
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

from PIL import Image
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
