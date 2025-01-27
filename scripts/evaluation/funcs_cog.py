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
from sam2.build_sam import build_sam2
from sam2.sam2_video_predictor import SAM2VideoPredictor
from PIL import Image
from diffusers.utils import export_to_video
import torch.nn as nn

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@torch.no_grad()
def visualize_latents(latents, output_dir, step, model):
    """
    Decodes latents into images and saves them for visualization.
    
    Args:
        latents (torch.Tensor): The latent tensor to decode.
        output_dir (str): Directory to save the images.
        step (int): The current denoising step.
        model: The model to decode the latents.
    """
    # Decode latents
    decoded_frames = model.decode_latents(latents)
    decoded_frames = decoded_frames.squeeze(0).permute(1, 0, 2, 3)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save each frame as an image
    for i, frame in enumerate(decoded_frames):
        # Convert tensor to PIL image
        frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(frame_np)
        
        # Save image with a unique filename
        image.save(os.path.join(output_dir, f"latent_step_{step:03d}_frame_{i:03d}.png"))

    print(f"Saved latent visualizations for step {step}.")

def print_gpu_usage():
    print(f"GPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Max Reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    print("=" * 50)

def prepare_latents(args, latents_dir, model):
    latents_list = []    
    # Load pre-saved latents from the directory
    video = torch.load(latents_dir + f"/{args.num_inference_steps}.pt")
    print(f"Loaded video latents with shape: {video.shape}")  # torch.Size([3, 480, 49, 720])
    breakpoint()
    last_frame = video[:, :, -1, :]
    last_frame_np = (last_frame.permute(1, 2, 0).cpu().numpy())
    last_frame_np = ((last_frame_np - last_frame_np.min())/ (last_frame_np.max() - last_frame_np.min()) * 255).astype(np.uint8)
    image = Image.fromarray(last_frame_np)

    latents_channels = model.transformer.config.in_channels // 2
    latents, image_latents = model.prepare_latents(
        image, 
        batch_size,
        latent_channels,
        num_frames,
        height,
        width,
        torch.float16,
        device,
        generator,
    )

    # # Convert 3-channel latents to 4-channel format (e.g., add an alpha channel)
    # if video.shape[0] == 3:
    #     alpha_channel = torch.ones_like(video[:1, :, :, :])  # Create an alpha channel filled with 1s
    #     video = torch.cat([video, alpha_channel], dim=0)  # Concatenate along the channel dimension
    #     print(f"Converted video to 4-channel format with shape: {video.shape}")

    # # Handle lookahead denoising
    # if args.lookahead_denoising:
    #     """
    #     To have enough initiail noisy frames to start the lookahead denoising
    #     Generate dummy noisy frames for the first half of the video
    #     """
    #     for i in range(args.video_length // 2):
    #         alpha = model.scheduler.alphas[0]
    #         beta = 1 - alpha
    #         latents = alpha**(0.5) * video[:,:,[0]] + beta**(0.5) * torch.randn_like(video[:,:,[0]])
    #         latents_list.append(latents) # [z1, z1]

    # # Prepare latents for each inference step
    # for i in range(args.num_inference_steps):
    #     # Simulate the noise schedule (linear approximation)
    #     alpha = model.scheduler.alphas[i]
    #     beta = 1 - alpha
    #     frame_idx = max(0, i-(args.num_inference_steps - args.video_length)) 
    #     latents = (alpha)**(0.5) * video[:,:,[frame_idx]] + (1-alpha)**(0.5) * torch.randn_like(video[:,:,[frame_idx]])
    #     latents_list.append(latents)

    # # Concatenate all latents along the temporal dimension
    # latents = torch.cat(latents_list, dim=2)  # Shape: [batch, channels, frames, height, width]
    # print(f"Final latents shape: {latents.shape}")  # torch.Size([4, 480, 72, 720])

    return latents

def shift_latents(latents, output_dir, index):
    anchor_frame = latents[:, 0].clone().unsqueeze(2) # b,c,1,h,w
    ## Decode the anchor frame
    clean_image = tensor2image(anchor_frame)
    breakpoint()
    clean_image.save(f"{output_dir}/fifo/{index}:03d.png")
    latents[:, :, :-1] = latents[:, :, 1:].clone()
    new_noise = torch.randn_like(latents[:, :, -1]).unsqueeze(2)
    freq_filter = get_freq_filter(anchor_frame.shape, latents.device, "gaussian", 1, 0.25, 0.25)
    latents[:, :, -1] = freq_mix_3d(anchor_frame, new_noise, freq_filter).squeeze(2)
    return latents

def video_segmentation_with_sam2(video_path, output_dir, text_prompt, model_cfg, checkpoint, device="cuda"):
    """
    Segment objects in a video using SAM2 with temporal tracking.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save the output.
        text_prompt (str): Text prompt for segmentation.
        model_cfg (str): Path to SAM2 model configuration file.
        checkpoint (str): Path to SAM2 model checkpoint.
        device (str): "cuda" or "cpu".
    """
    # Load video
    sam2_checkpoint = "../sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1_hiera_l.yaml"

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize SAM2
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    video_predictor = SAM2VideoPredictor(sam2_model)

    # Process frames with SAM2
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB for SAM2

    frames = np.stack(frames)  # (num_frames, H, W, C)

    # Generate masks for the entire video
    video_predictor.set_video(frames, text_prompt=text_prompt)
    masks, scores = video_predictor.predict()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        mask_overlay = cv2.addWeighted(frames[i], 0.5, mask * 255, 0.5, 0)
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}_mask.png"), mask_overlay)

    # Release resources
    cap.release()
    print(f"Processed video saved in {output_dir}")

@torch.no_grad()
def fifo_sampling_cogvideo(args, model, conditioning, cfg_scale=1.0, output_dir=None,
                           latents_dir=None, save_frames=False, save_mid_steps=False,
                           save_mid_steps_every=10, **kwargs):

    generator = torch.Generator().manual_seed(321)

    model = accelerator.prepare(model)
    device = torch.device("cuda")
    batch_size = 1  # Fixed to 1 since CogVideo generates videos one prompt at a time.
    kwargs.update({"clean_cond": True})
    num_frames = args.video_length // 2 + args.num_inference_steps
    num_channels_latents = model.transformer.config.in_channels

    # Prepare initial latents
    video = torch.load(latents_dir + f"/{args.num_inference_steps}.pt")
    print(f"Loaded video latents with shape: {video.shape}")  # torch.Size([3, 480, 49, 720])
    height = video.shape[1]
    width = video.shape[3]
    last_frame = video[:, :, -1, :].unsqueeze(0)
    print(f"Last frame shape: {last_frame.shape}")

    # encoder_states = conditioning["prompts"].to(torch.float16)
    encoder_states = conditioning["prompts"].to(torch.float16) # torch.cat([prompt for prompt in conditioning["prompts"]], dim=0).to(torch.float16)
    latents_channels = model.transformer.config.in_channels // 2
    latents, image_latents = model.prepare_latents(
        last_frame, 
        batch_size,
        latents_channels,
        num_frames,
        height,
        width,
        torch.float16,
        device,
        generator,
    )
    

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

    for i, t in enumerate(tqdm(timesteps)):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
        latent_image_input = torch.cat([image_latents] * 2)
        latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

        timestep = t.expand(latent_model_input.shape[0]) 
        
        image_rotary_emb = (
            model._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if model.transformer.config.use_rotary_positional_embeddings
            else None
        ) 

        noise_pred = model.transformer(
            hidden_states=latent_model_input, # torch.Size([2, 18, 32, 60, 90])
            encoder_hidden_states=encoder_states, # torch.Size([2, 226, 4096])
            timestep=timestep,
            image_rotary_emb=image_rotary_emb, # torch.Size([24300, 64])
            return_dict=False,
        )[0] # torch.Size([2, 18, 16, 60, 90])
        # compute the previous noisy sample x_t -> x_t-1
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        if cfg_scale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, cfg_scale)
        
        print(f"Noise prediction shape: {noise_pred.shape}")

        output = model.scheduler.step(
            noise_pred, t, latents, return_dict=True
        )
        
        latents = output['prev_sample']
        # visualize_latents(latents, os.path.join(output_dir, "latent_visualizations"), i, model)

    video = model.decode_latents(latents)
    videos = model.video_processor.postprocess_video(video=video, output_type='np')

    torch.cuda.empty_cache()  # Clear GPU cache to free up memory
    return videos

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
