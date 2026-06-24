from argparse import ArgumentParser
from omegaconf import OmegaConf
import os
import torch
import numpy as np
from PIL import Image
import imageio
import warnings
warnings.filterwarnings("ignore")
from pytorch_lightning import seed_everything
import logging
logging.getLogger().setLevel(logging.ERROR)  # Only show ERROR messages
logging.disable(logging.INFO)
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)
from scripts.evaluation.funcs import load_model_checkpoint, load_prompts,save_gif, save_videos
from scripts.evaluation.funcs import base_ddim_sampling, fifo_ddim_sampling
from utils.utils import instantiate_from_config
from lvdm.models.samplers.ddim import DDIMSampler
from torch.nn import functional as F
from torchvision import transforms
import time

def set_directory(args, prompt, conditioned_image_path=None):
    if args.output_dir is None:
        output_dir = f"results/videocraft_v2_fifo/random_noise/sam2/{prompt[:100]}"
        if args.eta != 1.0:
            output_dir += f"/eta{args.eta}"

        if args.new_video_length != 100:
            output_dir += f"/{args.new_video_length}frames"
        if not args.lookahead_denoising:
            output_dir = output_dir.replace(f"{prompt[:100]}", f"{prompt[:100]}/no_lookahead_denoising")
        if args.num_partitions != 4:
            output_dir = output_dir.replace(f"{prompt[:100]}", f"{prompt[:100]}/n={args.num_partitions}")
        if args.video_length != 16:
            output_dir = output_dir.replace(f"{prompt[:100]}", f"{prompt[:100]}/f={args.video_length}")
        if args.experiment_condition != "baseline":
            output_dir = output_dir.replace(f"{prompt[:100]}", f"{prompt[:100]}/{args.experiment_condition}")

    else:
        output_dir = args.output_dir

    latents_dir = f"results/videocraft_v2_fifo/latents/{args.num_inference_steps}steps/{prompt[:100]}/eta{args.eta}"

    print("The results should be saved in", output_dir)
    print("The latents should be saved in", latents_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)

    return output_dir, latents_dir


def main(args):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda()
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    latent_height = args.height // 8
    latent_width = args.width // 8
    frames = args.video_length
    channels = model.channels

    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    prompt_list = load_prompts(args.prompt_file, args.prompt_index)
    num_samples = len(prompt_list)
    indices = list(range(num_samples))
    indices = indices[args.rank::args.num_processes]
    
    for idx in indices:
        data = prompt_list[idx]
        prompt = data["prompt"]
        conditioned_object = data["conditioned_object"]
        conditioned_image_path = data["conditioned_image_path"]
        conditioned_prompt = data["conditioned_prompt"]
        gamma = data["gamma"]
        output_dir, latents_dir = set_directory(args, prompt, conditioned_image_path)
        if os.path.exists(output_dir):   
            # Check if the output directory has a file called "fifo.mp4"
            if os.path.exists(output_dir+"/fifo.mp4"):
                print(f"The output directory {output_dir} already exists and has a file called fifo.mp4")
                continue
            else:
                print(f"The output directory {output_dir} already exists but does not have a file called fifo.mp4")

        batch_size = 1
        noise_shape = [batch_size, channels, frames, latent_height, latent_width]
        fps = torch.tensor([args.fps]*batch_size).to(model.device).long()
        prompts = [prompt]
        targets = conditioned_object + "."
        print(f"The targets are {targets}")
        print(f"The prompts are {prompts}")
        text_emb = model.get_learned_conditioning(prompts)
        cond = {"c_crossattn": [text_emb], "fps": fps}
            
        transform = transforms.Compose([
            transforms.Resize((args.height//8, args.width//8)),
            transforms.CenterCrop((args.height//8, args.width//8)),
            transforms.ToTensor(),
        ])
        cond_image = Image.open(conditioned_image_path).convert("RGBA")
        cond_image = transform(cond_image).unsqueeze(1).unsqueeze(0)
        cond_image = cond_image.to("cuda")

        ## inference
        is_run_base = not (os.path.exists(latents_dir+f"/{args.num_inference_steps}.pt") and os.path.exists(latents_dir+f"/0.pt"))
        if not is_run_base:
            ddim_sampler = DDIMSampler(model)
            ddim_sampler.make_schedule(ddim_num_steps=args.num_inference_steps, ddim_eta=args.eta, verbose=False)
        else:
            base_tensor, ddim_sampler, _ = base_ddim_sampling(model, cond, noise_shape, \
                                                args.num_inference_steps, args.eta, args.unconditional_guidance_scale, \
                                                latents_dir=latents_dir)
            save_gif(base_tensor, output_dir, "origin")
        if conditioned_prompt:
            cond["c_crossattn"].append(model.get_learned_conditioning([conditioned_prompt]))
        
        start_time = time.time()
        video_frames = fifo_ddim_sampling(
            args, model, cond, noise_shape, ddim_sampler, 
            args.unconditional_guidance_scale, 
            output_dir=output_dir, 
            latents_dir=latents_dir, 
            save_frames=args.save_frames, 
            conditioned_image=cond_image, 
            targets=targets, 
            gamma=gamma,
            experiment_condition=args.experiment_condition
        )
        end_time = time.time()
        print(f"Time taken in FIFO Semantic Mixing Pipeline: {end_time - start_time} seconds")
        if args.output_dir is None:
            output_path = output_dir+"/fifo"
        else:
            output_path = output_dir+f"/{prompt[:100]}"

        if args.use_mp4:
            imageio.mimsave(output_path+".mp4", video_frames[-args.new_video_length//2:], fps=args.output_fps) # 
        else:
            imageio.mimsave(output_path+".gif", video_frames[-args.new_video_length//2:], duration=int(1000/args.output_fps)) # 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='videocrafter_models/base_512_v2/model.ckpt', help="checkpoint path")
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml", help="config (yaml) path")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--video_length", type=int, default=16, help="f in paper")
    parser.add_argument("--num_partitions", "-n", type=int, default=4, help="n in paper")
    parser.add_argument("--num_inference_steps", type=int, default=16, help="number of inference steps, it will be f * n forcedly")
    parser.add_argument("--prompt_file", "-p", type=str, default="prompts/prompts.csv", help="path to the prompt file")
    parser.add_argument("--new_video_length", "-l", type=int, default=100, help="N in paper; desired length of the output video")
    parser.add_argument("--num_processes", type=int, default=1, help="number of processes if you want to run only the subset of the prompts")
    parser.add_argument("--rank", type=int, default=0, help="rank of the process(0~num_processes-1)")
    parser.add_argument("--height", type=int, default=320, help="height of the output video")
    parser.add_argument("--width", type=int, default=512, help="width of the output video")
    parser.add_argument("--save_frames", action="store_true", default=True, help="save generated frames for each step")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="prompt classifier-free guidance")
    parser.add_argument("--lookahead_denoising", "-ld", action="store_true", default=True)
    parser.add_argument("--eta", "-e", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default=None, help="custom output directory")
    parser.add_argument("--use_mp4", action="store_true", default=True, help="use mp4 format for the output video")
    parser.add_argument("--output_fps", type=int, default=5, help="fps of the output video")
    parser.add_argument("--prompt_index", type=int, default=1, help="index of the prompt to run")
    parser.add_argument("--experiment_condition", type=str, default="baseline", help="Experiment condition")
    
    args = parser.parse_args()

    args.num_inference_steps = args.video_length * args.num_partitions

    seed_everything(args.seed)

    main(args)