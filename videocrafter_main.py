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
        gamma = args.gamma_override if args.gamma_override is not None else data["gamma"]
        output_dir, latents_dir = set_directory(args, prompt, conditioned_image_path)
        if os.path.exists(output_dir):
            # Check if the output directory has a file called "fifo.mp4"
            if os.path.exists(output_dir+"/fifo.mp4"):
                print(f"The output directory {output_dir} already exists and has a file called fifo.mp4")
                continue
            else:
                print(f"The output directory {output_dir} already exists but does not have a file called fifo.mp4")

        # Copy conditioning image into output directory for easy comparison
        if conditioned_image_path and os.path.exists(conditioned_image_path):
            import shutil
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(conditioned_image_path, os.path.join(output_dir, "conditioned_image" + os.path.splitext(conditioned_image_path)[1]))

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
        # Encode conditioning image to VAE latent space (same space as pred_x0)
        cond_image_pil = Image.open(conditioned_image_path).convert("RGB")
        cond_image_rgb = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.CenterCrop((args.height, args.width)),
            transforms.ToTensor(),
        ])(cond_image_pil).unsqueeze(0).to("cuda")  # [1, 3, H, W] in [0,1]
        cond_image_rgb = cond_image_rgb * 2.0 - 1.0  # scale to [-1, 1] for VAE
        cond_image = model.encode_first_stage_2DAE(cond_image_rgb.unsqueeze(2))  # [1, 4, 1, h, w] latent

        ## inference
        is_run_base = not (os.path.exists(latents_dir+f"/{args.num_inference_steps}.pt") and os.path.exists(latents_dir+f"/0.pt"))
        if not is_run_base:
            ddim_sampler = DDIMSampler(model, experiment_condition=args.experiment_condition)
            ddim_sampler.make_schedule(ddim_num_steps=args.num_inference_steps, ddim_eta=args.eta, verbose=False)
        else:
            base_tensor, ddim_sampler, _ = base_ddim_sampling(model, cond, noise_shape, \
                                                args.num_inference_steps, args.eta, args.unconditional_guidance_scale, \
                                                latents_dir=latents_dir)
            save_gif(base_tensor, output_dir, "origin")
        # Enable profiling if requested
        if getattr(args, 'profile', False):
            ddim_sampler.profiler.enabled = True
        # Pass full prompt for concept attention token matching
        if args.experiment_condition == "concept_attention":
            ddim_sampler.set_prompt_for_concept_attention(prompt)
            ddim_sampler.set_concept_mask_strategy(
                strategy=args.concept_mask_strategy,
                threshold=args.concept_mask_threshold,
                topk_ratio=args.concept_mask_topk_ratio,
                std_scale=args.concept_mask_std_scale,
                max_components=args.concept_mask_max_components,
            )

        # Save original conditioning (before adding conditioned prompt)
        # for noise prediction blending in semantic mixing
        cond_original = {k: v if not isinstance(v, list) else list(v) for k, v in cond.items()}

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
            experiment_condition=args.experiment_condition,
            cond_original=cond_original,
            mixing_strength=args.mixing_strength,
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
    parser.add_argument("--prompt_file", "-p", type=str, default="datasets/DAVIS/davis_text_annotations/Davis16_annot2.txt", help="path to the prompt file")
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
    parser.add_argument("--output_fps", type=int, default=10, help="fps of the output video")
    parser.add_argument("--prompt_index", type=int, default=1, help="index of the prompt to run")
    parser.add_argument("--experiment_condition", type=str, default="bounding_box", help="Experiment condition")
    parser.add_argument("--profile", action="store_true", default=False, help="Enable profiling of denoising steps")
    parser.add_argument("--mixing_strength", type=float, default=0.3, help="Composable diffusion mixing strength (0=original, 1=full conditioned)")
    parser.add_argument("--gamma_override", type=float, default=None, help="Override gamma from prompt file")
    parser.add_argument("--concept_mask_strategy", type=str, default="relative_threshold",
                        choices=["relative_threshold", "topk", "mean_std", "largest_component", "absolute", "soft"],
                        help="How to convert concept attention into a mask")
    parser.add_argument("--concept_mask_threshold", type=float, default=0.3,
                        help="Threshold for relative_threshold/absolute concept masks")
    parser.add_argument("--concept_mask_topk_ratio", type=float, default=0.2,
                        help="Fraction of strongest attention pixels to keep for topk masks")
    parser.add_argument("--concept_mask_std_scale", type=float, default=1.0,
                        help="Std multiplier for mean_std concept masks")
    parser.add_argument("--concept_mask_max_components", type=int, default=1,
                        help="Connected components to keep for largest_component concept masks")

    args = parser.parse_args()

    args.num_inference_steps = args.video_length * args.num_partitions

    seed_everything(args.seed)

    main(args)