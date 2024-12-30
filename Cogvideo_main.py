import os
import torch
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from diffusers import CogVideoXPipeline, AutoencoderKLCogVideoX, CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np

from scripts.evaluation.funcs_cog import load_prompts
from scripts.evaluation.funcs_cog import fifo_sampling_cogvideo

# Set the directory of generated results
def set_directory(args, prompt):
    if args.output_dir is None:
        # Obtain the model name from the args
        model_name = args.model_name
        output_dir = f"results/{model_name}/random_noise/{prompt[:100]}"
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

    else:
        output_dir = args.output_dir

    latents_dir = f"results/{model_name}/latents/{args.num_inference_steps}steps/{prompt[:100]}/eta{args.eta}"

    print("The results will be saved in", output_dir)
    print("The latents will be saved in", latents_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)
    
    return output_dir, latents_dir

def main(args):
    ## step 1: model config
    # ================ CogvideoX ================
    # Load CogVideo components
    model_id = "THUDM/CogVideoX-5b"
    cog_text_pipeline = CogVideoXPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")
    cog_text_pipeline.enable_sequential_cpu_offload()
    cog_text_pipeline.vae.enable_tiling()
    cog_text_pipeline.vae.enable_slicing()

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.num_partitions > 0, "num_partitions must be greater than 0."

    ## step 2: load data
    ## -----------------------------------------------------------------
    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    prompt_list = load_prompts(args.prompt_file)
    num_samples = len(prompt_list)

    indices = list(range(num_samples))
    indices = indices[args.rank::args.num_processes]

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    for idx in indices:
        ## ======= CogVideoX =======
        prompt = prompt_list[idx]
        output_dir, latents_dir = set_directory(args, prompt)

        batch_size = 1
        # Prepare prompts and fps
        
        fps = torch.tensor([args.fps] * batch_size).to("cuda").long()

        # Encode the prompts into embeddings
        # Encode the prompts into embeddings
        positive_embeds, negative_embeds = cog_text_pipeline.encode_prompt(
            prompt,  # Pass the text prompt
            device="cuda",
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
        )

        # Use only the positive embeddings for conditioning
        cond = {
            "prompts": [positive_embeds],  # Use the positive embeddings
            "fps": fps                    # Frame rate
        }
        # Check if base video exists
        base_video_path = os.path.join(latents_dir, f"{args.num_inference_steps}.pt")
        base_output_path = os.path.join(output_dir, "base_video.mp4")
        is_run_base = not (os.path.exists(base_video_path) and os.path.exists(base_output_path))

        if is_run_base:
            print("Generating base video...")
            base_frames = cog_text_pipeline(prompt=prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=args.num_inference_steps).frames[0]
            print(f"The base frames are already being generated.")
            export_to_video(base_frames, base_output_path, fps=args.output_fps)

            # Check if base_frames is a list of images or a single image
            if isinstance(base_frames, list):
                # Flatten the list if it contains nested lists
                flat_base_frames = [frame for frame in base_frames]
            else:
                # Wrap the single image into a list for consistency
                flat_base_frames = [base_frames]

            # Convert frames to tensors and stack them
            transform = ToTensor()

            try:
                base_tensor = torch.stack([transform(frame) for frame in flat_base_frames], dim=2)  # [batch, channels, frames, height, width]
            except Exception as e:
                print(f"Error converting frames to tensors: {e}")
                import pdb; pdb.set_trace()  # Debugging point

            os.makedirs(latents_dir, exist_ok=True)
            torch.save(base_tensor, base_video_path)
            print(f"Base video tensor saved at: {base_video_path}")
        # Load the base video latents for further processing
        video_frames = fifo_sampling_cogvideo(
            args, cog_text_pipeline, cond, cfg_scale=args.unconditional_guidance_scale, 
            output_dir=output_dir, latents_dir=latents_dir, save_frames=args.save_frames
        )

        output_path = os.path.join(output_dir, "fifo_video.mp4")
        export_to_video(video_frames, output_path, fps=args.output_fps)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Essential Model and Configuration Arguments
    parser.add_argument("--model_name", type=str, default="Cogvideox-5B", help="Model name")
    parser.add_argument("--prompt_file", "-p", type=str, default="prompts/test_prompts.txt", help="Path to the prompt file")
    
    # Video Generation Parameters
    parser.add_argument("--video_length", type=int, default=16, help="Length of the video in frames")
    parser.add_argument("--new_video_length", "-l", type=int, default=100, help="Desired output video length")
    parser.add_argument("--height", type=int, default=320, help="Height of the output video")
    parser.add_argument("--width", type=int, default=512, help="Width of the output video")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output video")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")

    # Guidance and Denoising Options
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="Classifier-free guidance scale")
    parser.add_argument("--lookahead_denoising", "-ld", default=True, action="store_true", help="Enable lookahead denoising")
    parser.add_argument("--num_partitions", "-n", type=int, default=4, help="Number of partitions for diagonal denoising")
    parser.add_argument("--num_processes", type=int, default=1, help="number of processes if you want to run only the subset of the prompts")
    parser.add_argument("--rank", type=int, default=0, help="rank of the process(0~num_processes-1)")

    # Output Settings
    parser.add_argument("--save_frames", action="store_true", help="Save individual frames during generation")
    parser.add_argument("--output_fps", type=int, default=10, help="FPS of the output video")

    # Miscellaneous
    parser.add_argument("--eta", "-e", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=321, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Adjust dependent arguments
    args.num_inference_steps = args.video_length * args.num_partitions

    # Set random seed
    seed_everything(args.seed)

    # Run main function
    main(args)

