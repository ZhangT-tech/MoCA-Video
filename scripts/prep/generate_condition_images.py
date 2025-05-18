import torch
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from diffusers import DiffusionPipeline
from compel import Compel
import os

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate missing conditioned images using Stable Diffusion")
    
    # Model and device setup
    parser.add_argument('--model_id', type=str, default="stabilityai/stable-diffusion-2-1", 
                       help='Pretrained model ID')
    parser.add_argument('--gpu_index', type=int, default=0, 
                       help='GPU index (set -1 for CPU)')
    parser.add_argument('--prompts_file', type=str, default="prompts/prompts.csv",
                       help='Path to the prompts CSV file')
    parser.add_argument('--assets_dir', type=str, default="assets",
                       help='Directory for saving generated images')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of diffusion steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Guidance scale')
    parser.add_argument('--num_images', type=int, default=1,
                       help='Number of images to generate per condition')
    
    return parser.parse_args()

def setup_pipeline(args):
    """Setup the Stable Diffusion pipeline."""
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() and args.gpu_index >= 0 else "cpu")
    pipeline = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    # Setup Compel for better prompt handling
    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    
    return pipeline, compel_proc, device

def get_missing_conditions(prompts_df, assets_dir):
    """Find which conditioned images are missing from assets directory."""
    missing_conditions = []
    for _, row in prompts_df.iterrows():
        image_path = row['conditioned_image_path']
        if not os.path.exists(image_path):
            # Extract the filename without 'assets/' prefix
            condition = image_path.replace('assets/', '').replace('.jpg', '').replace('.png', '')
            if condition not in missing_conditions:
                missing_conditions.append(condition)
    return missing_conditions

def generate_condition_image(pipeline, compel_proc, condition, args):
    """Generate an image for a specific condition."""
    # Create a detailed prompt for the condition
    prompt = f"A high quality photo of a {condition}, detailed, professional photography, 8k, realistic"
    
    # Generate embedding for the prompt
    embedding = compel_proc(prompt)
    
    # Generate the image
    image = pipeline(
        prompt_embeds=embedding,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    ).images[0]
    
    return image

def main():
    args = parse_args()
    
    # Setup pipeline
    pipeline, compel_proc, device = setup_pipeline(args)
    
    # Read prompts file
    prompts_df = pd.read_csv(args.prompts_file)
    
    # Create assets directory if it doesn't exist
    os.makedirs(args.assets_dir, exist_ok=True)
    
    # Get list of missing conditions
    missing_conditions = get_missing_conditions(prompts_df, args.assets_dir)
    print(f"Found {len(missing_conditions)} missing conditions: {missing_conditions}")
    
    # Generate images for each missing condition
    for condition in tqdm(missing_conditions, desc="Generating condition images"):
        for i in range(args.num_images):
            # Generate the image
            image = generate_condition_image(pipeline, compel_proc, condition, args)
            
            # Save the image
            save_path = os.path.join(args.assets_dir, f"{condition}.png")
            image.save(save_path)
            print(f"Generated and saved {save_path}")

if __name__ == "__main__":
    main() 