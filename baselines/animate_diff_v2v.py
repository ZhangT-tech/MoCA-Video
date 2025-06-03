import imageio
import requests
import torch
from diffusers import AnimateDiffVideoToVideoPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from io import BytesIO
from PIL import Image
import os

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# helper function to load videos
def load_video(file_path: str):
    images = []

    if file_path.startswith(('http://', 'https://')):
        # If the file_path is a URL
        response = requests.get(file_path)
        response.raise_for_status()
        content = BytesIO(response.content)
        vid = imageio.get_reader(content)
    else:
        # Assuming it's a local file path
        vid = imageio.get_reader(file_path)

    for frame in vid:
        pil_image = Image.fromarray(frame)
        # Resize the image to 512x512
        pil_image = pil_image.resize((512, 512), Image.Resampling.LANCZOS)
        images.append(pil_image)

    # Print dimensions of first frame
    if images:
        print(f"Input video frame size: {images[0].size}")
    return images

# --- Begin dynamic video/prompt selection ---
ROOT_DIR = "results/videocraft/sam2/random_noise"

# Get all prompt folders
prompt_folders = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
prompt_folders.sort()  # for reproducibility

for prompt_folder in prompt_folders:
    try:
        prompt_folder_path = os.path.join(ROOT_DIR, prompt_folder)
        print(f"\nProcessing folder: {prompt_folder}")

        # Check for subfolders (object)
        subfolders = [d for d in os.listdir(prompt_folder_path) if os.path.isdir(os.path.join(prompt_folder_path, d))]
        if subfolders:
            for subfolder in subfolders:
                try:
                    object_name = subfolder.split(".")[0]  # remove file extension if present
                    video_path = os.path.join(prompt_folder_path, subfolder, "origin.gif")
                    
                    if not os.path.exists(video_path):
                        print(f"Skipping {video_path} - file not found")
                        continue
                        
                    print(f"Processing video: {video_path}")
                    prompt = f"a video of {object_name}"
                    video = load_video(video_path)
                    video = video[:32]
                    
                    output = pipe(
                        video=video,
                        prompt=prompt,
                        negative_prompt="bad quality, worse quality",
                        guidance_scale=7.5,
                        num_inference_steps=25,
                        strength=0.5,
                        generator=torch.Generator("cpu").manual_seed(42),
                    )
                    frames = output.frames[0]
                    # Resize output frames to 512x512
                    frames = [frame.resize((512, 512), Image.Resampling.LANCZOS) for frame in frames]
                    # Print dimensions of first output frame
                    if frames:
                        print(f"Output video frame size: {frames[0].size}")
                    output_dir = os.path.dirname(video_path)
                    output_gif = os.path.join(output_dir, "animation.gif")
                    export_to_gif(frames, output_gif)
                    print(f"Saved animation to {output_gif}")
                    
                except Exception as e:
                    print(f"Error processing subfolder {subfolder}: {str(e)}")
                    continue
        else:
            # Handle case with no subfolders
            try:
                object_name = "cat"
                video_path = os.path.join(prompt_folder_path, "origin.gif")
                
                if not os.path.exists(video_path):
                    print(f"Skipping {video_path} - file not found")
                    continue
                    
                print(f"Processing video: {video_path}")
                prompt = f"a video of {object_name}"
                video = load_video(video_path)
                video = video[:32]
                
                output = pipe(
                    video=video,
                    prompt=prompt,
                    negative_prompt="bad quality, worse quality",
                    guidance_scale=7.5,
                    num_inference_steps=25,
                    strength=0.5,
                    generator=torch.Generator("cpu").manual_seed(42),
                )
                frames = output.frames[0]
                # Resize output frames to 512x512
                frames = [frame.resize((512, 512), Image.Resampling.LANCZOS) for frame in frames]
                # Print dimensions of first output frame
                if frames:
                    print(f"Output video frame size: {frames[0].size}")
                output_dir = os.path.dirname(video_path)
                output_gif = os.path.join(output_dir, "animation.gif")
                export_to_gif(frames, output_gif)
                print(f"Saved animation to {output_gif}")
                
            except Exception as e:
                print(f"Error processing folder {prompt_folder}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error processing main folder {prompt_folder}: {str(e)}")
        continue

print("\nProcessing completed!")
# --- End dynamic video/prompt selection ---