import os
import subprocess
from pathlib import Path

def convert_gif_to_mp4(gif_path):
    """Convert a GIF file to MP4 using ffmpeg"""
    mp4_path = str(Path(gif_path).with_suffix('.mp4'))
    
    # Use ffmpeg to convert GIF to MP4
    cmd = [
        "ffmpeg",
        "-i", gif_path,
        "-movflags", "faststart",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
        "-preset", "ultrafast",  # Use fastest encoding preset
        "-y",  # Overwrite output file if it exists
        mp4_path
    ]
    
    try:
        print(f"Converting {gif_path} to {mp4_path}...")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Successfully converted {gif_path} to {mp4_path}")
        return mp4_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {gif_path}: {e.stderr.decode() if e.stderr else str(e)}")
        return None

def main():
    # Directory containing the GIF files
    base_dir = "/ibex/user/zhant0g/code/MoCA-Video/user-study-samples/An astronaut floating in space, high quality, 4K resolution"
    
    # Convert all GIF files in the directory
    for file in os.listdir(base_dir):
        if file.lower().endswith('.gif'):
            gif_path = os.path.join(base_dir, file)
            convert_gif_to_mp4(gif_path)

if __name__ == "__main__":
    main() 