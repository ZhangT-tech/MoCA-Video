import cv2
import os
from pathlib import Path
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert video to images")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", type=str, default="output_frames", help="Directory to save extracted frames")
    parser.add_argument("--frame_rate", type=int, default=1, help="Extract one frame every N frames (default: 1)")
    parser.add_argument("--image_format", type=str, default="jpg", help="Output image format (default: jpg)")
    return parser.parse_args()

def video_to_images(video_path: str, output_dir: str, frame_rate: int = 1, image_format: str = "jpg"):
    """
    Convert video to images by extracting frames.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames
        frame_rate (int): Extract one frame every N frames
        image_format (str): Output image format
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties:")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    
    frame_count = 0
    saved_count = 0
    
    # Process video frames
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_rate == 0:
                # Save frame as image
                output_file = output_path / f"frame_{saved_count:06d}.{image_format}"
                cv2.imwrite(str(output_file), frame)
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    # Release video capture
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"Saved {saved_count} frames to {output_dir}")

def main():
    args = parse_args()
    video_to_images(
        video_path=args.video_path,
        output_dir=args.output_dir,
        frame_rate=args.frame_rate,
        image_format=args.image_format
    )

if __name__ == "__main__":
    main() 