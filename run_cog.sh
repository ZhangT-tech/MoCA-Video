#!/bin/bash --login
#SBATCH --job-name=VideoInpainting
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=90GB
#SBATCH -o .logs/%x.%A.%a.out
#SBATCH -e .logs/%x.%A.%a.err
#SBATCH --array=0
#SBATCH --account=conf-icml-2025.01.31-ghanembs

# Activate the conda environment
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

source /ibex/user/zhant0g/mambaforge/bin/activate
conda activate videoinpainting
module load cuda/11.8  

# Ensure logs directory exists
if [ ! -d ".logs" ]; then
    mkdir -p .logs
fi

# Run the Python script
echo "Running Cogvideo_main.py..."
python Cogvideo_main.py --prompt_file="prompts/test_prompts.txt" --video_length=16 --new_video_length=20

echo "Job completed!"
