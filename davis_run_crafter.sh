#!/bin/bash --login
#SBATCH --job-name=DAVISVideoInpainting
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=90GB
#SBATCH -o .logs/%x.%A.%a.out
#SBATCH -e .logs/%x.%A.%a.err
#SBATCH --array=0-49
#SBATCH --account conf-neurips-2025.05.22-ghanembs

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

# Get list of DAVIS cases
DAVIS_CASES=(
    "blackswan" "bmx-bumps" "car-roundabout" "car-turn" "dance-twirl" "dog" "drift-turn" "elephant" "flamingo" "goat"
    "hockey" "horsejump-high" "kite-surf" "kite-walk" "mallard-fly" "mallard-water" "motocross-bumps" "motocross-jump"
    "motorbike" "paragliding-launch" "parkour" "rhino" "scooter-gray" "soapbox" "stroller" "bear" "bmx-trees" "boat"
    "breakdance" "breakdance-flare" "bus" "camel" "car-shadow" "cows" "dance-jump" "dog-agility" "drift-chicane"
    "drift-straight" "hike" "horsejump-low" "libby" "lucia" "paragliding" "rollerblade" "scooter-black" "soccerball"
    "surf" "swing" "tennis" "train"
)

# Get the case for this array job
SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID + 0))
CASE=${DAVIS_CASES[$SLURM_ARRAY_TASK_ID]}

# Create output directory for this case
OUTPUT_DIR="outputs/with_prompt/davis_${CASE}"
mkdir -p $OUTPUT_DIR

# Run the Python script with DAVIS parameters
echo "Processing DAVIS case: $CASE"

python videocrafter_main.py \
    --use_davis \
    --davis_root="DAVIS" \
    --video_name="$CASE" \
    --sampling_strategy="first" \
    --video_length=16 \
    --new_video_length=100 \
    --fps=8 \
    --unconditional_guidance_scale=12.0 \
    --lookahead_denoising \
    --eta=1.0 \
    --use_mp4 \
    --output_fps=10 \
    --output_dir="$OUTPUT_DIR"

echo "Job completed for case: $CASE!"
