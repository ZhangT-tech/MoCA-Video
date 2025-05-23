# MoCA-Video: Motion-aware Concept Alignment for Video

<div align="center">

<p>
🚀 **Training-free** &nbsp;&nbsp;&nbsp;&nbsp; 🎨 **Semantic Mixing**
</p>

</div>

---

## 📽️ Teaser  
<!-- insert teaser GIF or static images here -->
[![Teaser Preview](assets/illustration/teaser.png)](assets/illustration/teaser.pdf)
---

## 🚀 Quick Start

### 1. Clone This Repository  

```bash
git clone https://github.com/your-username/MoCA-Video.git
cd MoCA-Video
```

### 2. Environment Setup

```bash
# Create and activate conda environment
conda create -n moca python=3.8
conda activate moca

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Pre-trained Models

Download the required pre-trained models and place them in the `checkpoints` directory:

```bash
mkdir -p checkpoints
# Download models from the provided links below
# Place them in the checkpoints directory
```

### 4. Run Inference

To run inference on your own videos:
```bash
python videocrafter_main.py \
```

Thank for the community contrinutors for the code base:
[VideoCrafter Codebase](https://github.com/AILab-CVC/VideoCrafter) 
[Baseline FreeBlend Repo](https://github.com/WiserZhou/FreeBlend) 
[Metrics Repo](https://github.com/JunyaoHu/common_metrics_on_video_quality)
[FIFO Diffusion Pipeline](https://github.com/jjihwan/FIFO-Diffusion_public)
[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)


