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

## 🎥 Video Results

### Qualitative Results

<div align="center">
<table>
<tr>
<td>
<img src="assets/results/mouse_cat.gif" width="400"/>
<p><b>Mouse mixed with Cat</b></p>
</td>
<td>
<img src="assets/results/cow_sheep.gif" width="400"/>
<p><b>Cow mixed with Sheep</b></p>
</td>
</tr>
<tr>
<td>
<img src="assets/results/bird_cat.gif" width="400"/>
<p><b>Bird mixed with Cat</b></p>
</td>
<td>
<img src="assets/results/horse_unicorn.gif" width="400"/>
<p><b>Horse mixed with Unicorn</b></p>
</td>
</tr>
<tr>
<td>
<img src="assets/results/surfer_kayak.gif" width="400"/>
<p><b>Surfer mixed with Kayak</b></p>
</td>
<td>
<img src="assets/results/astronaut_cat.gif" width="400"/>
<p><b>Astronaut mixed with Cat</b></p>
</td>
</tr>
</table>
</div>

### Quantitative Results

Our method achieves superior performance across multiple metrics both quantatively and qualitatively:

<div align="center">
<img src="assets/results/metric.png" width="800"/>
<img src="assets/results/user_study.png" width="800"/>
</div>

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
conda create -n moca python=3.10
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

The model can be run on a single NVIDIA V100 GPU with 32GB memory.

Thank for the community contrinutors for the code base:
[VideoCrafter Codebase](https://github.com/AILab-CVC/VideoCrafter) 
[Baseline FreeBlend Repo](https://github.com/WiserZhou/FreeBlend) 
[Metrics Repo](https://github.com/JunyaoHu/common_metrics_on_video_quality)
[FIFO Diffusion Pipeline](https://github.com/jjihwan/FIFO-Diffusion_public)
[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
[AnimateDiffV2V Pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/animatediff#animatediffvideotovideopipeline)

