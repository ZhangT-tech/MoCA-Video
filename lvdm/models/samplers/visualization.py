import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.utils as vutils
import math

class VisualizationHelper:
    @staticmethod
    def visualize_mask_and_latent(mask, latent, timestep, frame_idx, save_dir):
        """Visualize the mask and latent during denoising process"""
        # Create a subdirectory for this timestep
        step_dir = os.path.join(save_dir, f'timestep_{timestep:04d}')
        os.makedirs(step_dir, exist_ok=True)
        
        # Process mask for visualization
        mask_vis = mask[0, 0, 0].cpu().numpy()  # Take first channel of first batch
        
        # Process latent for visualization
        # Normalize latent to [0, 1] range for visualization
        latent_vis = latent[0, :3, 0].cpu()  # Take first 3 channels of first batch
        latent_vis = (latent_vis - latent_vis.min()) / (latent_vis.max() - latent_vis.min())
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot mask
        mask_img = ax1.imshow(mask_vis, cmap='hot')
        ax1.set_title(f'Mask (Frame {frame_idx})')
        plt.colorbar(mask_img, ax=ax1)
        
        # Plot latent
        latent_img = ax2.imshow(latent_vis.permute(1, 2, 0).numpy())
        ax2.set_title(f'Latent (Frame {frame_idx})')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, f'frame_{frame_idx:03d}.png'))
        plt.close()
        
        # Also save the raw tensors for further analysis
        torch.save(mask, os.path.join(step_dir, f'mask_{frame_idx:03d}.pt'))
        torch.save(latent, os.path.join(step_dir, f'latent_{frame_idx:03d}.pt'))

    @staticmethod
    def visualize_sampling(pred_x0, noise, save_dir, step, is_manipulated=False):
        """Visualize the sampling process"""
        # Create step subfolder with manipulation status
        status = "after_manipulation" if is_manipulated else "before_manipulation"
        step_dir = os.path.join(save_dir, f'step_{step:03d}_{status}')
        os.makedirs(step_dir, exist_ok=True)
        
        # Function to process tensor for visualization
        def process_for_vis(tensor):
            # Move to CPU, take first batch and normalize to [0,1]
            vis = tensor[0].detach().cpu()  # Remove batch dim
            vis = (vis - vis.min()) / (vis.max() - vis.min())  # Normalize
            return vis

        # Save each frame
        grid = torch.stack([
            process_for_vis(pred_x0[:,:,0]),
            process_for_vis(noise[:,:,0])
        ])
        
        # Save grid in step subfolder
        vutils.save_image(
            grid,
            os.path.join(step_dir, f'frame_{0:03d}.png'),
            nrow=2,
            normalize=False
        )

    @staticmethod
    def visualize_object_attention(pred_image, cond_image, attention_mask, attention_map, 
                                 labeled_regions, target_object, save_dir, step):
        """Visualize attention and region detection"""
        step_dir = os.path.join(save_dir, f'step_{step:03d}_object_attention')
        os.makedirs(step_dir, exist_ok=True)
        
        # Convert tensors to numpy
        pred_image = pred_image.cpu().numpy().transpose(1, 2, 0)[:, :, :3]
        cond_image = cond_image.cpu().numpy().transpose(1, 2, 0)[:, :, :3]
        attention_mask = attention_mask.cpu().numpy()
        attention_map = attention_map.cpu().numpy()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original images
        axes[0, 0].imshow(pred_image)
        axes[0, 0].set_title('Generated Image')
        axes[0, 1].imshow(cond_image)
        axes[0, 1].set_title('Conditioning Image')
        
        # Raw attention map
        attention_vis = axes[0, 2].imshow(attention_map, cmap='hot')
        axes[0, 2].set_title('Raw Attention Map')
        plt.colorbar(attention_vis, ax=axes[0, 2])
        
        # Connected components
        if labeled_regions is not None:
            axes[1, 0].imshow(labeled_regions.cpu().numpy(), cmap='nipy_spectral')
            axes[1, 0].set_title('Connected Components')
        
        # Final mask
        axes[1, 1].imshow(attention_mask, cmap='hot')
        axes[1, 1].set_title(f'Mask for {target_object}' if target_object else 'Overall Mask')
        
        # Masked result
        masked_image = pred_image.copy()
        masked_image[attention_mask > 0.5] = cond_image[attention_mask > 0.5]
        axes[1, 2].imshow(masked_image)
        axes[1, 2].set_title('Masked Result')
        
        plt.savefig(os.path.join(step_dir, f'object_attention_{target_object}.png'))
        plt.close()

    @staticmethod
    def visualize_masks(masks, save_dir, step):
        """Visualize the segmentation masks"""
        # Create masks subfolder in step directory
        masks_dir = os.path.join(save_dir, f'step_{step:03d}_masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        # Convert masks to tensor if they're numpy arrays
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks).float()
        
        # Process each mask
        for i, mask in enumerate(masks):
            # Convert to numpy, scale to [0, 255], and save directly as PIL Image
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np)
            mask_pil.save(os.path.join(masks_dir, f'mask_{i:03d}.png')) 

    @staticmethod
    def visualize_latents(latents, save_dir):
        """Visualize the latents"""
        # Create latents subfolder in save_dir
        if type(latents) == torch.Tensor:
            latents_dir = os.path.join(save_dir, 'latents')
            os.makedirs(latents_dir, exist_ok=True)
        # Convert latents to numpy and normalize to [0, 1]
        latents_np = latents.cpu().numpy()
        latents_np = (latents_np - latents_np.min()) / (latents_np.max() - latents_np.min())

        # Save each frame as a separate PNG file
        for i in range(latents_np.shape[2]):
            frame = latents_np[:, :, i].squeeze(0).transpose(1, 2, 0)
            frame = (frame + 1.0) / 2.0  # Normalize to [0, 1]
            frame_pil = Image.fromarray((frame * 255).astype(np.uint8))
            frame_pil.save(os.path.join(latents_dir, f'frame_{i:03d}.png'))
        
        