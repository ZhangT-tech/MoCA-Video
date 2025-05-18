import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import is_torch_xla_available
from typing import Optional, Union, List, Tuple, Dict, Any, Callable
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
import math
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import retrieve_timesteps
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXDPMScheduler
from tqdm import trange
from diffusers.utils import export_to_video


# if is_torch_xla_available():
#     import torch_xla.core.xla_model as xm

#     XLA_AVAILABLE = True
# else:
#     XLA_AVAILABLE = False
XLA_AVAILABLE = False


class CogVideoFIFOPipeline(CogVideoXImageToVideoPipeline):
    def __init__(
        self,
        tokenizer,
        text_encoder,
        vae,
        transformer,
        scheduler,
        image_processor=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler
            )

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        lookahead_denoising: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        num_partitions: int = 4,
        new_video_length: int = 16,
        video_length: int = 16,
        target_object: Optional[str] = None,
        mask: Optional[torch.FloatTensor] = None,
        cond_latents: Optional[torch.FloatTensor] = None,
        gamma: float = 0.1
    ) -> Union[CogVideoXPipelineOutput, Tuple]:

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1
        num_frames = (new_video_length + num_inference_steps)*self.vae_scale_factor_temporal

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        if lookahead_denoising:
            timesteps = torch.cat([torch.full((video_length//2,), timesteps[0], device=device), timesteps])
            num_inference_steps += video_length//2
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )

        latent_channels = self.transformer.config.in_channels // 2
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        ) # torch.Size([1, 13, 16, 60, 90])

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        
        # 8. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # Initialize video frames storage
        video_frames = []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            latents = latents[:, :(new_video_length + num_inference_steps - video_length)].clone()
            for i in trange(new_video_length + num_inference_steps - video_length, desc="fifo sampling"):
                for rank in reversed(range(2 * num_partitions if lookahead_denoising else num_partitions)):
                    start_idx = rank * (video_length // 2) if lookahead_denoising else 0
                    mid_idx = start_idx + video_length // 2
                    end_idx = start_idx + video_length
                    # Get current timestep batch
                    t = timesteps[start_idx:end_idx]
                    # Process current frame window
                    latent_model_input = torch.cat([latents[:,start_idx:end_idx]] * 2) if do_classifier_free_guidance else latents[:,start_idx:end_idx]
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t[0])                    

                    # Add image conditioning
                    latent_image_input = torch.cat([image_latents[:,start_idx:end_idx]] * 2) if do_classifier_free_guidance else image_latents
                    latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                    
                    image_rotary_emb = (
                        self._prepare_rotary_positional_embeddings(height, width, latent_model_input.size(1), device)
                        if self.transformer.config.use_rotary_positional_embeddings
                        else None
                    ) # torch.Size([17550, 64])

                    # Predict noise
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=t[0].expand(latent_model_input.shape[0]),
                        ofs=ofs_emb,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred.float() # torch.Size([2, 16, 16, 60, 90])

                    # Apply guidance
                    if use_dynamic_cfg:
                        self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t[0].item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Denoising step
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        if lookahead_denoising:
                            denoised_latents = self.scheduler.step(noise_pred, t[0], latents[:,start_idx:end_idx], **extra_step_kwargs, return_dict=False)[0]
                            latents[:,mid_idx:end_idx] = denoised_latents[:,video_length//2:]
                        else:
                            latents[:,start_idx:end_idx] = self.scheduler.step(noise_pred, t[0], latents[:,start_idx:end_idx], **extra_step_kwargs, return_dict=False)[0]
                    else:
                        if lookahead_denoising:
                            denoised_latents, old_pred_original_sample = self.scheduler.step(
                                noise_pred,
                                old_pred_original_sample,
                                t[0],
                                timesteps[i - 1] if i > 0 else None,
                                latents[:,start_idx:end_idx],
                                **extra_step_kwargs,
                                return_dict=False,
                            )
                            latents[:,mid_idx:end_idx] = denoised_latents[:,video_length//2:]
                        else:
                            latents[:,:,start_idx:end_idx], old_pred_original_sample = self.scheduler.step(
                                noise_pred,
                                old_pred_original_sample,
                                t[0],
                                timesteps[i - 1] if i > 0 else None,
                                latents[:,start_idx:end_idx],
                                **extra_step_kwargs,
                                return_dict=False,
                            )

                # Extract completed frame if we've done enough denoising steps
                # reconstruct from latent to pixel space
                # first_frame_idx = video_length//2 if lookahead_denoising else 0 # 2
                frame = self.decode_latents(latents[:, [0]])
                video_frames.append(self.video_processor.postprocess_video(video=frame, output_type=output_type))
                

                # Shift latents and add new noise
                latents = self._shift_latents(latents, generator)
                
                # Handle callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t[0], callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                progress_bar.update()

        # Combine frames and return
        video = torch.cat(video_frames, dim=2)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)

    def _shift_latents(self, latents: torch.FloatTensor, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        """Shift latents by moving frames forward and adding new noise at the end"""
        latents[:,:,:-1] = latents[:,:,1:].clone()
        latents[:,:,-1] = torch.randn(
            latents[:,:,-1].shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype
        )
        return latents

    