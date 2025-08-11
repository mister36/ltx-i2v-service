import os, tempfile, torch, gc
from typing import Optional, Tuple
from PIL import Image
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video

def _round_to_vae(height: int, width: int, pipe) -> Tuple[int, int]:
    ratio = getattr(pipe, "vae_spatial_compression_ratio", 32)
    return height - (height % ratio), width - (width % ratio)

class LTXService:
    def __init__(
        self,
        base_model: str = "Lightricks/LTX-Video-0.9.8-13B-distilled",
        upsampler_model: str = "Lightricks/ltxv-spatial-upscaler-0.9.7",
        device: str = "cuda",
        dtype = torch.bfloat16,
        enable_cpu_offload: bool = False,  # Disable for faster inference
        enable_sequential_cpu_offload: bool = False,  # For extreme memory savings
    ):
        self.device = device
        self.dtype = dtype
        self.is_distilled = "distilled" in base_model.lower()
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload
        
        print(f"Loading LTX models with CPU offload: {enable_cpu_offload}, Sequential offload: {enable_sequential_cpu_offload}")
        
        # Load pipelines
        self.pipe = LTXConditionPipeline.from_pretrained(base_model, torch_dtype=dtype)
        self.pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
            upsampler_model, vae=self.pipe.vae, torch_dtype=dtype
        )
        
        # Apply memory optimizations
        if enable_sequential_cpu_offload:
            # Most aggressive memory saving - keeps only active components on GPU
            print("Enabling sequential CPU offload for maximum memory efficiency")
            self.pipe.enable_sequential_cpu_offload()
            self.pipe_upsample.enable_sequential_cpu_offload()
        elif enable_cpu_offload:
            # Moderate memory saving - keeps some components on GPU
            print("Enabling CPU offload for memory efficiency")
            self.pipe.enable_model_cpu_offload()
            self.pipe_upsample.enable_model_cpu_offload()
        else:
            # Keep everything on GPU (original behavior)
            self.pipe.to(device)
            self.pipe_upsample.to(device)
            
        # Enable VAE tiling for lower memory usage
        self.pipe.vae.enable_tiling()
        
        # Enable attention slicing for lower memory usage
        self.pipe.enable_attention_slicing(1)
        self.pipe_upsample.enable_attention_slicing(1)
        
    def clear_memory(self):
        """Clear GPU memory cache and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Cleared GPU memory cache")

    def image_to_video(
        self,
        image_pil: Image.Image,
        prompt: str,
        *,
        negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
        expected_height: int = 480,
        expected_width: int = 704,  # Better default based on guide
        downscale_factor: float = 2/3,
        num_frames: int = 96,
        steps_lowres: int = 40,  # More steps for better quality
        steps_refine: int = 15,
        denoise_strength: float = 0.4,
        decode_timestep: float = 0.05,
        image_cond_noise_scale: float = 0.025,
        fps: int = 30,  # 30 FPS as recommended
        guidance_scale: float = 3.2,  # Recommended range 3-3.5
        seed: int = 0,
    ) -> str:
        # compress the single image into a 1-frame video (as in docs)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            export_to_video([image_pil], tmp_vid.name, fps=1)
            video = load_video(tmp_vid.name)

        condition = LTXVideoCondition(video=video, frame_index=0)

        # Part 1: generate at smaller resolution (rounded to VAE grid)
        h0 = int(expected_height * downscale_factor)
        w0 = int(expected_width * downscale_factor)
        h0, w0 = _round_to_vae(h0, w0, self.pipe)

        gen = torch.Generator(device=self.device).manual_seed(seed)
        
        # Distilled models don't need guidance_scale
        pipe_kwargs = {
            "conditions": [condition],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": w0,
            "height": h0,
            "num_frames": num_frames,
            "num_inference_steps": steps_lowres,

            "generator": gen,
            "output_type": "latent",
        }
        
        if not self.is_distilled:
            pipe_kwargs["guidance_scale"] = guidance_scale
            
        lowres_latents = self.pipe(**pipe_kwargs).frames

        # Part 2: latent upsample (2x spatial)
        up_latents = self.pipe_upsample(
            latents=lowres_latents,
            output_type="latent"
        ).frames

        # Part 3: light denoise/refine
        h_up, w_up = h0 * 2, w0 * 2
        # Refinement step
        refine_kwargs = {
            "conditions": [condition],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": w_up,
            "height": h_up,
            "num_frames": num_frames,
            "denoise_strength": denoise_strength,
            "num_inference_steps": steps_refine,

            "latents": up_latents,
            "decode_timestep": decode_timestep,
            "image_cond_noise_scale": image_cond_noise_scale,
            "generator": gen,
            "output_type": "pil",
        }
        
        if not self.is_distilled:
            refine_kwargs["guidance_scale"] = guidance_scale
            
        video_frames = self.pipe(**refine_kwargs).frames[0]

        # Part 4: resize to requested output resolution
        video_frames = [f.resize((expected_width, expected_height), Image.Resampling.LANCZOS)
                        for f in video_frames]

        # Write to a temp mp4, return path
        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        export_to_video(video_frames, out.name, fps=fps)
        
        # Clear memory after generation
        self.clear_memory()
        
        return out.name