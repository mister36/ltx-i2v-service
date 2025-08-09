import io, os
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
from PIL import Image
from ltx_service import LTXService

app = FastAPI()
svc = None

@app.on_event("startup")
def _load():
    global svc
    base = os.getenv("BASE_MODEL_ID", "Lightricks/LTX-Video-0.9.7-dev")
    up   = os.getenv("UPSAMPLER_ID", "Lightricks/ltxv-spatial-upscaler-0.9.7")
    svc = LTXService(base_model=base, upsampler_model=up)

@app.post("/i2v")
async def i2v(
    image: UploadFile,
    prompt: str = Form(...),
    negative_prompt: str = Form("worst quality, inconsistent motion, blurry, jittery, distorted"),
    expected_height: int = Form(None),
    expected_width: int  = Form(None),
    max_dimension: int = Form(832),  # Maximum dimension for scaling
    downscale_factor: float = Form(2/3),
    num_frames: int = Form(96),
    steps_lowres: int = Form(30),
    steps_refine: int = Form(10),
    denoise_strength: float = Form(0.4),
    decode_timestep: float = Form(0.05),
    image_cond_noise_scale: float = Form(0.025),
    fps: int = Form(24),
    seed: int = Form(0),
):
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    
    # Calculate aspect ratio preserving dimensions if not explicitly provided
    if expected_height is None or expected_width is None:
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height
        
        # Scale to fit within max_dimension while preserving aspect ratio
        if original_width > original_height:
            # Landscape or square
            expected_width = max_dimension
            expected_height = int(max_dimension / aspect_ratio)
        else:
            # Portrait
            expected_height = max_dimension
            expected_width = int(max_dimension * aspect_ratio)
        
        # Round to multiples of 8 for better model compatibility
        expected_width = (expected_width // 8) * 8
        expected_height = (expected_height // 8) * 8
    mp4_path = svc.image_to_video(
        img, prompt,
        negative_prompt=negative_prompt,
        expected_height=expected_height,
        expected_width=expected_width,
        downscale_factor=downscale_factor,
        num_frames=num_frames,
        steps_lowres=steps_lowres,
        steps_refine=steps_refine,
        denoise_strength=denoise_strength,
        decode_timestep=decode_timestep,
        image_cond_noise_scale=image_cond_noise_scale,
        fps=fps,
        seed=seed,
    )
    def _stream():
        with open(mp4_path, "rb") as f:
            yield from f
        os.remove(mp4_path)
    return StreamingResponse(_stream(), media_type="video/mp4")