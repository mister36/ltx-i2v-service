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
    base = os.getenv("BASE_MODEL_ID", "Lightricks/LTX-Video-0.9.7-distilled")
    up   = os.getenv("UPSAMPLER_ID", "Lightricks/ltxv-spatial-upscaler-0.9.7")
    svc = LTXService(base_model=base, upsampler_model=up)

@app.post("/i2v")
async def i2v(
    image: UploadFile,
    prompt: str = Form(...),
    negative_prompt: str = Form("worst quality, inconsistent motion, blurry, jittery, distorted"),
    expected_height: int = Form(480),
    expected_width: int  = Form(832),
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