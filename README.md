# LTX I2V Service

FastAPI wrapper around LTX-Video 0.9.7 docs flow (low-res → latent upsample → refine).

**Note**: The InstantID functionality has been moved to a separate service. See the `realvisxl-instantid-service` folder for RealVisXL/InstantID image generation.

### Run locally (GPU)

```bash
pip install -r requirements.txt
export HUGGING_FACE_HUB_TOKEN=...
uvicorn app:app --host 0.0.0.0 --port 8000
```
