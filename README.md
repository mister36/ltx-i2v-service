# LTX I2V Service

FastAPI wrapper around LTX-Video 0.9.8 docs flow (low-res → latent upsample → refine).

### Run locally (GPU)

```bash
pip install -r requirements.txt
export HUGGING_FACE_HUB_TOKEN=...
uvicorn app:app --host 0.0.0.0 --port 8000
```
