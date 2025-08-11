# Memory Optimization Guide for LTX-I2V Service

## The Problem

You were encountering a **CUDA Out of Memory** error when starting the service on your 2x RTX 2000 Ada setup (15.7 GiB VRAM each). The error occurred because both the LTX Video Generation and InstantID services were loading simultaneously on startup, consuming ~15.6 GiB of memory and leaving insufficient space for additional allocations.

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB.
GPU 0 has a total capacity of 15.70 GiB of which 61.81 MiB is free.
Process 2052085 has 15.63 GiB memory in use.
```

## Solutions Implemented

### 1. **Lazy Loading** ✅

-   **Before**: Both services loaded during app startup
-   **After**: Services load only when their respective endpoints are called
-   **Benefit**: Reduces initial memory footprint and startup time

### 2. **CPU Offloading** ✅

-   **Model CPU Offload**: Moves inactive model components to CPU memory
-   **Sequential CPU Offload**: Most aggressive - keeps only active layers on GPU
-   **Benefit**: Can reduce GPU memory usage by 40-60%

### 3. **Memory Optimizations** ✅

-   **VAE Tiling**: Reduces memory for VAE operations
-   **Attention Slicing**: Reduces attention computation memory
-   **Automatic Cleanup**: Clears GPU cache after each generation

### 4. **Environment Variables** ✅

Configure memory optimization through environment variables:

```bash
# LTX Video Service Memory Settings
LTX_ENABLE_CPU_OFFLOAD=true              # Default: true
LTX_ENABLE_SEQUENTIAL_CPU_OFFLOAD=false  # Default: false (more aggressive)

# InstantID Service Memory Settings
INSTANTID_ENABLE_CPU_OFFLOAD=true              # Default: true
INSTANTID_ENABLE_SEQUENTIAL_CPU_OFFLOAD=false  # Default: false (more aggressive)
```

## Recommended Usage for Your Setup

### Option 1: Standard CPU Offload (Recommended)

```bash
# Use defaults - models offload inactive components to CPU
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Option 2: Maximum Memory Efficiency (If still having issues)

```bash
# Enable sequential offload for both services
export LTX_ENABLE_SEQUENTIAL_CPU_OFFLOAD=true
export INSTANTID_ENABLE_SEQUENTIAL_CPU_OFFLOAD=true
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Option 3: PyTorch Environment Optimization

```bash
# Add PyTorch memory management as suggested in error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Memory Usage Expectations

| Configuration      | Expected GPU Memory | Performance Impact |
| ------------------ | ------------------- | ------------------ |
| No Offloading      | ~15+ GiB            | Fastest            |
| CPU Offload        | ~8-10 GiB           | 10-20% slower      |
| Sequential Offload | ~4-6 GiB            | 20-40% slower      |

## What Changed in Your Code

1. **`app.py`**:

    - Removed startup model loading
    - Added lazy loading functions
    - Added environment variable support

2. **`ltx_service.py`**:

    - Added CPU offloading options
    - Added memory optimization features
    - Added automatic memory cleanup

3. **`instantid_service.py`**:
    - Added CPU offloading options
    - Added memory optimization features
    - Added automatic memory cleanup

## Troubleshooting

### If you still get OOM errors:

1. **Enable sequential offload**:

    ```bash
    export LTX_ENABLE_SEQUENTIAL_CPU_OFFLOAD=true
    export INSTANTID_ENABLE_SEQUENTIAL_CPU_OFFLOAD=true
    ```

2. **Use only one service at a time** by commenting out the unused service in `app.py`

3. **Reduce batch sizes** or image resolutions in your requests

4. **Monitor GPU memory**:
    ```bash
    watch -n 1 nvidia-smi
    ```

### Performance vs Memory Trade-offs

-   **Fastest**: No offloading (if you have enough VRAM)
-   **Balanced**: CPU offloading (recommended for your setup)
-   **Most Memory Efficient**: Sequential CPU offloading (slowest but uses least GPU memory)

## Next Steps

1. Start the service with the new optimizations
2. Test both endpoints to ensure they work
3. Monitor memory usage with `nvidia-smi`
4. Adjust offloading settings based on your performance needs

The lazy loading alone should solve your immediate problem, as only one service will load at a time, staying well within your 15.7 GiB VRAM limit.
