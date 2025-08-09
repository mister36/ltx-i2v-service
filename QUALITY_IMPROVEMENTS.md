# Quality Improvements Applied

Based on the LTX-Video official guide, the following improvements have been implemented to enhance video quality:

## 🚀 Model Upgrade

-   **Upgraded to LTX-Video-0.9.8-dev** (from 0.9.7-dev)
-   Latest model includes "improved prompt understanding and detail generation"
-   Better physical understanding and prompt adherence

## 📝 Prompt Enhancement

-   **Added automatic prompt enhancement** with `enhance_prompt=True`
-   This feature automatically improves user prompts for better results
-   Can be toggled via API parameter

## ⚙️ Optimized Parameters

-   **Guidance Scale**: Set to 3.2 (within recommended range of 3-3.5)
-   **Inference Steps**: Increased to 40+15 total (was 30+10) for better quality
-   **FPS**: Default changed to 30 FPS (was 24) as recommended
-   **Resolution**: Lowered max dimension to 704 (was 832) - model works best under 720×1280

## 🎯 Resolution Optimization

-   **VAE Alignment**: Round to multiples of 32 (was 8) for better VAE compatibility
-   **Minimum Resolution**: Ensure at least 256×256 to avoid quality degradation
-   Better aspect ratio preservation

## ⚡ Model Variants

-   **Added distilled model support** for faster generation
-   Two options available:
    -   `"dev"`: Highest quality (default)
    -   `"distilled"`: Faster generation with slight quality trade-off
-   Distilled models automatically skip guidance scale (as recommended)

## 🔧 Technical Improvements

-   Proper parameter handling for distilled vs full models
-   Better error handling and model selection
-   Environment variable support for easy model switching

## Usage

### High Quality (slower):

```bash
curl -X POST "http://localhost:8000/i2v" \
  -F "image=@your_image.jpg" \
  -F "prompt=your detailed prompt" \
  -F "model_variant=dev"
```

### Fast Generation:

```bash
curl -X POST "http://localhost:8000/i2v" \
  -F "image=@your_image.jpg" \
  -F "prompt=your detailed prompt" \
  -F "model_variant=distilled"
```

### Environment Variables

```bash
export BASE_MODEL_ID="Lightricks/LTX-Video-0.9.8-dev"
export DISTILLED_MODEL_ID="Lightricks/LTX-Video-0.9.8-13B-distilled"
export UPSAMPLER_ID="Lightricks/ltxv-spatial-upscaler-0.9.7"
```

## Expected Quality Improvements

-   Better prompt following and detail generation
-   Improved motion consistency
-   More realistic physics and object interactions
-   Better handling of complex scenes
-   Faster iteration with distilled model option
