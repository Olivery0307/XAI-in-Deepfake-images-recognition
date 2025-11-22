# XAI Auto-Switch Implementation Summary

## Overview
Successfully implemented automatic switching between **Attention Rollout** (for ViT models) and **Grad-CAM** (for CNN models) in the Deepfake Detection app.

## Changes Made

### 1. `src/xai_utils.py`

#### Added Functions:
- **`compute_attention_rollout()`** (lines 230-273)
  - Core attention rollout algorithm from ViT notebook
  - Averages attention across all heads
  - Discards lowest 90% of attention values (configurable)
  - Multiplies attention matrices across layers
  - Returns attention map for CLS token

- **`visualize_attention_rollout_vit()`** (lines 276-354)
  - Full implementation replacing the placeholder
  - Extracts attention weights from ViT model
  - Computes attention rollout
  - Reshapes to image dimensions
  - Creates visualization overlay (same format as Grad-CAM)
  - Supports original image size resizing

- **`compute_xai_visualization()`** (lines 357-413)
  - **NEW unified function** that auto-switches:
    - **ViT models** → Attention Rollout
    - **ResNet/EfficientNet** → Grad-CAM
  - Fallback: if Attention Rollout fails, uses Grad-CAM
  - Returns which method was used for UI display

### 2. `app.py`

#### Updated Imports:
```python
from src.xai_utils import compute_xai_visualization, get_gemini_explanation
```
(Changed from `compute_gradcam` to `compute_xai_visualization`)

#### Updated XAI Generation (lines 331-346):
```python
# Generate XAI visualization (auto-switches between Grad-CAM and Attention Rollout)
try:
    heatmap_img, heatmap_raw, xai_method = compute_xai_visualization(
        model=model,
        input_tensor=input_tensor,
        model_name=model_name,
        target_layer=None,
        original_size=image.size
    )
    heatmap_pil = Image.fromarray(heatmap_img.astype('uint8'))
    xai_success = True
except Exception as e:
    st.warning(f"⚠️ XAI visualization generation failed: {e}")
    heatmap_pil = None
    xai_success = False
    xai_method = "Unknown"
```

#### Updated UI Display (lines 378-386):
- Now shows which method was used: `"Grad-CAM"` or `"Attention Rollout"`
- Caption dynamically displays the XAI method used
- Example: `"🔥 Exhibit B: AI Attention Heatmap (Attention Rollout)"`

#### Updated Gemini Report (lines 388-420):
- Changed all `gradcam_success` to `xai_success`
- Report header shows which XAI method was used
- Example: `"🔍 Forensic Analysis Report (via Attention Rollout)"`

#### Updated About Section (lines 282-286):
```
Features:
- Multi-architecture support (ResNet, EfficientNet, ViT)
- XAI visualization (Grad-CAM for CNNs, Attention Rollout for ViT)
- AI-powered forensic analysis (Gemini)
```

## How It Works

### Automatic Detection:
1. When user selects a **ViT model** (vit_b_16, vit_b_32):
   - `compute_xai_visualization()` detects 'vit' in model name
   - Uses **Attention Rollout** (native to transformers)
   - Returns method name: `"Attention Rollout"`

2. When user selects a **CNN model** (resnet34, resnet50, efficientnet_b0, efficientnet_b4):
   - `compute_xai_visualization()` detects non-vit model
   - Uses **Grad-CAM** (gradient-based)
   - Returns method name: `"Grad-CAM"`

### Fallback Mechanism:
- If Attention Rollout fails for any reason, automatically falls back to Grad-CAM
- Ensures the app always provides some visualization

## Benefits

1. **Architecture-appropriate XAI**:
   - ViT models use their native attention mechanism
   - CNN models use gradient-based CAM

2. **Transparent to Users**:
   - UI shows which method was used
   - No manual selection needed

3. **Robust**:
   - Fallback mechanism prevents failures
   - Consistent API across both methods

4. **Better Explanations**:
   - Attention Rollout is more natural for ViT (shows what patches the model attended to)
   - Grad-CAM remains optimal for CNNs (shows gradient-weighted feature maps)

## Testing Recommendations

1. Test with ResNet model → should show "Grad-CAM"
2. Test with ViT model → should show "Attention Rollout"
3. Verify heatmaps look different between the two methods
4. Check Gemini report mentions the correct XAI method

## Files Modified
- `src/xai_utils.py` - Added Attention Rollout implementation and unified function
- `app.py` - Updated to use auto-switching XAI visualization
- Variable name changes: `gradcam_success` → `xai_success` throughout
