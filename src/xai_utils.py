"""
Explainable AI (XAI) Utilities
Grad-CAM visualization and Gemini API integration for forensic analysis
"""

import torch
import torch.nn as nn
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import cv2
from typing import Optional, Tuple, List
import google.generativeai as genai
import io
import base64
import time


def get_reshape_transform_vit(model_name: str):
    """
    Get reshape transform function for ViT models (required for Grad-CAM)

    Args:
        model_name: Name of the ViT model

    Returns:
        Reshape transform function
    """
    def reshape_transform(tensor, height=14, width=14):
        """
        Reshape ViT output for Grad-CAM compatibility

        ViT outputs shape: (batch_size, num_tokens, embedding_dim)
        Need to reshape to: (batch_size, embedding_dim, height, width)
        """
        # Remove CLS token (first token)
        result = tensor[:, 1:, :]

        # Get dimensions
        batch_size, num_tokens, embedding_dim = result.shape

        # Calculate grid size
        # For ViT-B/16 with 224x224 input: 224/16 = 14x14 patches
        grid_size = int(np.sqrt(num_tokens))

        # Reshape to spatial format
        result = result.reshape(batch_size, grid_size, grid_size, embedding_dim)

        # Permute to (batch_size, embedding_dim, height, width)
        result = result.permute(0, 3, 1, 2)

        return result

    return reshape_transform


def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    model_name: str,
    target_layer: Optional[nn.Module] = None,
    target_category: Optional[int] = None,
    original_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Grad-CAM heatmap for input image

    Args:
        model: Trained model
        input_tensor: Input image tensor (1, 3, H, W) - MUST be normalized
        model_name: Name of model (for architecture-specific handling)
        target_layer: Target layer for Grad-CAM (auto-selected if None)
        target_category: Target class index (uses predicted class if None)
        original_size: Optional (width, height) to resize heatmap to original image size

    Returns:
        cam_image: Grad-CAM visualization as RGB numpy array (0-255)
        heatmap: Raw heatmap as numpy array (0-1)
    """

    model.eval()
    model_name = model_name.lower()

    # Auto-select target layer if not provided
    if target_layer is None:
        from .models import get_target_layer
        target_layer = get_target_layer(model, model_name)

    # Prepare target layers for GradCAM
    target_layers = [target_layer]

    # Handle ViT models with reshape transform
    if 'vit' in model_name:
        reshape_transform = get_reshape_transform_vit(model_name)
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform
        )
    else:
        # ResNet and EfficientNet
        cam = GradCAM(
            model=model,
            target_layers=target_layers
        )

    # Generate CAM
    # targets=None means use predicted class
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)

    # grayscale_cam shape: (batch_size, height, width)
    # Take first image from batch
    grayscale_cam = grayscale_cam[0, :]

    # Denormalize input tensor for visualization
    from .preprocessing import denormalize_image
    denorm_tensor = denormalize_image(input_tensor.squeeze(0).cpu())

    # Convert to numpy (H, W, C) in range [0, 1]
    # IMPORTANT: show_cam_on_image expects float32 numpy array in [0, 1]
    rgb_img = denorm_tensor.permute(1, 2, 0).numpy()
    rgb_img = np.float32(rgb_img)  # Ensure float32 type

    # Ensure grayscale_cam is also float32
    grayscale_cam = np.float32(grayscale_cam)

    # If original_size is provided, resize to match original image
    if original_size is not None:
        import cv2
        width, height = original_size

        # Resize rgb_img to original size
        rgb_img_resized = cv2.resize(rgb_img, (width, height), interpolation=cv2.INTER_LINEAR)

        # Resize grayscale_cam to original size
        grayscale_cam_resized = cv2.resize(grayscale_cam, (width, height), interpolation=cv2.INTER_LINEAR)

        # Create CAM visualization at original size
        cam_image = show_cam_on_image(rgb_img_resized, grayscale_cam_resized, use_rgb=True)

        return cam_image, grayscale_cam_resized
    else:
        # Create CAM visualization at model input size (224x224)
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return cam_image, grayscale_cam


def get_gemini_explanation(
    api_key: str,
    original_image: Image.Image,
    heatmap_image: np.ndarray,
    prediction: str,
    confidence: float,
    model_name: str = "gemini-2.5-flash",
    timeout: int = 30
) -> str:
    """
    Get forensic analysis from Gemini AI based on Grad-CAM heatmap

    Args:
        api_key: Gemini API key
        original_image: Original PIL image
        heatmap_image: Grad-CAM visualization (numpy array RGB)
        prediction: Prediction label ('Real' or 'Fake')
        confidence: Prediction confidence (0-1)
        model_name: Gemini model name
        timeout: Request timeout in seconds

    Returns:
        Analysis text from Gemini
    """

    if not api_key:
        return "⚠ Gemini API key not provided. Please set GEMINI_API_KEY to enable AI analysis."

    try:
        # Configure Gemini API
        genai.configure(api_key=api_key)

        # Convert heatmap numpy array to PIL Image
        heatmap_pil = Image.fromarray(heatmap_image.astype('uint8'))

        # Create forensic analysis prompt
        prompt = f"""You are a forensic image analyst specializing in deepfake detection.

The AI model has classified this image as **{prediction}** with {confidence*100:.1f}% confidence.

The heatmap overlay shows regions that influenced the model's decision:
- RED/YELLOW regions: Areas the model focused on most
- GREEN/BLUE regions: Less influential areas

**Your Task:**
Analyze the heatmap and provide a detailed forensic report covering:

1. **Focus Areas**: Which facial features or regions did the model focus on? (eyes, nose, mouth, skin texture, edges, background, etc.)

2. **Potential Artifacts**: Do the highlighted regions correspond to known deepfake artifacts?
   - Unnatural blending around facial boundaries
   - Inconsistent lighting or shadows
   - Unusual texture patterns
   - Asymmetries in facial features
   - Artifacts in teeth, eyes, or hair

3. **Confidence Assessment**: Does the heatmap support the {confidence*100:.1f}% confidence level? Are the focus areas typical for {prediction.lower()} images?

4. **Limitations**: What limitations might this analysis have? Are there ambiguous regions?

Provide your analysis in a clear, professional forensic report format (3-4 paragraphs). Be specific about spatial locations (e.g., "left eye region", "mouth boundary", "right cheek").
"""

        # Initialize model
        model = genai.GenerativeModel(model_name)

        # Generate response with both images
        response = model.generate_content(
            [prompt, original_image, heatmap_pil],
            request_options={"timeout": timeout}
        )

        return response.text

    except Exception as e:
        error_msg = f"⚠ Gemini API Error: {str(e)}"
        print(error_msg)
        return error_msg


def visualize_attention_rollout_vit(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Compute attention rollout for Vision Transformer models
    Alternative to Grad-CAM that's more native to ViT architecture

    Args:
        model: ViT model
        input_tensor: Input image tensor
        device: Device

    Returns:
        Attention map as numpy array
    """

    # This is a simplified placeholder
    # Full implementation would require hooking into ViT attention layers
    # For now, we rely on Grad-CAM with reshape_transform

    raise NotImplementedError(
        "Full Attention Rollout not implemented. "
        "Use compute_gradcam with ViT reshape_transform instead."
    )


def create_side_by_side_comparison(
    original_image: Image.Image,
    heatmap_image: np.ndarray,
    prediction: str,
    confidence: float
) -> Image.Image:
    """
    Create side-by-side comparison of original and heatmap for visualization

    Args:
        original_image: Original PIL image
        heatmap_image: Grad-CAM heatmap (numpy array)
        prediction: Prediction label
        confidence: Confidence score

    Returns:
        Combined PIL image
    """

    # Ensure original image is RGB
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')

    # Resize original to match heatmap size
    heatmap_pil = Image.fromarray(heatmap_image.astype('uint8'))
    original_resized = original_image.resize(heatmap_pil.size)

    # Create side-by-side image
    combined_width = original_resized.width + heatmap_pil.width + 10  # 10px gap
    combined_height = max(original_resized.height, heatmap_pil.height)

    combined = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

    # Paste images
    combined.paste(original_resized, (0, 0))
    combined.paste(heatmap_pil, (original_resized.width + 10, 0))

    return combined


def batch_gradcam(
    model: nn.Module,
    images: List[torch.Tensor],
    model_name: str,
    device: str = 'cuda'
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute Grad-CAM for a batch of images

    Args:
        model: Trained model
        images: List of image tensors
        model_name: Model name
        device: Device

    Returns:
        List of (cam_image, heatmap) tuples
    """

    results = []

    for img_tensor in images:
        # Ensure batch dimension
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.to(device)

        cam_image, heatmap = compute_gradcam(
            model=model,
            input_tensor=img_tensor,
            model_name=model_name
        )

        results.append((cam_image, heatmap))

    return results
