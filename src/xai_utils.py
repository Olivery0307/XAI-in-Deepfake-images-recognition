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


def compute_attention_rollout(
    attentions: tuple,
    discard_ratio: float = 0.9
) -> torch.Tensor:
    """
    Compute attention rollout from all transformer layers.

    Args:
        attentions: Tuple of attention tensors from each layer
                   Shape per layer: (batch_size, num_heads, num_tokens, num_tokens)
        discard_ratio: Percentage of lowest attention values to discard (default: 0.9)

    Returns:
        Attention map for the [CLS] token (excluding CLS token itself)
        Shape: (num_patches,)
    """
    # Get device from first attention tensor
    device = attentions[0].device

    # Create identity matrix on the same device
    result = torch.eye(attentions[0].size(-1)).to(device)

    for attention in attentions:
        # Average across all heads: (batch_size, num_heads, num_tokens, num_tokens) -> (batch_size, num_tokens, num_tokens)
        attention_heads_fused = attention.mean(dim=1)
        # Take first batch
        attention_heads_fused = attention_heads_fused[0]

        # Drop the lowest attentions
        flat = attention_heads_fused.view(-1)
        _, indices = flat.topk(k=int(flat.size(-1) * discard_ratio), largest=False)
        flat[indices] = 0

        # Normalize
        I = torch.eye(attention_heads_fused.size(-1)).to(device)
        a = (attention_heads_fused + 1.0 * I) / 2
        a = a / a.sum(dim=-1, keepdim=True)

        # Multiply with result
        result = torch.matmul(a, result)

    # Get attention for CLS token, excluding CLS token itself
    mask = result[0, 1:]
    return mask


def visualize_attention_rollout_vit(
    model: nn.Module,
    input_tensor: torch.Tensor,
    discard_ratio: float = 0.9,
    original_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute attention rollout for Vision Transformer models
    More native to ViT architecture than Grad-CAM

    Args:
        model: ViT model (must support output_attentions=True)
        input_tensor: Input image tensor (1, 3, H, W) - MUST be normalized
        discard_ratio: Percentage of lowest attention values to discard (default: 0.9)
        original_size: Optional (width, height) to resize heatmap to original image size

    Returns:
        cam_image: Attention rollout visualization as RGB numpy array (0-255)
        heatmap: Raw attention map as numpy array (0-1)
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Get model outputs with attentions
    with torch.no_grad():
        # For HuggingFace models
        if hasattr(model, 'vit'):
            # ViTForImageClassification
            outputs = model(input_tensor, output_attentions=True)
        else:
            # Try generic forward with output_attentions
            outputs = model(input_tensor, output_attentions=True)

        # Extract attentions
        if hasattr(outputs, 'attentions'):
            attentions = outputs.attentions
        else:
            raise ValueError("Model does not support output_attentions=True")

    # Compute attention rollout
    mask = compute_attention_rollout(attentions, discard_ratio=discard_ratio)

    # Reshape mask to image dimensions
    # mask shape: (num_patches,) where num_patches = (H/patch_size) * (W/patch_size)
    num_patches = int(mask.shape[0] ** 0.5)
    mask = mask.reshape(num_patches, num_patches).cpu().numpy()

    # Denormalize input tensor for visualization
    from .preprocessing import denormalize_image
    denorm_tensor = denormalize_image(input_tensor.squeeze(0).cpu())

    # Convert to numpy (H, W, C) in range [0, 1]
    rgb_img = denorm_tensor.permute(1, 2, 0).numpy()
    rgb_img = np.float32(rgb_img)

    # Determine target size
    if original_size is not None:
        width, height = original_size
    else:
        # Use input tensor size (typically 224x224)
        height, width = input_tensor.shape[2], input_tensor.shape[3]

    # Resize mask to target size
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

    # Normalize mask to [0, 1]
    mask_resized = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8)

    # Resize rgb_img if needed
    if original_size is not None:
        rgb_img_resized = cv2.resize(rgb_img, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        rgb_img_resized = rgb_img

    # Create visualization using same function as Grad-CAM for consistency
    cam_image = show_cam_on_image(rgb_img_resized, mask_resized, use_rgb=True)

    return cam_image, mask_resized


def compute_xai_visualization(
    model: nn.Module,
    input_tensor: torch.Tensor,
    model_name: str,
    target_layer: Optional[nn.Module] = None,
    target_category: Optional[int] = None,
    original_size: Optional[Tuple[int, int]] = None,
    use_attention_rollout: bool = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Unified function to compute XAI visualization.
    Automatically chooses between Grad-CAM and Attention Rollout based on model type.

    Args:
        model: Trained model
        input_tensor: Input image tensor (1, 3, H, W) - MUST be normalized
        model_name: Name of model
        target_layer: Target layer for Grad-CAM (auto-selected if None)
        target_category: Target class index (uses predicted class if None)
        original_size: Optional (width, height) to resize heatmap to original image size
        use_attention_rollout: Force use of attention rollout (None = auto-detect for ViT)

    Returns:
        cam_image: XAI visualization as RGB numpy array (0-255)
        heatmap: Raw heatmap as numpy array (0-1)
        method_used: String indicating which method was used ("Grad-CAM" or "Attention Rollout")
    """
    model_name_lower = model_name.lower()

    # Auto-detect if we should use attention rollout for ViT models
    if use_attention_rollout is None:
        use_attention_rollout = 'vit' in model_name_lower

    if use_attention_rollout:
        try:
            cam_image, heatmap = visualize_attention_rollout_vit(
                model=model,
                input_tensor=input_tensor,
                discard_ratio=0.9,
                original_size=original_size
            )
            return cam_image, heatmap, "Attention Rollout"
        except Exception as e:
            print(f"Attention Rollout failed, falling back to Grad-CAM: {e}")
            # Fall back to Grad-CAM
            use_attention_rollout = False

    if not use_attention_rollout:
        cam_image, heatmap = compute_gradcam(
            model=model,
            input_tensor=input_tensor,
            model_name=model_name,
            target_layer=target_layer,
            target_category=target_category,
            original_size=original_size
        )
        return cam_image, heatmap, "Grad-CAM"


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
