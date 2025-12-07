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

    # 1. Check for ViT and redirect/error
    if 'vit' in model_name:
        raise ValueError(
            f"Model '{model_name}' is a Vision Transformer. "
            "Please use 'compute_attention_rollout' instead of Grad-CAM."
        )

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
    heatmap_image: Image.Image,
    prediction: str,
    confidence: float,
    model_name_for_context: str = "CNN"
) -> str:
    """
    Generate forensic report using Gemini Pro Vision
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Determine visualization type based on model context
        viz_type = "Attention Rollout (Transformer Attention)" if "vit" in model_name_for_context.lower() else "Grad-CAM (Gradient Activation)"

        prompt = f"""
        You are a forensic AI analyst specializing in Deepfake Detection.
        
        **Context:**
        - **Model Prediction:** {prediction}
        - **Confidence:** {confidence:.1%}
        - **Visualization Type:** {viz_type}
        
        **Visual Evidence:**
        1.  **Image 1 (Original):** The suspect image.
        2.  **Image 2 (Heatmap):** The {viz_type} map. Red/Yellow areas indicate the primary visual features the model used to make its decision.
        
        **Task:**
        Provide a concise forensic summary listing exactly THREE reasons why the image is likely {prediction}, based on the visual evidence.
        
        **Format:**
        1.  **[Feature]:** [Observation from heatmap] -> [Forensic Interpretation]
        2.  **[Feature]:** [Observation from heatmap] -> [Forensic Interpretation]
        3.  **[Feature]:** [Observation from heatmap] -> [Forensic Interpretation]
        
        Keep it brief and objective.
        """

        response = model.generate_content([prompt, original_image, heatmap_image])
        return response.text

    except Exception as e:
        return f"Error generating Gemini report: {str(e)}"


def attention_rollout(attentions: Tuple[torch.Tensor], discard_ratio: float = 0.9) -> torch.Tensor:
    """
    Compute attention rollout from all transformer layers.
    
    Args:
        attentions: tuple of attention tensors from each layer
        discard_ratio: percentage of lowest attention values to discard
        
    Returns:
        Attention map for the [CLS] token
    """
    # Get device from first attention tensor
    device = attentions[0].device
    
    # Start with Identity matrix
    # Attention shape: (batch_size, num_heads, num_tokens, num_tokens)
    # We assume batch_size is handled outside or we take first element if it's 1
    # But here we process assuming attentions are already squeezed or we process per item
    # The provided logic processes a single item's attention map typically
    
    # Let's adapt to handle the tuple structure directly
    # Assuming attentions is a tuple of tensors of shape (1, num_heads, num_tokens, num_tokens)
    
    num_tokens = attentions[0].size(-1)
    result = torch.eye(num_tokens).to(device)
    
    for attention in attentions:
        # Fuse heads (batch dim is 0)
        # attention shape: [1, num_heads, num_tokens, num_tokens]
        attention_heads_fused = attention.mean(dim=1) # [1, num_tokens, num_tokens]
        attention_heads_fused = attention_heads_fused[0] # [num_tokens, num_tokens]
        
        # Drop the lowest attentions (noise filtering)
        flat = attention_heads_fused.view(-1)
        _, indices = flat.topk(k=int(flat.size(-1) * discard_ratio), largest=False)
        flat[indices] = 0
        
        # Normalize: (A + I) / 2
        I = torch.eye(num_tokens).to(device)
        a = (attention_heads_fused + 1.0 * I) / 2
        a = a / a.sum(dim=-1, keepdim=True)
        
        # Recursive multiplication
        result = torch.matmul(a, result)
    
    # Extract Mask from Last Layer ([CLS] token which is index 0)
    mask = result[0, 1:]
    return mask


def visualize_attention_rollout_vit(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_size: Optional[Tuple[int, int]] = None,
    discard_ratio: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Attention Rollout for Vision Transformers (ViT) and return visualization.
    
    Args:
        model: HuggingFace ViT model (must be initialized with output_attentions=True)
        input_tensor: Preprocessed image tensor (1, 3, H, W)
        original_size: (width, height) to resize heatmap
        discard_ratio: Percentage of lowest attentions to discard (noise filtering)
    """
    model.eval()
    
    # 1. Forward pass to get attentions
    with torch.no_grad():
        outputs = model(input_tensor, output_attentions=True)
    
    # tuple of (num_layers) tensors, each [batch, num_heads, num_tokens, num_tokens]
    attentions = outputs.attentions 
    
    # 2. Compute Rollout
    mask = attention_rollout(attentions, discard_ratio=discard_ratio)
    
    # 3. Reshape to 2D grid
    # Assuming square grid: 14x14 for base models (196 tokens)
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).cpu().numpy()
    
    # Normalize mask to [0, 1] for visualization
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    
    # 4. Visualization Preparation
    # Denormalize original image
    from .preprocessing import denormalize_image
    # ViT standard normalization (0.5, 0.5, 0.5)
    denorm_tensor = denormalize_image(input_tensor.squeeze(0).cpu(), mean=[0.5]*3, std=[0.5]*3)
    rgb_img = denorm_tensor.permute(1, 2, 0).numpy()
    rgb_img = np.float32(rgb_img)

    # Resize both to target size
    if original_size:
        w, h = original_size
    else:
        w, h = 224, 224 # Default
        
    mask_resized = cv2.resize(mask, (w, h))
    rgb_resized = cv2.resize(rgb_img, (w, h))
    
    # Apply color map
    visualization = show_cam_on_image(rgb_resized, mask_resized, use_rgb=True)
    
    return visualization, mask_resized


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
