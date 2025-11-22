"""
Image Preprocessing and Transforms
Handles training augmentation and inference transforms for CNNs and ViTs
"""

import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional


def get_transforms(split: str = 'train', model_type: str = 'cnn', img_size: int = 224):
    """
    Get data transforms for different splits and model types

    Args:
        split: 'train', 'val', or 'test'
        model_type: 'cnn' or 'vit'
        img_size: Target image size (default: 224)

    Returns:
        torchvision.transforms.Compose object
    """

    # Normalization parameters
    # ImageNet stats (standard for pretrained models)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Some ViT models may use different normalization
    # For now, we default to ImageNet for compatibility
    if model_type == 'vit':
        # ViT models from torchvision typically use ImageNet normalization
        # If using timm with specific pretrained weights, may need adjustment
        mean = imagenet_mean
        std = imagenet_std
    else:
        mean = imagenet_mean
        std = imagenet_std

    if split == 'train':
        # Training transforms with data augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return transform


def process_uploaded_image(
    pil_image: Image.Image,
    model_type: str = 'cnn',
    img_size: int = 224,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Process a single uploaded PIL image for inference

    Args:
        pil_image: PIL Image object
        model_type: 'cnn' or 'vit'
        img_size: Target image size
        device: Target device ('cuda', 'mps', or 'cpu')

    Returns:
        Tensor with shape (1, 3, img_size, img_size) ready for model input
    """

    # Convert to RGB if needed (handles RGBA, grayscale, etc.)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Get validation transforms
    transform = get_transforms(split='val', model_type=model_type, img_size=img_size)

    # Apply transforms and add batch dimension
    image_tensor = transform(pil_image).unsqueeze(0)  # (1, 3, H, W)

    # Move to device
    image_tensor = image_tensor.to(device)

    return image_tensor


def denormalize_image(
    tensor: torch.Tensor,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None
) -> torch.Tensor:
    """
    Denormalize an image tensor for visualization

    Args:
        tensor: Normalized image tensor (C, H, W) or (B, C, H, W)
        mean: Mean used for normalization (defaults to ImageNet)
        std: Std used for normalization (defaults to ImageNet)

    Returns:
        Denormalized tensor
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    # Handle batched input
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to PIL Image for visualization

    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W)

    Returns:
        PIL Image
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Denormalize
    tensor = denormalize_image(tensor)

    # Convert to numpy and PIL
    numpy_image = tensor.permute(1, 2, 0).cpu().numpy()
    numpy_image = (numpy_image * 255).astype('uint8')

    return Image.fromarray(numpy_image)
