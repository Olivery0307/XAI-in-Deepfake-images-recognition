"""
Model Factory for Deepfake Detection
Supports ResNet, EfficientNet, and ViT with proper handling of architecture differences
"""

import torch
import torch.nn as nn
from torchvision import models
import timm
from typing import Optional


def get_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    device: str = 'cuda'
) -> nn.Module:
    """
    Factory function to create and configure models

    Args:
        model_name: One of ['resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b4', 'vit_b_16', 'vit_b_32']
        num_classes: Number of output classes (default: 2 for Real/Fake)
        pretrained: Whether to use pretrained weights
        device: Target device ('cuda', 'mps', or 'cpu')

    Returns:
        Configured model ready for training/inference
    """

    model_name = model_name.lower()
    print(f"Loading {model_name} (pretrained={pretrained})...")

    # ========== RESNET MODELS ==========
    if model_name == 'resnet34':
        if pretrained:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet34(weights=None)

        # Replace final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(weights=None)

        # Replace final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    # ========== EFFICIENTNET MODELS ==========
    elif model_name == 'efficientnet_b0':
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)

        # Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    elif model_name == 'efficientnet_b4':
        if pretrained:
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b4(weights=None)

        # Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    # ========== VISION TRANSFORMER (ViT) MODELS ==========
    # NOTE: ViT has different architecture - uses 'heads' instead of 'fc'
    elif model_name == 'vit_b_16':
        if pretrained:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_b_16(weights=None)

        # Replace classification head
        # ViT uses model.heads.head instead of model.fc
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)

    elif model_name == 'vit_b_32':
        if pretrained:
            model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_b_32(weights=None)

        # Replace classification head
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)

    # ========== ALTERNATIVE: Use timm for more ViT variants ==========
    elif model_name.startswith('vit_') and pretrained:
        # Fallback to timm library for additional ViT models
        try:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            print(f"Loaded {model_name} from timm library")
        except Exception as e:
            raise ValueError(f"Model {model_name} not found in torchvision or timm: {e}")

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: resnet34, resnet50, efficientnet_b0, efficientnet_b4, vit_b_16, vit_b_32"
        )

    # Move model to device
    model = model.to(device)

    print(f"Model loaded successfully on {device}")
    print(f"Output classes: {num_classes}")

    return model


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the appropriate target layer for Grad-CAM based on model architecture

    Args:
        model: The model instance
        model_name: Name of the model

    Returns:
        Target layer for Grad-CAM
    """
    model_name = model_name.lower()

    # ResNet: Use last layer of layer4
    if 'resnet' in model_name:
        return model.layer4[-1]

    # EfficientNet: Use last feature layer
    elif 'efficientnet' in model_name:
        return model.features[-1]

    # ViT: Use last encoder block
    # Note: ViT requires special handling for Grad-CAM (reshape_transform)
    elif 'vit' in model_name:
        # For torchvision ViT models
        if hasattr(model, 'encoder'):
            return model.encoder.layers.encoder_layer_11  # Last encoder layer
        # For timm ViT models
        elif hasattr(model, 'blocks'):
            return model.blocks[-1]
        else:
            raise ValueError(f"Unable to determine target layer for ViT model: {model_name}")

    else:
        raise ValueError(f"Unknown model type for target layer selection: {model_name}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cuda',
    strict: bool = True
) -> nn.Module:
    """
    Load model weights from checkpoint

    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Target device
        strict: Whether to strictly enforce that the keys in checkpoint match model

    Returns:
        Model with loaded weights
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    print("Checkpoint loaded successfully")

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_acc: float,
    save_path: str,
    additional_info: Optional[dict] = None
):
    """
    Save model checkpoint with metadata

    Args:
        model: Model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch number
        val_acc: Validation accuracy
        save_path: Path to save checkpoint
        additional_info: Additional metadata to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if additional_info is not None:
        checkpoint.update(additional_info)

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")
