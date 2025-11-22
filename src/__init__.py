"""
Deepfake Detection Package
Modular implementation for training and inference
"""

__version__ = "1.0.0"

from .models import get_model
from .preprocessing import get_transforms, process_uploaded_image
from .data_loader import LocalImageDataset, get_data_mixed_structure, save_splits, load_splits
from .trainer import train_one_epoch, validate_one_epoch, main_training_loop
from .xai_utils import compute_gradcam, get_gemini_explanation

__all__ = [
    'get_model',
    'get_transforms',
    'process_uploaded_image',
    'LocalImageDataset',
    'get_data_mixed_structure',
    'save_splits',
    'load_splits',
    'train_one_epoch',
    'validate_one_epoch',
    'main_training_loop',
    'compute_gradcam',
    'get_gemini_explanation',
]
