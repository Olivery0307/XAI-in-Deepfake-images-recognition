"""
Central Configuration File for Deepfake Detection Project
Handles paths, hyperparameters, and API secrets
"""

import os
from typing import Dict

class Config:
    """Configuration class for the deepfake detection pipeline"""

    # ========== PATHS ==========
    # Google Cloud Storage mount point (for Colab training)
    GCS_MOUNT_POINT = "/content/gcs_data"

    # Dataset sub-paths
    PATHS = {
        # Video-based datasets (requires folder-level splitting)
        'celeb_real': os.path.join(GCS_MOUNT_POINT, 'Celeb-real'),
        'youtube_real': os.path.join(GCS_MOUNT_POINT, 'YouTube-real'),
        'celeb_synthesis': os.path.join(GCS_MOUNT_POINT, 'Celeb-synthesis'),

        # Image-based datasets (file-level splitting)
        'ffhq_real': os.path.join(GCS_MOUNT_POINT, 'FFHQ-real-V2'),
        'stylegan_fake': os.path.join(GCS_MOUNT_POINT, 'StyleGan'),
        'stablediffusion_fake': os.path.join(GCS_MOUNT_POINT, 'StableDiffusion-fake-V2'),
    }

    # Output paths
    CHECKPOINT_DIR = "./checkpoints"
    SPLITS_DIR = "./splits"
    LOGS_DIR = "./logs"
    HOLDOUT_DIR = "./hold-out-set"

    # ========== HYPERPARAMETERS ==========
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5
    IMG_SIZE = 380
    # IMG_SIZE = 224
    SEED = 42
    NUM_WORKERS = 4

    # Model settings
    NUM_CLASSES = 2  # Real vs Fake
    PRETRAINED = True

    # Train/Val/Test splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Early stopping
    PATIENCE = 3
    MIN_DELTA = 0.001

    # ========== SUPPORTED MODELS ==========
    SUPPORTED_MODELS = ['resnet34', 'efficientnet_b4', 'vit_b_16']

    # ========== SECRETS ==========
    # Gemini API Key (load from environment or Kaggle secrets)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

    # ========== XAI SETTINGS ==========
    # Target layers for Grad-CAM (auto-selected if None)
    GRADCAM_TARGET_LAYERS = {
        'resnet34': None,  # Will auto-select layer4[-1]
        'efficientnet_b4': None,
        'vit_b_16': None,
    }

    @classmethod
    def get_device(cls):
        """Get available device (CUDA/MPS/CPU)"""
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.CHECKPOINT_DIR, cls.SPLITS_DIR, cls.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def validate_paths(cls) -> bool:
        """Validate that GCS mount point exists (for training environment)"""
        return os.path.exists(cls.GCS_MOUNT_POINT)

    @classmethod
    def get_config_dict(cls) -> Dict:
        """Return configuration as dictionary for logging"""
        return {
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'num_epochs': cls.NUM_EPOCHS,
            'img_size': cls.IMG_SIZE,
            'seed': cls.SEED,
            'train_ratio': cls.TRAIN_RATIO,
            'val_ratio': cls.VAL_RATIO,
            'test_ratio': cls.TEST_RATIO,
        }
