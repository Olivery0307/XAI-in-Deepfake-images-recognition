"""
Data Loading and Dataset Management
Handles efficient data loading, smart splitting, and persistence
"""

import os
import pickle
import random
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings


class LocalImageDataset(Dataset):
    """
    Custom Dataset for loading images from file paths with robust error handling
    """

    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        """
        Args:
            file_paths: List of absolute file paths to images
            labels: List of labels (0 or 1)
            transform: Optional transform to apply to images
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

        assert len(file_paths) == len(labels), "file_paths and labels must have same length"

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            image_tensor: Transformed image tensor
            label_tensor: Label as tensor
            file_path: Original file path as string
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # Load image
            image = Image.open(file_path).convert('RGB')

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            label_tensor = torch.tensor(label, dtype=torch.long)

            return image, label_tensor, file_path

        except Exception as e:
            # Handle corrupt images gracefully
            warnings.warn(f"Error loading image {file_path}: {str(e)}. Returning dummy tensor.")

            # Return dummy zero tensor with correct shape
            dummy_image = torch.zeros((3, 224, 224))
            label_tensor = torch.tensor(label, dtype=torch.long)

            return dummy_image, label_tensor, file_path


def get_data_mixed_structure(
    celeb_real_path: str,
    youtube_real_path: str,
    celeb_synthesis_path: str,
    ffhq_real_path: str,
    stylegan_fake_path: str,
    stablediffusion_fake_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_samples_per_category: Optional[int] = None
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """
    Smart data splitting with video-awareness and reproducibility

    Video-based datasets (Celeb-DF, YouTube): Split by FOLDER (Video ID) to prevent leakage
    Image-based datasets (FFHQ, GANs): Split by FILE

    Args:
        *_path: Paths to dataset directories
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
        max_samples_per_category: Optional limit per category for debugging

    Returns:
        train_data: (file_paths, labels)
        val_data: (file_paths, labels)
        test_data: (file_paths, labels)
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)

    # Initialize lists
    train_files, train_labels = [], []
    val_files, val_labels = [], []
    test_files, test_labels = [], []

    # ========== VIDEO-BASED DATASETS: Split by Folder ==========
    video_datasets = [
        (celeb_real_path, 0, "Celeb-real"),
        (youtube_real_path, 0, "YouTube-real"),
        (celeb_synthesis_path, 1, "Celeb-synthesis"),
    ]

    for dataset_path, label, dataset_name in video_datasets:
        if not os.path.exists(dataset_path):
            warnings.warn(f"Path not found: {dataset_path}. Skipping {dataset_name}.")
            continue

        print(f"Processing {dataset_name} (video-based, folder splitting)...")

        # Get all video folders (each folder = one video)
        video_folders = []
        for root, dirs, files in os.walk(dataset_path):
            if files:  # Only consider folders with files
                # Check if this folder contains images
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    video_folders.append(root)

        # CRUCIAL: Sort folders for reproducibility
        video_folders = sorted(video_folders)

        if max_samples_per_category:
            video_folders = video_folders[:max_samples_per_category]

        # Split folders into train/val/test
        train_folders, temp_folders = train_test_split(
            video_folders,
            train_size=train_ratio,
            random_state=seed
        )
        val_folders, test_folders = train_test_split(
            temp_folders,
            train_size=val_ratio / (val_ratio + test_ratio),
            random_state=seed
        )

        # Collect all files from each split
        for folder_list, file_list, label_list in [
            (train_folders, train_files, train_labels),
            (val_folders, val_files, val_labels),
            (test_folders, test_files, test_labels)
        ]:
            for folder in folder_list:
                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            file_list.append(os.path.join(root, file))
                            label_list.append(label)

        print(f"  {dataset_name}: {len(video_folders)} videos total")

    # ========== IMAGE-BASED DATASETS: Split by File ==========
    image_datasets = [
        (ffhq_real_path, 0, "FFHQ-real"),
        (stylegan_fake_path, 1, "StyleGAN-fake"),
        (stablediffusion_fake_path, 1, "StableDiffusion-fake"),
    ]

    for dataset_path, label, dataset_name in image_datasets:
        if not os.path.exists(dataset_path):
            warnings.warn(f"Path not found: {dataset_path}. Skipping {dataset_name}.")
            continue

        print(f"Processing {dataset_name} (image-based, file splitting)...")

        # Efficiently collect all image files using os.walk
        all_files = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append(os.path.join(root, file))

        #s Sort files for reproducibility
        all_files = sorted(all_files)

        if max_samples_per_category:
            all_files = all_files[:max_samples_per_category]

        # Create labels
        labels_list = [label] * len(all_files)

        # Split files
        train_f, temp_f, train_l, temp_l = train_test_split(
            all_files,
            labels_list,
            train_size=train_ratio,
            random_state=seed,
            stratify=labels_list
        )
        val_f, test_f, val_l, test_l = train_test_split(
            temp_f,
            temp_l,
            train_size=val_ratio / (val_ratio + test_ratio),
            random_state=seed,
            stratify=temp_l
        )

        # Append to global lists
        train_files.extend(train_f)
        train_labels.extend(train_l)
        val_files.extend(val_f)
        val_labels.extend(val_l)
        test_files.extend(test_f)
        test_labels.extend(test_l)

        print(f"  {dataset_name}: {len(all_files)} images total")

    # Final summary
    print(f"\n=== Final Split Summary ===")
    print(f"Train: {len(train_files)} images (Real: {train_labels.count(0)}, Fake: {train_labels.count(1)})")
    print(f"Val:   {len(val_files)} images (Real: {val_labels.count(0)}, Fake: {val_labels.count(1)})")
    print(f"Test:  {len(test_files)} images (Real: {test_labels.count(0)}, Fake: {test_labels.count(1)})")

    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


def save_splits(
    train_data: Tuple[List[str], List[int]],
    val_data: Tuple[List[str], List[int]],
    test_data: Tuple[List[str], List[int]],
    save_path: str
):
    """
    Save train/val/test splits to disk for reproducibility

    Args:
        train_data: (file_paths, labels)
        val_data: (file_paths, labels)
        test_data: (file_paths, labels)
        save_path: Path to save pickle file
    """
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(splits, f)

    print(f"Splits saved to {save_path}")


def load_splits(load_path: str) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """
    Load saved train/val/test splits from disk

    Args:
        load_path: Path to pickle file

    Returns:
        train_data, val_data, test_data
    """
    with open(load_path, 'rb') as f:
        splits = pickle.load(f)

    print(f"Splits loaded from {load_path}")
    print(f"Train: {len(splits['train'][0])} images")
    print(f"Val:   {len(splits['val'][0])} images")
    print(f"Test:  {len(splits['test'][0])} images")

    return splits['train'], splits['val'], splits['test']
