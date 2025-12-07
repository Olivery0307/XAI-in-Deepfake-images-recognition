"""
Evaluation Utilities
Functions for evaluating model performance on test sets, specific domains, and holdout datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np
from tqdm.auto import tqdm
import os
import glob
from PIL import Image
from collections import defaultdict

from .data_loader import LocalImageDataset
from .models import get_model
from .preprocessing import get_transforms

def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> dict:
    """
    Standard evaluation on the test set.
    Returns a dictionary of metrics.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    print("\n" + "="*60)
    print("Running Standard Test Evaluation")
    print("="*60)

    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            # Handle HF ViT output object
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            loss = criterion(logits, labels)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total_loss = running_loss / len(test_loader.dataset)
    
    print("\n--- Overall Results ---")
    print(classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"], digits=4))
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"AUC Score: {auc:.4f}")
    except ValueError:
        auc = 0.0
        print("AUC Score: N/A (Single class present)")

    return {
        'loss': total_loss,
        'predictions': all_preds,
        'labels': all_labels,
        'probs': all_probs,
        'auc': auc
    }


def evaluate_per_domain(
    model: nn.Module,
    test_dataset: LocalImageDataset,
    device: str,
    batch_size: int = 32
):
    """
    Evaluates the model and breaks down performance by domain 
    (Celeb-real, YouTube-real, Celeb-synthesis, FFHQ, etc.)
    based on file paths.
    """
    print("\n" + "="*70)
    print("PER-DOMAIN EVALUATION")
    print("="*70)

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Predicting"):
            images = images.to(device)
            
            outputs = model(images)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    # Define domain keywords mapping
    domains = {
        'Celeb-real': 'Celeb-real',
        'YouTube-real': 'YouTube-real',
        'Celeb-synthesis': 'Celeb-synthesis',
        'FFHQ-real': 'FFHQ-real',
        'StableDiffusion-fake': 'StableDiffusion-fake', # Adjusted to match common path keywords
        'StyleGAN-fake': 'stylegan',
    }

    # Analyze per domain
    for domain_name, keyword in domains.items():
        # Find indices where the path contains the keyword
        indices = [i for i, p in enumerate(all_paths) if keyword in p]
        
        if not indices:
            print(f"\n--- {domain_name} (0 samples) ---")
            continue
            
        domain_labels = [all_labels[i] for i in indices]
        domain_preds = [all_preds[i] for i in indices]
        
        print(f"\n--- {domain_name} ({len(indices)} samples) ---")
        
        # specific check for single-class domains (common in domain splits)
        if len(set(domain_labels)) > 1:
            print(classification_report(domain_labels, domain_preds, 
                                       target_names=["REAL", "FAKE"], digits=4))
        else:
            # Manually calculate accuracy if only one class exists
            acc = sum(1 for i in range(len(domain_labels)) if domain_labels[i] == domain_preds[i])
            print(f"Accuracy: {acc/len(domain_labels)*100:.2f}%")
            
            # Print confusion details
            unique, counts = np.unique(domain_preds, return_counts=True)
            pred_counts = dict(zip(unique, counts))
            print(f"Predictions: {pred_counts} (0=REAL, 1=FAKE)")


def evaluate_holdout(
    model: nn.Module, 
    holdout_dir: str, 
    model_type: str = 'cnn',
    img_size: int = 224,
    device: str = 'cuda'
):
    """
    Evaluates the model on a separate holdout dataset structured as:
    holdout_dir/
      ├── real/
      └── fake/
    """
    print("\n" + "="*70)
    print(f"HOLDOUT EVALUATION: {holdout_dir}")
    print("="*70)
    
    # 1. Setup Data
    # Use get_transforms from preprocessing to ensure consistency
    transform = get_transforms(split='test', model_type=model_type, img_size=img_size)
    
    # Build file list manually to match LocalImageDataset structure
    file_paths = []
    labels = []
    
    # Real images (Label 0)
    real_path = os.path.join(holdout_dir, "real")
    if os.path.exists(real_path):
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            found = glob.glob(os.path.join(real_path, ext))
            file_paths.extend(found)
            labels.extend([0] * len(found))
            
    # Fake images (Label 1)
    fake_path = os.path.join(holdout_dir, "fake")
    if os.path.exists(fake_path):
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            found = glob.glob(os.path.join(fake_path, ext))
            file_paths.extend(found)
            labels.extend([1] * len(found))
            
    print(f"Found {len(file_paths)} images ({labels.count(0)} Real, {labels.count(1)} Fake)")
    
    if len(file_paths) == 0:
        print("❌ No images found. Skipping evaluation.")
        return

    # Create dataset and loader
    dataset = LocalImageDataset(file_paths, labels, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 2. Inference
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, batch_labels, _ in tqdm(loader, desc="Holdout Inference"):
            images = images.to(device)
            
            outputs = model(images)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # 3. Metrics
    print("\n--- Holdout Results ---")
    print(classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"], digits=4))
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"AUC Score: {auc:.4f}")
    except:
        print("AUC: N/A")
        
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")