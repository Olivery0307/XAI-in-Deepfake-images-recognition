"""
Training and Validation Loops
Handles model training with best model checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Dict, Optional, Tuple
import time

from .models import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> Tuple[float, float]:
    """
    Train model for one epoch

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        avg_loss: Average training loss
        avg_acc: Average training accuracy
    """
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        _, preds = torch.max(logits, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += images.size(0)

        # Update progress bar
        current_acc = running_corrects / total_samples
        pbar.set_postfix({
            'loss': f'{running_loss / total_samples:.4f}',
            'acc': f'{current_acc:.4f}'
        })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: Optional[int] = None
) -> Tuple[float, float]:
    """
    Validate model for one epoch

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number (optional, for display)

    Returns:
        avg_loss: Average validation loss
        avg_acc: Average validation accuracy
    """
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
    pbar = tqdm(dataloader, desc=desc)

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            loss = criterion(logits, labels)

            # Statistics
            _, preds = torch.max(logits, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += images.size(0)

            # Update progress bar
            current_acc = running_corrects / total_samples
            pbar.set_postfix({
                'loss': f'{running_loss / total_samples:.4f}',
                'acc': f'{current_acc:.4f}'
            })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def main_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str,
    checkpoint_dir: str = './checkpoints',
    model_name: str = 'model',
    patience: int = 3,
    min_delta: float = 0.001,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict:
    """
    Main training loop with best model saving and early stopping

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        model_name: Name prefix for saved models
        patience: Early stopping patience
        min_delta: Minimum improvement to reset patience
        scheduler: Optional learning rate scheduler

    Returns:
        history: Dictionary containing training history
    """

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"Starting Training: {model_name}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc + min_delta:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best model checkpoint
            best_model_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_acc=val_acc,
                save_path=best_model_path,
                additional_info={
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }
            )
            print(f"  ✓ New best model saved! (Improvement: +{improvement:.4f})")

        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            print(f"  Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break

        # Save latest model checkpoint
        if epoch % 5 == 0 or epoch == num_epochs:
            latest_model_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch}.pth')
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_acc=val_acc,
                save_path=latest_model_path,
                additional_info={
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }
            )

    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total time: {total_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"{'='*60}\n")

    return history


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """
    Test the model on test dataset

    Args:
        model: Trained model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to test on

    Returns:
        test_loss: Test loss
        test_acc: Test accuracy
    """
    print("\n" + "="*60)
    print("Running Final Test Evaluation")
    print("="*60 + "\n")

    test_loss, test_acc = validate_one_epoch(
        model, test_loader, criterion, device, epoch=None
    )

    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.4f}")
    print("="*60 + "\n")

    return test_loss, test_acc
