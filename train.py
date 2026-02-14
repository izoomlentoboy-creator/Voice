"""
EchoFlow 2.0 - Maximum Quality Training Script

Advanced training techniques:
- Label smoothing
- Mixed precision training (FP16)
- Gradient accumulation
- Cosine annealing with warm restarts
- Early stopping with patience
- Model checkpointing
- Comprehensive metrics (accuracy, sensitivity, specificity, F1)
- Learning rate warmup
- Gradient clipping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os
import json
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging

from models.echoflow_v2 import EchoFlowV2, count_parameters
from utils.dataset import create_dataloaders
from utils.augmentation import AugmentationPipeline


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.
    Prevents overconfidence and improves generalization.
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        log_probs = torch.log_softmax(pred, dim=-1)
        
        # One-hot encoding with smoothing
        targets = torch.zeros_like(log_probs).scatter_(
            1, target.unsqueeze(1), 1
        )
        targets = targets * (1 - self.smoothing) + self.smoothing / n_classes
        
        loss = (-targets * log_probs).sum(dim=-1).mean()
        return loss


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """
    Advanced training manager with:
    - Mixed precision training
    - Gradient accumulation
    - Label smoothing
    - Early stopping
    - Comprehensive metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        label_smoothing: float = 0.1,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True,
        early_stopping_patience: int = 10,
        save_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision training
        self.mixed_precision = mixed_precision and device == 'cuda'
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        self.warmup_epochs = 5
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Metrics tracking
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_sensitivity': [],
            'val_specificity': [],
            'learning_rate': []
        }
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Mixed precision: {self.mixed_precision}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Label smoothing: {label_smoothing}")
    
    def train_epoch(self, epoch: int, total_epochs: int) -> dict:
        """Train for one epoch with advanced techniques."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Learning rate warmup
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_scale
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
        
        for batch_idx, (audio, labels) in enumerate(pbar):
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            # Mixed precision forward pass
            with autocast(enabled=self.mixed_precision):
                logits = self.model(audio)
                loss = self.criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'acc': f'{100.0 * current_acc:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self, epoch: int, total_epochs: int) -> dict:
        """Validate with comprehensive metrics."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Val]')
        
        with torch.no_grad():
            for audio, labels in pbar:
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                with autocast(enabled=self.mixed_precision):
                    logits = self.model(audio)
                    loss = self.criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=-1)
                _, predicted = torch.max(logits, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Update progress bar
                current_acc = accuracy_score(all_labels, all_preds)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * current_acc:.2f}%'
                })
        
        # Calculate comprehensive metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        # Confusion matrix for sensitivity and specificity
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for positive class
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for negative class
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    
    def train(self, num_epochs: int):
        """Main training loop."""
        logger.info("="*70)
        logger.info("Starting training...")
        logger.info("="*70)
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch, num_epochs)
            
            # Validate
            val_metrics = self.validate(epoch, num_epochs)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"  Val Loss:   {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Val F1:     {val_metrics['f1']:.4f}")
            logger.info(f"  Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.2e}")
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_sensitivity'].append(val_metrics['sensitivity'])
            self.history['val_specificity'].append(val_metrics['specificity'])
            self.history['learning_rate'].append(current_lr)
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_f1 = val_metrics['f1']
                self.save_checkpoint(epoch, 'best.pt', val_metrics)
                logger.info(f"  ✓ New best model saved! (Acc: {self.best_val_acc:.4f}, F1: {self.best_val_f1:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch+1}.pt', val_metrics)
            
            # Save last checkpoint
            self.save_checkpoint(epoch, 'last.pt', val_metrics)
            
            # Early stopping
            if self.early_stopping(val_metrics['f1']):
                logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        logger.info("="*70)
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Best validation F1-score: {self.best_val_f1:.4f}")
        logger.info("="*70)
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, epoch: int, filename: str, metrics: dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def save_history(self):
        """Save training history as JSON."""
        history_path = self.save_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train EchoFlow 2.0')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--gradient_accumulation', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating EchoFlow 2.0 model...")
    model = EchoFlowV2(
        freeze_wav2vec2=True,
        unfreeze_last_n_layers=0,
        d_model=512,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        stochastic_depth_prob=0.1
    )
    
    # Print model info
    params = count_parameters(model)
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {params['total_millions']:.2f}M")
    logger.info(f"  Trainable: {params['trainable_millions']:.2f}M")
    logger.info(f"  Frozen: {params['frozen_millions']:.2f}M")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
        early_stopping_patience=args.early_stopping_patience,
        save_dir=args.save_dir
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
