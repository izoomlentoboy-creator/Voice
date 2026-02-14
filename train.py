"""
EchoFlow 2.0 - Training Script
Train the model on Saarbruecken Voice Database
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from pathlib import Path
import argparse

from models.transformer_classifier import EchoFlowV2
from utils.dataset import create_dataloaders
from utils.augmentation import AugmentationPipeline


class Trainer:
    """Training manager for EchoFlow 2.0."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (audio, labels) in enumerate(pbar):
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model.classifier(audio.unsqueeze(1))
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, epoch: int) -> dict:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for audio, labels in pbar:
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model.classifier(audio.unsqueeze(1))
                loss = self.criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        # Calculate additional metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Sensitivity (recall for pathological class)
        pathological_mask = all_labels == 1
        sensitivity = 100.0 * np.sum(all_preds[pathological_mask] == 1) / np.sum(pathological_mask)
        
        # Specificity (recall for healthy class)
        healthy_mask = all_labels == 0
        specificity = 100.0 * np.sum(all_preds[healthy_mask] == 0) / np.sum(healthy_mask)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pt')
            print(f"✓ Saved best model (val_acc: {metrics['accuracy']:.2f}%)")
    
    def train(self, num_epochs: int):
        """Train the model for multiple epochs."""
        print(f"\n{'='*60}")
        print(f"Starting Training - EchoFlow 2.0")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
            print(f"  Sensitivity: {val_metrics['sensitivity']:.2f}% | Specificity: {val_metrics['specificity']:.2f}%")
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
            
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Save training history
            with open(self.save_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train EchoFlow 2.0')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create augmentation pipeline
    augmentation = AugmentationPipeline(augmentation_prob=0.8)
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation=augmentation
    )
    
    # Create model
    print("Creating model...")
    model = EchoFlowV2(
        wav2vec2_model="facebook/wav2vec2-large-xlsr-53",
        freeze_wav2vec2=True,
        d_model=512,
        nhead=8,
        num_layers=4,
        num_classes=2,
        dropout=0.1
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
