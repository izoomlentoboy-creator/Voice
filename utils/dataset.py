"""
EchoFlow 2.0 - Dataset Loader
Handles Saarbruecken Voice Database and custom datasets
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from typing import List, Tuple, Optional, Dict
import json
from pathlib import Path


class VoicePathologyDataset(Dataset):
    """
    Dataset for voice pathology detection.
    Compatible with Saarbruecken Voice Database structure.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sr: int = 16000,
        augmentation=None,
        max_duration: float = 5.0
    ):
        """
        Args:
            data_dir: Root directory of dataset
            split: 'train', 'val', or 'test'
            sr: Sampling rate
            augmentation: Augmentation pipeline
            max_duration: Maximum audio duration in seconds
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sr = sr
        self.augmentation = augmentation
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sr)
        
        # Load file list
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """
        Load sample list from dataset directory.
        Expected structure:
        data_dir/
            healthy/
                *.wav
            pathological/
                *.wav
        """
        samples = []
        
        # Healthy samples (label 0)
        healthy_dir = self.data_dir / 'healthy'
        if healthy_dir.exists():
            for audio_file in healthy_dir.glob('*.wav'):
                samples.append({
                    'path': str(audio_file),
                    'label': 0,
                    'class_name': 'healthy'
                })
        
        # Pathological samples (label 1)
        pathological_dir = self.data_dir / 'pathological'
        if pathological_dir.exists():
            for audio_file in pathological_dir.glob('*.wav'):
                samples.append({
                    'path': str(audio_file),
                    'label': 1,
                    'class_name': 'pathological'
                })
        
        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(samples)
        
        # Split into train/val/test (70/15/15)
        n_samples = len(samples)
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)
        
        if self.split == 'train':
            samples = samples[:train_end]
        elif self.split == 'val':
            samples = samples[train_end:val_end]
        elif self.split == 'test':
            samples = samples[val_end:]
        
        return samples
    
    def _load_audio(self, path: str) -> np.ndarray:
        """Load and preprocess audio file."""
        # Load audio
        audio, sr = librosa.load(path, sr=self.sr, mono=True)
        
        # Pad or trim to fixed length
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)), mode='constant')
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get a sample.
        
        Returns:
            Tuple of (audio, label)
        """
        sample = self.samples[idx]
        
        # Load audio
        audio = self._load_audio(sample['path'])
        
        # Apply augmentation (only for training)
        if self.augmentation is not None and self.split == 'train':
            audio = self.augmentation(audio)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return audio_tensor, label


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 8,  # OPTIMIZED: More workers
    augmentation=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        augmentation: Augmentation pipeline for training
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = VoicePathologyDataset(
        data_dir=data_dir,
        split='train',
        augmentation=augmentation
    )
    
    val_dataset = VoicePathologyDataset(
        data_dir=data_dir,
        split='val',
        augmentation=None  # No augmentation for validation
    )
    
    test_dataset = VoicePathologyDataset(
        data_dir=data_dir,
        split='test',
        augmentation=None  # No augmentation for test
    )
    
    # OPTIMIZED: Create dataloaders with better parameters
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Fast GPU transfer
        prefetch_factor=4,  # OPTIMIZED: More prefetch
        persistent_workers=True  # OPTIMIZED: Don't recreate workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader
