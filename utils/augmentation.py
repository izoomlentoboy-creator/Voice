"""
EchoFlow 2.0 - Data Augmentation Pipeline
Comprehensive augmentation for robust voice pathology detection
"""

import numpy as np
import librosa
import torch
from typing import Optional, List, Tuple
import random


class AudioAugmentation:
    """
    Comprehensive audio augmentation pipeline.
    Combines time-domain, frequency-domain, and sample mixing techniques.
    """
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def add_noise(
        self,
        audio: np.ndarray,
        noise_type: str = 'white',
        snr_db: float = 20.0
    ) -> np.ndarray:
        """
        Add noise to audio signal.
        
        Args:
            audio: Input audio
            noise_type: Type of noise ('white', 'pink', 'brown')
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Noisy audio
        """
        # Generate noise
        if noise_type == 'white':
            noise = np.random.randn(len(audio))
        elif noise_type == 'pink':
            # Pink noise (1/f spectrum)
            noise = self._generate_pink_noise(len(audio))
        elif noise_type == 'brown':
            # Brown noise (1/f^2 spectrum)
            noise = self._generate_brown_noise(len(audio))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Calculate signal and noise power
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Calculate noise scaling factor for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = np.sqrt(signal_power / (noise_power * snr_linear))
        
        # Add scaled noise
        noisy_audio = audio + noise_scale * noise
        
        return noisy_audio
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise (1/f spectrum)."""
        # Simple approximation using multiple white noise sources
        white = np.random.randn(length)
        pink = np.cumsum(white)
        pink = pink - np.mean(pink)
        pink = pink / np.std(pink)
        return pink
    
    def _generate_brown_noise(self, length: int) -> np.ndarray:
        """Generate brown noise (1/f^2 spectrum)."""
        white = np.random.randn(length)
        brown = np.cumsum(np.cumsum(white))
        brown = brown - np.mean(brown)
        brown = brown / np.std(brown)
        return brown
    
    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """
        Time stretch audio without changing pitch.
        
        Args:
            audio: Input audio
            rate: Stretch factor (>1 = faster, <1 = slower)
            
        Returns:
            Time-stretched audio
        """
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return stretched
    
    def pitch_shift(self, audio: np.ndarray, n_steps: float = 0.0) -> np.ndarray:
        """
        Shift pitch without changing tempo.
        
        Args:
            audio: Input audio
            n_steps: Number of semitones to shift
            
        Returns:
            Pitch-shifted audio
        """
        shifted = librosa.effects.pitch_shift(
            audio,
            sr=self.sr,
            n_steps=n_steps
        )
        return shifted
    
    def add_reverb(
        self,
        audio: np.ndarray,
        room_scale: float = 0.5
    ) -> np.ndarray:
        """
        Add simple reverb effect.
        
        Args:
            audio: Input audio
            room_scale: Room size (0-1)
            
        Returns:
            Audio with reverb
        """
        # Simple reverb using delayed copies
        delay_samples = int(room_scale * self.sr * 0.05)  # Up to 50ms delay
        decay = 0.3
        
        reverb = np.zeros_like(audio)
        reverb[:] = audio
        
        if delay_samples > 0:
            delayed = np.pad(audio, (delay_samples, 0), mode='constant')[:-delay_samples]
            reverb += decay * delayed
        
        # Normalize
        reverb = reverb / np.max(np.abs(reverb) + 1e-8)
        
        return reverb
    
    def random_gain(
        self,
        audio: np.ndarray,
        min_gain_db: float = -6.0,
        max_gain_db: float = 6.0
    ) -> np.ndarray:
        """
        Apply random gain.
        
        Args:
            audio: Input audio
            min_gain_db: Minimum gain in dB
            max_gain_db: Maximum gain in dB
            
        Returns:
            Gain-adjusted audio
        """
        gain_db = np.random.uniform(min_gain_db, max_gain_db)
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def random_clip(
        self,
        audio: np.ndarray,
        threshold: float = 0.95
    ) -> np.ndarray:
        """
        Randomly clip audio (simulate microphone saturation).
        
        Args:
            audio: Input audio
            threshold: Clipping threshold (0-1)
            
        Returns:
            Clipped audio
        """
        if np.random.rand() < 0.3:  # 30% chance of clipping
            return np.clip(audio, -threshold, threshold)
        return audio


class SpecAugment:
    """
    SpecAugment for mel-spectrogram augmentation.
    Implements time and frequency masking.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spec: Spectrogram [freq, time]
            
        Returns:
            Augmented spectrogram
        """
        spec = spec.copy()
        num_freqs, num_frames = spec.shape
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, num_freqs - f)
            spec[f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, num_frames - t)
            spec[:, t0:t0+t] = 0
        
        return spec


class Mixup:
    """
    Mixup data augmentation.
    Creates synthetic training examples by mixing two samples.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        label1: int,
        label2: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mix two audio samples.
        
        Args:
            audio1: First audio sample
            audio2: Second audio sample
            label1: Label of first sample
            label2: Label of second sample
            
        Returns:
            Mixed audio and soft label
        """
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Ensure same length
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # Mix audio
        mixed_audio = lam * audio1 + (1 - lam) * audio2
        
        # Mix labels (soft labels)
        mixed_label = np.zeros(2)
        mixed_label[label1] = lam
        mixed_label[label2] = 1 - lam
        
        return mixed_audio, mixed_label


class AugmentationPipeline:
    """
    Complete augmentation pipeline.
    Randomly applies multiple augmentation techniques.
    """
    
    def __init__(
        self,
        sr: int = 16000,
        augmentation_prob: float = 0.8,
        use_mixup: bool = False
    ):
        """
        Args:
            sr: Sampling rate
            augmentation_prob: Probability of applying augmentation
            use_mixup: Whether to use Mixup (requires batch processing)
        """
        self.sr = sr
        self.augmentation_prob = augmentation_prob
        self.use_mixup = use_mixup
        
        self.audio_aug = AudioAugmentation(sr=sr)
        self.spec_aug = SpecAugment()
        self.mixup = Mixup() if use_mixup else None
    
    def augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random audio augmentations.
        
        Args:
            audio: Input audio
            
        Returns:
            Augmented audio
        """
        if np.random.rand() > self.augmentation_prob:
            return audio
        
        augmented = audio.copy()
        
        # Time stretch (30% chance)
        if np.random.rand() < 0.3:
            rate = np.random.uniform(0.9, 1.1)
            augmented = self.audio_aug.time_stretch(augmented, rate=rate)
        
        # Pitch shift (30% chance)
        if np.random.rand() < 0.3:
            n_steps = np.random.uniform(-2, 2)
            augmented = self.audio_aug.pitch_shift(augmented, n_steps=n_steps)
        
        # Add noise (50% chance)
        if np.random.rand() < 0.5:
            noise_type = np.random.choice(['white', 'pink', 'brown'])
            snr_db = np.random.uniform(15, 30)
            augmented = self.audio_aug.add_noise(augmented, noise_type=noise_type, snr_db=snr_db)
        
        # Add reverb (30% chance)
        if np.random.rand() < 0.3:
            room_scale = np.random.uniform(0.2, 0.8)
            augmented = self.audio_aug.add_reverb(augmented, room_scale=room_scale)
        
        # Random gain (40% chance)
        if np.random.rand() < 0.4:
            augmented = self.audio_aug.random_gain(augmented)
        
        # Random clip (20% chance)
        if np.random.rand() < 0.2:
            augmented = self.audio_aug.random_clip(augmented)
        
        # Normalize
        augmented = augmented / (np.max(np.abs(augmented)) + 1e-8)
        
        return augmented
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline."""
        return self.augment_audio(audio)
