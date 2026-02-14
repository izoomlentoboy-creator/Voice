"""
EchoFlow 2.0 - Wav2Vec2 Feature Extraction Module
Extracts high-level speech representations using pre-trained Wav2Vec2
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
import numpy as np
from typing import Union, List


class Wav2Vec2FeatureExtractor(nn.Module):
    """
    Feature extractor using pre-trained Wav2Vec2-LARGE model.
    Extracts contextualized speech embeddings from raw audio.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-xlsr-53",
        freeze_encoder: bool = True,
        target_sr: int = 16000
    ):
        """
        Args:
            model_name: Hugging Face model identifier
            freeze_encoder: Whether to freeze Wav2Vec2 weights during training
            target_sr: Target sampling rate (Wav2Vec2 requires 16kHz)
        """
        super().__init__()
        
        self.target_sr = target_sr
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze encoder if specified (recommended for small datasets)
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Output dimension of Wav2Vec2-LARGE
        self.output_dim = self.model.config.hidden_size  # 1024
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio array
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio
    
    def forward(self, audio_input: Union[str, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Extract Wav2Vec2 features from audio.
        
        Args:
            audio_input: Audio file path, numpy array, or torch tensor
            
        Returns:
            Contextualized embeddings [batch_size, seq_len, hidden_size]
        """
        # Handle different input types
        if isinstance(audio_input, str):
            audio = self.load_audio(audio_input)
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input
        elif isinstance(audio_input, torch.Tensor):
            audio = audio_input.cpu().numpy()
        else:
            raise ValueError(f"Unsupported input type: {type(audio_input)}")
        
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        )
        
        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.set_grad_enabled(self.training):
            outputs = self.model(**inputs)
        
        # Return last hidden states
        return outputs.last_hidden_state
    
    def extract_pooled_features(self, audio_input: Union[str, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Extract pooled (mean) features from audio.
        
        Args:
            audio_input: Audio file path, numpy array, or torch tensor
            
        Returns:
            Pooled embeddings [batch_size, hidden_size]
        """
        features = self.forward(audio_input)
        
        # Mean pooling over time dimension
        pooled = torch.mean(features, dim=1)
        
        return pooled


class TraditionalAcousticFeatures:
    """
    Extracts traditional acoustic features for voice pathology detection.
    Complements Wav2Vec2 with domain-specific features.
    """
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def extract_pitch_features(self, audio: np.ndarray) -> dict:
        """Extract pitch-related features (F0, jitter, shimmer)."""
        # Extract F0 using librosa
        f0 = librosa.yin(audio, fmin=50, fmax=400, sr=self.sr)
        
        # Remove unvoiced frames (F0 = 0)
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) == 0:
            return {
                'f0_mean': 0.0,
                'f0_std': 0.0,
                'jitter': 0.0
            }
        
        # Calculate features
        f0_mean = np.mean(f0_voiced)
        f0_std = np.std(f0_voiced)
        
        # Jitter (pitch perturbation)
        if len(f0_voiced) > 1:
            jitter = np.mean(np.abs(np.diff(f0_voiced))) / f0_mean
        else:
            jitter = 0.0
        
        return {
            'f0_mean': float(f0_mean),
            'f0_std': float(f0_std),
            'jitter': float(jitter)
        }
    
    def extract_energy_features(self, audio: np.ndarray) -> dict:
        """Extract energy-related features (RMS, shimmer)."""
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        
        # Shimmer (amplitude perturbation)
        if len(rms) > 1:
            shimmer = np.mean(np.abs(np.diff(rms))) / np.mean(rms)
        else:
            shimmer = 0.0
        
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'shimmer': float(shimmer)
        }
    
    def extract_spectral_features(self, audio: np.ndarray) -> dict:
        """Extract spectral features."""
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_centroid_std': float(np.std(spectral_centroid)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'zcr_mean': float(np.mean(zcr))
        }
    
    def extract_all(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract all traditional features and return as vector.
        
        Args:
            audio: Audio signal
            
        Returns:
            Feature vector (11 dimensions)
        """
        pitch_feats = self.extract_pitch_features(audio)
        energy_feats = self.extract_energy_features(audio)
        spectral_feats = self.extract_spectral_features(audio)
        
        # Combine all features
        features = [
            pitch_feats['f0_mean'],
            pitch_feats['f0_std'],
            pitch_feats['jitter'],
            energy_feats['rms_mean'],
            energy_feats['rms_std'],
            energy_feats['shimmer'],
            spectral_feats['spectral_centroid_mean'],
            spectral_feats['spectral_centroid_std'],
            spectral_feats['spectral_rolloff_mean'],
            spectral_feats['zcr_mean']
        ]
        
        return np.array(features, dtype=np.float32)


class HybridFeatureExtractor(nn.Module):
    """
    Combines Wav2Vec2 embeddings with traditional acoustic features.
    """
    
    def __init__(
        self,
        wav2vec2_model: str = "facebook/wav2vec2-large-xlsr-53",
        freeze_wav2vec2: bool = True
    ):
        super().__init__()
        
        self.wav2vec2 = Wav2Vec2FeatureExtractor(
            model_name=wav2vec2_model,
            freeze_encoder=freeze_wav2vec2
        )
        
        self.traditional = TraditionalAcousticFeatures()
        
        # Total feature dimension: 1024 (Wav2Vec2) + 10 (traditional) = 1034
        self.output_dim = self.wav2vec2.output_dim + 10
    
    def forward(self, audio_path: str) -> torch.Tensor:
        """
        Extract hybrid features from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Combined feature vector [1, feature_dim]
        """
        # Extract Wav2Vec2 features (pooled)
        wav2vec2_feats = self.wav2vec2.extract_pooled_features(audio_path)
        
        # Extract traditional features
        audio = self.wav2vec2.load_audio(audio_path)
        trad_feats = self.traditional.extract_all(audio)
        trad_feats = torch.from_numpy(trad_feats).unsqueeze(0).to(wav2vec2_feats.device)
        
        # Concatenate
        combined = torch.cat([wav2vec2_feats, trad_feats], dim=1)
        
        return combined
