"""
EchoFlow 2.0 - Transformer-based Classification Model
State-of-the-art architecture for voice pathology detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    Adds position information to embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for voice pathology detection.
    Uses multi-head self-attention to capture long-range dependencies.
    """
    
    def __init__(
        self,
        input_dim: int = 1034,  # Wav2Vec2 (1024) + Traditional (10)
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 2,  # Binary: healthy vs pathological
        use_positional_encoding: bool = True
    ):
        """
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            num_classes: Number of output classes
            use_positional_encoding: Whether to use positional encoding
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Handle single vector input (pooled features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Project input to d_model
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # Global average pooling over sequence
        pooled = torch.mean(encoded, dim=1)  # [batch_size, d_model]
        
        # Classification
        logits = self.classifier(pooled)  # [batch_size, num_classes]
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Probabilities [batch_size, num_classes]
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Class indices [batch_size]
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)


class EchoFlowV2(nn.Module):
    """
    Complete EchoFlow 2.0 model.
    Combines feature extraction and classification.
    """
    
    def __init__(
        self,
        wav2vec2_model: str = "facebook/wav2vec2-large-xlsr-53",
        freeze_wav2vec2: bool = True,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Import here to avoid circular dependency
        from .feature_extractor import HybridFeatureExtractor
        
        # Feature extraction
        self.feature_extractor = HybridFeatureExtractor(
            wav2vec2_model=wav2vec2_model,
            freeze_wav2vec2=freeze_wav2vec2
        )
        
        # Classification
        self.classifier = TransformerClassifier(
            input_dim=self.feature_extractor.output_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    
    def forward(self, audio_path: str) -> torch.Tensor:
        """
        End-to-end forward pass.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Logits [1, num_classes]
        """
        # Extract features
        features = self.feature_extractor(audio_path)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def predict(self, audio_path: str) -> dict:
        """
        Make prediction on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio_path)
            probs = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()
        
        return {
            'prediction': 'pathological' if pred_class == 1 else 'healthy',
            'class_id': pred_class,
            'confidence': confidence,
            'probabilities': {
                'healthy': probs[0, 0].item(),
                'pathological': probs[0, 1].item()
            }
        }
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'feature_extractor_dim': self.feature_extractor.output_dim,
            'classifier_config': {
                'd_model': self.classifier.d_model,
                'num_classes': 2
            }
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model
