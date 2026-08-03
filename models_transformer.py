"""
Federated Transformer Model for Intrusion Detection
Implements a Transformer Encoder with Self-Attention for tabular time-series data.
Compatible with Federated Averaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import OrderedDict


class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    Uses sine and cosine functions of different frequencies.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model).
            
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FedTransformerEncoder(nn.Module):
    """
    Federated Transformer Encoder for Intrusion Detection.
    
    Architecture:
        - Input Embedding (Linear projection)
        - Positional Encoding
        - N Transformer Encoder Layers (Multi-Head Self-Attention + Feed Forward)
        - Global Average Pooling
        - Classification Head
    
    Compatible with Federated Averaging via standard state_dict methods.
    """
    
    def __init__(
        self, 
        input_dim: int,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        """
        Initialize the Federated Transformer model.
        
        Args:
            input_dim: Number of input features.
            d_model: Dimension of the model (embedding size).
            nhead: Number of attention heads.
            num_encoder_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of feedforward network in transformer.
            dropout: Dropout probability.
            output_dim: Output dimension (1 for binary classification).
        """
        super(FedTransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input embedding: project features to d_model
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Input shape: (batch, seq, feature)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier[0].weight.data.uniform_(-initrange, initrange)
        self.classifier[2].weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim).
               For tabular data, seq_len=1 or can be treated as sequence of features.
            
        Returns:
            Predictions (batch_size, output_dim).
        """
        # Handle 2D input (batch, features) by adding sequence dimension
        if x.dim() == 2:
            # Option 1: Treat each feature as a sequence element
            # Reshape: (batch, features) -> (batch, features, 1) then embed
            x = x.unsqueeze(-1)  # (batch, input_dim, 1)
            # Transpose to (batch, seq=input_dim, feature=1)
            x = x.transpose(1, 2)  # (batch, 1, input_dim) - this doesn't help
            
            # Better approach: Treat entire row as single sequence with feature embedding
            x = x.unsqueeze(1)  # (batch, seq=1, input_dim)
        
        # Embed input to d_model
        x = self.embedding(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq, d_model)
        
        # Pass through transformer encoder
        # No mask needed for full attention
        x = self.transformer_encoder(x)  # (batch, seq, d_model)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch, output_dim)
        
        return logits.squeeze(-1)  # (batch,) for BCEWithLogitsLoss
    
    def get_weights(self) -> OrderedDict[str, torch.Tensor]:
        """
        Retrieve model weights for federated averaging.
        
        Returns:
            OrderedDict containing the model's state dictionary.
        """
        return self.state_dict()
    
    def set_weights(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        """
        Load weights from federated averaging.
        
        Args:
            state_dict: OrderedDict containing aggregated weights.
        """
        self.load_state_dict(state_dict)


class FedTransformerTabular(nn.Module):
    """
    Alternative Transformer architecture specifically designed for tabular data.
    Treats each feature as a token in the sequence.
    """
    
    def __init__(
        self, 
        input_dim: int,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        super(FedTransformerTabular, self).__init__()
        
        # Each feature becomes a token
        self.feature_embedding = nn.Linear(1, d_model)  # Embed each scalar feature
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass treating features as sequence tokens.
        
        Args:
            x: Input (batch_size, input_dim).
            
        Returns:
            Predictions (batch_size,).
        """
        # Reshape: (batch, input_dim) -> (batch, input_dim, 1)
        x = x.unsqueeze(-1)  # (batch, seq=input_dim, 1)
        
        # Embed each feature token
        x = self.feature_embedding(x)  # (batch, input_dim, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, input_dim, d_model)
        
        # Global pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        return self.classifier(x).squeeze(-1)
    
    def get_weights(self) -> OrderedDict[str, torch.Tensor]:
        return self.state_dict()
    
    def set_weights(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)
