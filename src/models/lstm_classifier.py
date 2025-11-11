from typing import Tuple
import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """
    LSTM-based binary classifier for risk day prediction
    
    Args:
        in_dim: Number of input features per time step
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    
    def __init__(self, in_dim: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, T, F) where
               B = batch size T = sequence length F = features
        
        Returns:
            Logits tensor of shape (B,)
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits.squeeze(-1)

