"""
Shared utilities used by both NGT and Vanilla Transformer models.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


def build_causal_mask(seq_len, device):
    """Build a lower-triangular causal mask: (1, 1, seq_len, seq_len)."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
