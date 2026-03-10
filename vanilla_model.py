import torch
import torch.nn as nn
import torch.nn.functional as F

from common import FeedForward


class VanillaAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=attn_scores.device)
            attn_scores = attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(attn_output)


class VanillaBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = VanillaAttention(hidden_dim, num_heads, dropout=dropout)
        self.ffn = FeedForward(hidden_dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, mask=None):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), mask=mask)
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        return hidden_states


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        hidden_dim,
        num_layers,
        num_heads,
        mlp_dim,
        max_seq_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.layers = nn.ModuleList(
            [VanillaBlock(hidden_dim, num_heads, mlp_dim, dropout=dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_tokens)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        device = x.device

        hidden_states = self.token_emb(x)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = hidden_states + self.pos_emb(positions)

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask)

        hidden_states = self.norm(hidden_states)
        return self.head(hidden_states)
