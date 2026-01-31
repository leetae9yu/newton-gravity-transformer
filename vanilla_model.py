import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaAttention(nn.Module):
    """Standard scaled dot-product multi-head attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, mask=None):
        B, L, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=attn_scores.device)
            attn_scores = attn_scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_dim)
        return self.out_proj(attn_output)


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


class VanillaBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = VanillaAttention(hidden_dim, num_heads, dropout=dropout)
        self.ffn = FeedForward(hidden_dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h, mask=None):
        h = h + self.attn(self.norm1(h), mask=mask)
        h = h + self.ffn(self.norm2(h))
        return h


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
            [VanillaBlock(hidden_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_tokens)

    def forward(self, x, mask=None):
        b, l = x.size()
        device = x.device

        h = self.token_emb(x)
        pos = torch.arange(l, device=device).unsqueeze(0).expand(b, l)
        h = h + self.pos_emb(pos)

        for layer in self.layers:
            h = layer(h, mask=mask)

        h = self.norm(h)
        logits = self.head(h)
        return logits


if __name__ == "__main__":
    model = VanillaTransformer(
        num_tokens=100,
        hidden_dim=64,
        num_layers=4,
        num_heads=8,
        mlp_dim=256,
    )

    x = torch.randint(0, 100, (2, 20))
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print("Vanilla Transformer Forward Pass Successful!")
