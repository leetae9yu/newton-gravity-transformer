import torch
import torch.nn as nn
import torch.nn.functional as F

from common import FeedForward

class GravityAttention(nn.Module):
    """
    Newton Gravity Transformer (NGT) Gravity Attention.
    
    Features:
    - Mass-based attention: Score = -γ × (m_i × m_j) / (dist² + ε)
    - Learnable radius sparse attention: masks pairs where dist² > radius²
    - Fixed sparse causal pattern: local window + selected far links
    """
    def __init__(
        self,
        hidden_dim: int,
        coord_dim: int,
        num_heads: int,
        head_coord_dim: int = 16,
        dropout: float = 0.1,
        initial_radius: float = 4.0,
        dist_eps: float = 1e-6,
        use_radius_cutoff: bool = True,
        use_rsqrt: bool = True,
        mass_in_value: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.num_heads = num_heads
        self.head_coord_dim = head_coord_dim
        self.dist_eps = dist_eps
        self.use_radius_cutoff = use_radius_cutoff
        self.use_rsqrt = use_rsqrt
        self.mass_in_value = mass_in_value
        
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        self.local_window = 64
        self.far_offsets = (96, 128, 192, 256)

        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.coord_proj_attn = nn.Linear(coord_dim, num_heads * head_coord_dim)
        self.coord_proj_next = nn.Linear(hidden_dim, coord_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable gravity constant (per head)
        self.gamma = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        # Learnable bias for gravity scores (per head) — widens softmax dynamic range
        self.gravity_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        # Learnable radius for sparse attention (shared across heads)
        self.radius_param = nn.Parameter(torch.tensor(initial_radius))
        
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(dropout)

    def _build_sparse_neighbors(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        local_offsets = torch.arange(self.local_window, -1, -1, device=device)
        far_offsets = torch.tensor(self.far_offsets, device=device)
        offsets = torch.cat((local_offsets, far_offsets), dim=0)
        neighbor_indices = positions.unsqueeze(1) - offsets.unsqueeze(0)
        valid_mask = neighbor_indices >= 0
        neighbor_indices = neighbor_indices.clamp_min(0).long()
        return neighbor_indices, valid_mask

    def forward(self, hidden_states, coordinates, mass=None, mask=None, return_stats=False, return_attn=False):
        batch_size, seq_len, _ = hidden_states.size()
        neighbor_indices, sparse_pattern_mask = self._build_sparse_neighbors(seq_len, hidden_states.device)

        # Value Projection
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.transpose(1, 2)
        selected_value = value[:, :, neighbor_indices, :]

        # Coordinate Projection for Attention
        z_heads = self.coord_proj_attn(coordinates)
        z_heads = z_heads.view(batch_size, seq_len, self.num_heads, self.head_coord_dim)
        z_heads = z_heads.transpose(1, 2)
        query_z = z_heads.unsqueeze(3)
        selected_z = z_heads[:, :, neighbor_indices, :]

        # Sparse vectorized distance calculation: ||z_i - z_j||^2 over selected pairs only.
        query_norm_sq = (query_z * query_z).sum(dim=-1)
        selected_norm_sq = (selected_z * selected_z).sum(dim=-1)
        pairwise_dot = (query_z * selected_z).sum(dim=-1)
        squared_dist = query_norm_sq + selected_norm_sq - 2.0 * pairwise_dot
        squared_dist = squared_dist.clamp_min(0.0)

        # Learnable parameters
        gamma = self.softplus(self.gamma)
        radius = self.softplus(self.radius_param)
        
        # --- Gravity score computation ---
        # Base inverse-distance score
        if self.use_rsqrt:
            inv_dist = torch.rsqrt(squared_dist + self.dist_eps)
            base_score = -gamma * inv_dist * inv_dist
        else:
            base_score = -gamma / (squared_dist + self.dist_eps)

        # Apply learnable bias to widen softmax dynamic range
        base_score = base_score + self.gravity_bias

        # Mass handling: either in attention scores (default) or in value weighting
        if mass is not None and not self.mass_in_value:
            # Default: mass affects "who to attend to"
            mass_squeezed = mass.squeeze(-1)  # [B, L]
            selected_mass = mass_squeezed[:, neighbor_indices]
            mass_products = mass_squeezed.unsqueeze(-1) * selected_mass  # [B, L, K]
            mass_products = mass_products.unsqueeze(1)  # [B, 1, L, K]
            attn_scores = base_score * mass_products
        else:
            attn_scores = base_score

        if mass is not None and self.mass_in_value:
            # Mass as "broadcasting power": heavier tokens send more information
            selected_mass = mass[:, neighbor_indices, :]
            selected_value = selected_value * selected_mass.unsqueeze(1)

        # Sparse attention: radius-based cutoff
        sparse_mask = None
        if self.use_radius_cutoff:
            radius_squared = radius ** 2
            sparse_mask = squared_dist > radius_squared
            attn_scores = attn_scores.masked_fill(sparse_mask, torch.finfo(attn_scores.dtype).min)

        # Apply causal/padding mask if provided
        attn_scores = attn_scores.masked_fill(~sparse_pattern_mask.view(1, 1, seq_len, -1), torch.finfo(attn_scores.dtype).min)
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=attn_scores.device)
            gather_index = neighbor_indices.view(1, 1, seq_len, -1).expand(mask.size(0), mask.size(1), -1, -1)
            selected_mask = torch.gather(mask, -1, gather_index)
            attn_scores = attn_scores.masked_fill(~selected_mask, torch.finfo(attn_scores.dtype).min)

        # Softmax normalization
        # Pre-masking with dtype min already ensures masked positions get ~0 probability,
        # so post-masking re-normalization is unnecessary and can cause gradient artifacts.
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute statistics for logging
        stats = None
        if return_stats:
            eps = 1e-9
            entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
            # Sparsity ratio: fraction of pairs masked/suppressed by radius
            if self.use_radius_cutoff and sparse_mask is not None:
                total_pairs = sparse_mask.numel()
                masked_pairs = sparse_mask.sum().float()
                sparsity_ratio = masked_pairs / total_pairs
            else:
                sparsity_ratio = torch.tensor(0.0, device=hidden_states.device)
            stats = {
                "gamma_mean": gamma.mean(),
                "radius": radius,
                "dist_mean": squared_dist.mean(),
                "energy_mean": (gamma * squared_dist).mean(),
                "entropy_mean": entropy.mean(),
                "sparsity_ratio": sparsity_ratio,
            }

        attn_weights = self.dropout(attn_weights)
        attn_output = (attn_weights.unsqueeze(-1) * selected_value).sum(dim=-2)

        # Finalize Outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        updated_hidden = self.out_proj(attn_output)

        # Coordinate Evolution (uses post-attention output for informed updates)
        updated_coords = self.coord_proj_next(updated_hidden)

        if return_stats and return_attn:
            return updated_hidden, updated_coords, stats, attn_weights
        if return_stats:
            return updated_hidden, updated_coords, stats
        if return_attn:
            return updated_hidden, updated_coords, attn_weights
        return updated_hidden, updated_coords

class NGTBlock(nn.Module):
    def __init__(self, hidden_dim, coord_dim, num_heads, mlp_dim, dropout=0.1,
                 use_radius_cutoff=True, use_rsqrt=True, mass_in_value=False):
        super().__init__()
        self.attn = GravityAttention(
            hidden_dim, coord_dim, num_heads, dropout=dropout,
            use_radius_cutoff=use_radius_cutoff, use_rsqrt=use_rsqrt,
            mass_in_value=mass_in_value,
        )
        self.ffn = FeedForward(hidden_dim, mlp_dim, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Coordinate Evolution Norm (optional, but good for stability)
        self.coord_norm = nn.LayerNorm(coord_dim)

    def forward(self, h, z, mass=None, mask=None, return_stats=False):
        # Attention + Residual
        stats = None
        if return_stats:
            h_attn, z_next, stats = self.attn(self.norm1(h), z, mass=mass, mask=mask, return_stats=True)
        else:
            h_attn, z_next = self.attn(self.norm1(h), z, mass=mass, mask=mask)
        h = h + h_attn
        
        # FFN + Residual
        h = h + self.ffn(self.norm2(h))
        # Coordinate Update (Residual + Norm)
        z = self.coord_norm(z + z_next)
        
        if return_stats:
            return h, z, stats
        return h, z

class NewtonGravityTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        hidden_dim,
        coord_dim,
        num_layers,
        num_heads,
        mlp_dim,
        max_seq_len=512,
        dropout=0.1,
        use_radius_cutoff=True,
        use_rsqrt=True,
        mass_in_value=False,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, hidden_dim)
        self.mass_emb = nn.Embedding(num_tokens, 1)
        # Coordinate embedding - can be learned or based on absolute positions
        self.coord_emb = nn.Embedding(max_seq_len, coord_dim)
        self.mass_nonlinearity = nn.Softplus()

        self.layers = nn.ModuleList([
            NGTBlock(
                hidden_dim, coord_dim, num_heads, mlp_dim, dropout,
                use_radius_cutoff=use_radius_cutoff, use_rsqrt=use_rsqrt,
                mass_in_value=mass_in_value,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_tokens)
        # Weight tying: share input embedding and output projection weights
        self.head.weight = self.token_emb.weight

    def forward(self, x, mask=None, return_stats=False, return_last_coords=False):
        b, l = x.size()
        max_seq_len = self.coord_emb.num_embeddings
        if l > max_seq_len:
            if self.training:
                raise ValueError(
                    f"Input sequence length {l} exceeds max_seq_len {max_seq_len} during training."
                )
            x = x[:, -max_seq_len:]
            if mask is not None and mask.size(-1) >= max_seq_len:
                mask = mask[..., -max_seq_len:, -max_seq_len:]
            b, l = x.size()
        device = x.device
        
        # Initial states
        h = self.token_emb(x)
        m = self.mass_nonlinearity(self.mass_emb(x))
        
        # Initial coordinates based on position
        pos = torch.arange(l, device=device).unsqueeze(0).expand(b, l)
        z = self.coord_emb(pos)
        
        stats_list = []
        for layer in self.layers:
            if return_stats:
                h, z, layer_stats = layer(h, z, mass=m, mask=mask, return_stats=True)
                stats_list.append(layer_stats)
            else:
                h, z = layer(h, z, mass=m, mask=mask)
            
        h = self.norm(h)
        logits = self.head(h)
        if return_stats:
            stack = {
                key: torch.stack([s[key] for s in stats_list]).mean()
                for key in stats_list[0].keys()
            }
            if return_last_coords:
                return logits, z, m, stack
            return logits, stack
        if return_last_coords:
            return logits, z, m
        return logits

if __name__ == "__main__":
    # Test Full Model
    model = NewtonGravityTransformer(
        num_tokens=100, 
        hidden_dim=64, 
        coord_dim=16, 
        num_layers=4, 
        num_heads=8, 
        mlp_dim=256
    )
    
    x = torch.randint(0, 100, (2, 20))
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print("NGT Model Forward Pass Successful!")
