"""
HGT Model Test Suite
====================
Regression tests for Hierarchical Gravity Transformer.
"""
import torch
import pytest


def test_gravity_attention_forward():
    """Test GravityAttention forward pass shape."""
    from hgt_model import GravityAttention
    
    B, L, D, Z, H = 2, 16, 64, 8, 4
    model = GravityAttention(hidden_dim=D, coord_dim=Z, num_heads=H)
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z)
    mass = torch.ones(B, L, 1)
    
    h_new, z_new = model(h, z, mass=mass)
    
    assert h_new.shape == (B, L, D)
    assert z_new.shape == (B, L, Z)


def test_gravity_attention_without_mass():
    """Test GravityAttention works without mass (backward compatible)."""
    from hgt_model import GravityAttention
    
    B, L, D, Z, H = 2, 8, 32, 4, 2
    model = GravityAttention(hidden_dim=D, coord_dim=Z, num_heads=H)
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z)
    
    # Should not raise error
    h_new, z_new = model(h, z, mass=None)
    
    assert h_new.shape == (B, L, D)
    assert not torch.isnan(h_new).any()


def test_hgt_block_forward():
    """Test HGTBlock forward pass with mass."""
    from hgt_model import HGTBlock
    
    B, L, D, Z, H, M = 2, 16, 64, 8, 4, 128
    block = HGTBlock(hidden_dim=D, coord_dim=Z, num_heads=H, mlp_dim=M)
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z)
    mass = torch.ones(B, L, 1)
    
    h_new, z_new = block(h, z, mass=mass)
    
    assert h_new.shape == (B, L, D)
    assert z_new.shape == (B, L, Z)


def test_hgt_model_forward():
    """Test full HGT model forward pass."""
    from hgt_model import HierarchicalGravityTransformer
    
    num_tokens = 100
    model = HierarchicalGravityTransformer(
        num_tokens=num_tokens, 
        hidden_dim=32, 
        coord_dim=8, 
        num_layers=2, 
        num_heads=4, 
        mlp_dim=64
    )
    
    x = torch.randint(0, num_tokens, (2, 10))
    logits = model(x)
    
    assert logits.shape == (2, 10, num_tokens)


def test_gravity_attention_masking():
    """Test causal masking prevents attending to future tokens."""
    from hgt_model import GravityAttention
    
    B, L, D, Z, H = 1, 4, 16, 4, 2
    model = GravityAttention(hidden_dim=D, coord_dim=Z, num_heads=H, dropout=0.0)
    model.eval()
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z)
    mass = torch.ones(B, L, 1)
    
    # Causal mask
    mask = torch.tril(torch.ones(L, L)).unsqueeze(0).unsqueeze(0)
    
    h_new, z_new, attn_weights = model(h, z, mass=mass, mask=mask, return_attn=True)
    future_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    
    assert attn_weights[..., future_mask].max().item() <= 1e-6
    assert h_new.shape == (B, L, D)


def test_sparse_attention_masking():
    """Test that learnable radius creates sparse attention."""
    from hgt_model import GravityAttention
    
    B, L, D, Z, H = 1, 8, 16, 4, 2
    # Small initial radius to force sparsity
    model = GravityAttention(hidden_dim=D, coord_dim=Z, num_heads=H, initial_radius=0.1)
    model.eval()
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z) * 10  # Spread out coordinates
    mass = torch.ones(B, L, 1)
    
    _, _, stats = model(h, z, mass=mass, return_stats=True)
    
    # With small radius and spread coordinates, sparsity should be high
    assert "sparsity_ratio" in stats
    assert stats["sparsity_ratio"] > 0.0  # Some pairs should be masked


def test_repulsion_loss_stability():
    """Test repulsion loss doesn't explode with identical coordinates."""
    from train_shakespeare import compute_repulsion_loss
    
    B, L, Z = 2, 4, 8
    z = torch.zeros(B, L, Z)  # All at origin
    loss = compute_repulsion_loss(z, min_dist=1e-3)
    
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss > 0


def test_hgt_return_signature():
    """Test HGT returns correct values based on flags."""
    from hgt_model import HierarchicalGravityTransformer
    
    num_tokens = 50
    model = HierarchicalGravityTransformer(
        num_tokens=num_tokens, 
        hidden_dim=16, 
        coord_dim=4, 
        num_layers=1, 
        num_heads=2, 
        mlp_dim=32
    )
    x = torch.randint(0, num_tokens, (1, 5))
    
    # Standard return
    logits = model(x)
    assert isinstance(logits, torch.Tensor)
    
    # Return last coords and mass
    logits, z, m = model(x, return_last_coords=True)
    assert logits.shape == (1, 5, num_tokens)
    assert z.shape == (1, 5, 4)
    assert m.shape == (1, 5, 1)
    assert (m >= 0).all()  # Mass should be positive (Softplus)


def test_gradient_flow():
    """Test gradients flow through all learnable parameters."""
    from hgt_model import HierarchicalGravityTransformer
    
    num_tokens = 20
    model = HierarchicalGravityTransformer(
        num_tokens=num_tokens,
        hidden_dim=16,
        coord_dim=4,
        num_layers=1,
        num_heads=2,
        mlp_dim=32
    )
    
    x = torch.randint(0, num_tokens, (1, 5))
    y = torch.randint(0, num_tokens, (1, 5))
    
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, num_tokens), y.view(-1))
    loss.backward()
    
    # Check key parameters have gradients
    assert model.layers[0].attn.gamma.grad is not None
    assert model.layers[0].attn.radius_param.grad is not None
    assert model.mass_emb.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
