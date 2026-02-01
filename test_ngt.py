"""
NGT Model Test Suite
====================
Regression tests for Newton Gravity Transformer.
"""
import torch
import pytest
from common import build_causal_mask, FeedForward


def test_gravity_attention_forward():
    """Test GravityAttention forward pass shape."""
    from ngt_model import GravityAttention
    
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
    from ngt_model import GravityAttention
    
    B, L, D, Z, H = 2, 8, 32, 4, 2
    model = GravityAttention(hidden_dim=D, coord_dim=Z, num_heads=H)
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z)
    
    # Should not raise error
    h_new, z_new = model(h, z, mass=None)
    
    assert h_new.shape == (B, L, D)
    assert not torch.isnan(h_new).any()


def test_ngt_block_forward():
    """Test NGTBlock forward pass with mass."""
    from ngt_model import NGTBlock
    
    B, L, D, Z, H, M = 2, 16, 64, 8, 4, 128
    block = NGTBlock(hidden_dim=D, coord_dim=Z, num_heads=H, mlp_dim=M)
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z)
    mass = torch.ones(B, L, 1)
    
    h_new, z_new = block(h, z, mass=mass)
    
    assert h_new.shape == (B, L, D)
    assert z_new.shape == (B, L, Z)


def test_ngt_model_forward():
    """Test full NGT model forward pass."""
    from ngt_model import NewtonGravityTransformer
    
    num_tokens = 100
    model = NewtonGravityTransformer(
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
    from ngt_model import GravityAttention
    
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
    from ngt_model import GravityAttention
    
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


def test_ngt_return_signature():
    """Test NGT returns correct values based on flags."""
    from ngt_model import NewtonGravityTransformer
    
    num_tokens = 50
    model = NewtonGravityTransformer(
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
    from ngt_model import NewtonGravityTransformer
    
    num_tokens = 20
    model = NewtonGravityTransformer(
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


def test_vanilla_transformer_forward():
    """Test VanillaTransformer forward pass shape."""
    from vanilla_model import VanillaTransformer

    num_tokens = 100
    model = VanillaTransformer(
        num_tokens=num_tokens,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_dim=64,
    )
    x = torch.randint(0, num_tokens, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, num_tokens)


def test_vanilla_transformer_truncation():
    """Test VanillaTransformer truncates input exceeding max_seq_len at inference."""
    from vanilla_model import VanillaTransformer

    num_tokens = 50
    model = VanillaTransformer(
        num_tokens=num_tokens,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        mlp_dim=32,
        max_seq_len=8,
    )
    model.eval()
    x = torch.randint(0, num_tokens, (1, 16))  # exceeds max_seq_len=8
    logits = model(x)
    assert logits.shape == (1, 8, num_tokens)


def test_vanilla_transformer_truncation_raises_in_training():
    """Test VanillaTransformer raises error for oversized input during training."""
    from vanilla_model import VanillaTransformer

    num_tokens = 50
    model = VanillaTransformer(
        num_tokens=num_tokens,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        mlp_dim=32,
        max_seq_len=8,
    )
    model.train()
    x = torch.randint(0, num_tokens, (1, 16))
    with pytest.raises(ValueError):
        model(x)


def test_tokenizer_encode_decode_roundtrip():
    """Test char tokenizer encode/decode round-trip."""
    from tokenizer_utils import CharTokenizer

    text = "Hello, world!"
    tok = CharTokenizer.from_text(text)
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text


def test_tokenizer_save_load_roundtrip():
    """Test tokenizer state save/load preserves behaviour."""
    from tokenizer_utils import CharTokenizer, load_tokenizer

    text = "abcdef"
    tok = CharTokenizer.from_text(text)
    state = tok.save_state()
    tok2 = load_tokenizer(state)
    assert tok2.encode(text) == tok.encode(text)
    assert tok2.decode(tok.encode(text)) == text


def test_ngt_ablation_use_rsqrt():
    """Test NGT forward pass with use_rsqrt=True."""
    from ngt_model import NewtonGravityTransformer

    num_tokens = 50
    model = NewtonGravityTransformer(
        num_tokens=num_tokens,
        hidden_dim=16, coord_dim=4, num_layers=1, num_heads=2, mlp_dim=32,
        use_rsqrt=True,
    )
    x = torch.randint(0, num_tokens, (1, 5))
    logits = model(x)
    assert logits.shape == (1, 5, num_tokens)
    assert not torch.isnan(logits).any()


def test_ngt_ablation_mass_in_value():
    """Test NGT forward pass with mass_in_value=True."""
    from ngt_model import NewtonGravityTransformer

    num_tokens = 50
    model = NewtonGravityTransformer(
        num_tokens=num_tokens,
        hidden_dim=16, coord_dim=4, num_layers=1, num_heads=2, mlp_dim=32,
        mass_in_value=True,
    )
    x = torch.randint(0, num_tokens, (1, 5))
    logits = model(x)
    assert logits.shape == (1, 5, num_tokens)
    assert not torch.isnan(logits).any()


def test_ngt_ablation_soft_cutoff():
    """Test NGT forward pass with use_soft_cutoff=True."""
    from ngt_model import NewtonGravityTransformer

    num_tokens = 50
    model = NewtonGravityTransformer(
        num_tokens=num_tokens,
        hidden_dim=16, coord_dim=4, num_layers=1, num_heads=2, mlp_dim=32,
        use_soft_cutoff=True,
    )
    x = torch.randint(0, num_tokens, (1, 5))
    logits = model(x)
    assert logits.shape == (1, 5, num_tokens)
    assert not torch.isnan(logits).any()


def test_ngt_seq_len_1():
    """Test NGT forward pass with seq_len=1."""
    from ngt_model import NewtonGravityTransformer

    num_tokens = 50
    model = NewtonGravityTransformer(
        num_tokens=num_tokens,
        hidden_dim=16, coord_dim=4, num_layers=1, num_heads=2, mlp_dim=32,
    )
    x = torch.randint(0, num_tokens, (1, 1))
    logits = model(x)
    assert logits.shape == (1, 1, num_tokens)
    assert not torch.isnan(logits).any()


def test_ngt_batch_size_1():
    """Test NGT forward pass with batch_size=1."""
    from ngt_model import NewtonGravityTransformer

    num_tokens = 50
    model = NewtonGravityTransformer(
        num_tokens=num_tokens,
        hidden_dim=16, coord_dim=4, num_layers=1, num_heads=2, mlp_dim=32,
    )
    x = torch.randint(0, num_tokens, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, num_tokens)


def test_ngt_exceeding_max_seq_len():
    """Test NGT truncates input exceeding max_seq_len at inference."""
    from ngt_model import NewtonGravityTransformer

    num_tokens = 50
    model = NewtonGravityTransformer(
        num_tokens=num_tokens,
        hidden_dim=16, coord_dim=4, num_layers=1, num_heads=2, mlp_dim=32,
        max_seq_len=8,
    )
    model.eval()
    x = torch.randint(0, num_tokens, (1, 16))
    logits = model(x)
    assert logits.shape == (1, 8, num_tokens)


def test_repulsion_loss_normal_distribution():
    """Test repulsion loss gives reasonable values for normally distributed coords."""
    from train_shakespeare import compute_repulsion_loss

    B, L, Z = 2, 8, 8
    z = torch.randn(B, L, Z)
    loss = compute_repulsion_loss(z, min_dist=1e-3)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss > 0
    assert loss < 1e6  # should not be astronomically large


def test_sparse_attention_soft_cutoff():
    """Test soft cutoff path in sparse attention."""
    from ngt_model import GravityAttention

    B, L, D, Z, H = 1, 8, 16, 4, 2
    model = GravityAttention(
        hidden_dim=D, coord_dim=Z, num_heads=H,
        initial_radius=0.1, use_soft_cutoff=True,
    )
    model.eval()
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z) * 10
    mass = torch.ones(B, L, 1)

    _, _, stats = model(h, z, mass=mass, return_stats=True)
    assert "sparsity_ratio" in stats
    assert stats["sparsity_ratio"] >= 0.0


def test_build_causal_mask():
    """Test causal mask shape and values."""
    mask = build_causal_mask(4, "cpu")
    assert mask.shape == (1, 1, 4, 4)
    # Should be lower triangular
    assert mask[0, 0, 0, 1] == 0.0
    assert mask[0, 0, 1, 0] == 1.0
    assert mask[0, 0, 3, 3] == 1.0


def test_weight_tying_ngt():
    """Test NGT model uses weight tying between embedding and head."""
    from ngt_model import NewtonGravityTransformer

    model = NewtonGravityTransformer(
        num_tokens=50, hidden_dim=16, coord_dim=4,
        num_layers=1, num_heads=2, mlp_dim=32,
    )
    assert model.head.weight is model.token_emb.weight


def test_weight_tying_vanilla():
    """Test Vanilla model uses weight tying between embedding and head."""
    from vanilla_model import VanillaTransformer

    model = VanillaTransformer(
        num_tokens=50, hidden_dim=16,
        num_layers=1, num_heads=2, mlp_dim=32,
    )
    assert model.head.weight is model.token_emb.weight


def test_gravity_bias_exists():
    """Test that gravity_bias parameter exists in GravityAttention."""
    from ngt_model import GravityAttention

    model = GravityAttention(hidden_dim=16, coord_dim=4, num_heads=2)
    assert hasattr(model, "gravity_bias")
    assert model.gravity_bias.shape == (1, 2, 1, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
