"""
3D PCA visualization of NGT coordinate space (z) and mass (m).

Produces an interactive HTML file with a 3D scatter plot where:
  - Each point is a token
  - Point size encodes mass (heavier → larger)
  - Point colour encodes mass (colour scale)
  - Hover shows the token character
"""

import argparse
import math
import os

import torch
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go

from common import build_causal_mask
from ngt_model import NewtonGravityTransformer
from chat import is_legacy_coord_proj
from tokenizer_utils import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize NGT coordinate space via 3D PCA."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.path.join("checkpoints", "shakespeare.pt_best.pt"),
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to visualize. If omitted, sample from checkpoint data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualize_coords.html",
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to visualize.",
    )
    return parser.parse_args()


def load_model_and_vocab(checkpoint_path, device):
    """Load NGT checkpoint – mirrors the logic in chat.py."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    state_dict = checkpoint["model_state"]

    if "coord_dim" not in config:
        raise RuntimeError(
            "This checkpoint is a Vanilla Transformer; coordinate visualization "
            "requires an NGT checkpoint."
        )

    coord_dim = config["coord_dim"]
    hidden_dim = config["hidden_dim"]

    model = NewtonGravityTransformer(
        num_tokens=config["vocab_size"],
        hidden_dim=hidden_dim,
        coord_dim=coord_dim,
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        use_radius_cutoff=config.get("use_radius_cutoff", True),
        use_rsqrt=config.get("use_rsqrt", False),
        mass_in_value=config.get("mass_in_value", False),
        use_soft_cutoff=config.get("use_soft_cutoff", False),
    ).to(device)

    if is_legacy_coord_proj(state_dict, coord_dim=coord_dim, hidden_dim=hidden_dim):
        for layer in model.layers:
            layer.attn.coord_proj_next = torch.nn.Linear(coord_dim, coord_dim).to(device)
        print("Detected legacy coord_proj_next; swapped to compatible layers.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys (handled with defaults): {missing}")
    if unexpected:
        print(f"Unexpected keys ignored: {unexpected}")

    has_mass_weights = any(k.startswith("mass_emb.") for k in state_dict.keys())
    if not has_mass_weights:
        neutral_mass = math.log(math.expm1(1.0))
        model.mass_emb.weight.data.fill_(neutral_mass)
        print("Initialized mass_emb to neutral mass (1.0) for legacy checkpoint.")

    tokenizer = load_tokenizer(vocab)
    return model, config, tokenizer


def get_sample_text(checkpoint_path, num_tokens):
    """Fall back: grab text from the training data referenced by the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "data_path" in checkpoint:
        data_path = checkpoint["data_path"]
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                return f.read()[:num_tokens]

    # Try common Shakespeare path
    for candidate in ["data/shakespeare.txt", "data/input.txt"]:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return f.read()[:num_tokens]

    raise FileNotFoundError(
        "No --text provided and could not find training data. "
        "Please pass --text explicitly."
    )


@torch.no_grad()
def extract_coords(model, tokens, block_size, device):
    """Run a forward pass and return z (coords) and m (mass) arrays."""
    model.eval()
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    idx = idx[:, :block_size]
    mask = build_causal_mask(idx.size(1), device)
    _, z, m = model(idx, mask=mask, return_last_coords=True)
    # z: (1, seq_len, coord_dim), m: (1, seq_len, 1) or (1, seq_len)
    z = z.squeeze(0).cpu().numpy()   # (seq_len, coord_dim)
    m = m.squeeze(0).cpu().numpy()   # (seq_len,) or (seq_len, 1)
    if m.ndim > 1:
        m = m.squeeze(-1)
    return z, m


def build_figure(z_3d, masses, token_labels, explained_var):
    """Create a Plotly 3D scatter figure."""
    # Normalise mass for marker sizing (range 4–24)
    m_min, m_max = masses.min(), masses.max()
    if m_max - m_min > 1e-8:
        m_norm = (masses - m_min) / (m_max - m_min)
    else:
        m_norm = np.ones_like(masses) * 0.5
    marker_sizes = 4 + m_norm * 20

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=z_3d[:, 0],
                y=z_3d[:, 1],
                z=z_3d[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_sizes,
                    color=masses,
                    colorscale="Viridis",
                    colorbar=dict(title="Mass"),
                    opacity=0.85,
                    line=dict(width=0.5, color="white"),
                ),
                text=token_labels,
                hovertemplate=(
                    "Token: %{text}<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<br>"
                    "PC3: %{z:.3f}<br>"
                    "Mass: %{marker.color:.4f}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    var_text = (
        f"PC1: {explained_var[0]:.1%}  "
        f"PC2: {explained_var[1]:.1%}  "
        f"PC3: {explained_var[2]:.1%}  "
        f"(Total: {sum(explained_var):.1%})"
    )

    fig.update_layout(
        title=dict(text="NGT Coordinate Space – 3D PCA", x=0.5),
        scene=dict(
            xaxis_title=f"PC1 ({explained_var[0]:.1%})",
            yaxis_title=f"PC2 ({explained_var[1]:.1%})",
            zaxis_title=f"PC3 ({explained_var[2]:.1%})",
        ),
        annotations=[
            dict(
                text=var_text,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.05,
                font=dict(size=12),
            )
        ],
        margin=dict(l=0, r=0, b=40, t=50),
    )
    return fig


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found: {args.checkpoint_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, tokenizer = load_model_and_vocab(args.checkpoint_path, device)
    block_size = config["max_seq_len"]

    # Resolve input text
    if args.text is not None:
        text = args.text
    else:
        text = get_sample_text(args.checkpoint_path, args.num_tokens)

    text = text[: args.num_tokens]
    tokens = tokenizer.encode(text)
    print(f"Visualizing {len(tokens)} tokens on {device}.")

    # Forward pass
    z, masses = extract_coords(model, tokens, block_size, device)

    # PCA 축소
    pca = PCA(n_components=3)
    z_3d = pca.fit_transform(z)
    explained_var = pca.explained_variance_ratio_

    # Token labels (make whitespace visible)
    token_labels = []
    for tok_id in tokens:
        decoded = tokenizer.decode([tok_id])
        label = decoded.replace(" ", "⎵").replace("\n", "↵").replace("\t", "⇥")
        token_labels.append(label if label else "<unk>")

    fig = build_figure(z_3d, masses, token_labels, explained_var)
    fig.write_html(args.output)
    print(f"Saved interactive 3D plot to {args.output}")


if __name__ == "__main__":
    main()
