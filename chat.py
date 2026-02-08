import argparse
import math
import os

import torch
import torch.nn.functional as F

from common import build_causal_mask
from ngt_model import NewtonGravityTransformer
from tokenizer_utils import load_tokenizer

def is_legacy_coord_proj(state_dict, coord_dim, hidden_dim):
    """
    Detect older checkpoints where coord_proj_next was Linear(coord_dim, coord_dim).
    """
    for name, tensor in state_dict.items():
        if name.endswith("attn.coord_proj_next.weight"):
            # Legacy: weight shape [coord_dim, coord_dim]; Current: [coord_dim, hidden_dim]
            if tensor.shape[1] == coord_dim and tensor.shape[1] != hidden_dim:
                return True
    return False


@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, device, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        mask = build_causal_mask(idx_cond.size(1), device)
        logits = model(idx_cond, mask=mask)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = torch.finfo(logits.dtype).min

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return idx


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with NGT model.")
    parser.add_argument(
        "--checkpoint-path", 
        type=str, 
        default=os.path.join("checkpoints", "w3_25m", "ngt_mass_in_value.pt_best.pt"),
        help="Path to the model checkpoint file."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint_path
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer = load_tokenizer(checkpoint["vocab"])
    vocab_size = tokenizer.vocab_size
    hidden_dim = config["hidden_dim"]
    state_dict = checkpoint["model_state"]

    # Auto-detect model type: coord_dim exists in config -> NGT
    is_ngt = "coord_dim" in config

    if is_ngt:
        coord_dim = config["coord_dim"]
        model = NewtonGravityTransformer(
            num_tokens=vocab_size,
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

        # Handle legacy coord_proj_next shape
        if is_legacy_coord_proj(state_dict, coord_dim=coord_dim, hidden_dim=hidden_dim):
            for layer in model.layers:
                layer.attn.coord_proj_next = torch.nn.Linear(coord_dim, coord_dim).to(device)
            print("Detected legacy coord_proj_next; swapped to compatible layers.")

        # Load with strict=False to tolerate missing legacy keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys during load (handled with defaults): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys ignored during load: {unexpected_keys}")

        # If legacy checkpoint lacks mass embeddings, initialize them to yield mass ~1.0
        has_mass_weights = any(k.startswith("mass_emb.") for k in state_dict.keys())
        if not has_mass_weights:
            neutral_mass = math.log(math.expm1(1.0))  # inverse softplus for target mass=1
            model.mass_emb.weight.data.fill_(neutral_mass)
            print("Initialized mass_emb to neutral mass (1.0) for legacy checkpoint.")

        print("Model type: NGT (Newton Gravity Transformer)")
    else:
        print("This repository is now NGT-only. Non-NGT checkpoints are not supported.")
        return

    block_size = config["max_seq_len"]
    dataset_name = config.get("dataset", "shakespeare")

    print(f"Dataset: {dataset_name}")
    print("Loaded checkpoint. Type /quit to exit.")
    while True:
        prompt = input("prompt> ")
        if prompt.strip() in {"/quit", "/exit"}:
            break

        if prompt == "":
            prompt = "\n"

        encoded = tokenizer.encode(prompt)
        idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
        out = generate(
            model,
            idx,
            max_new_tokens=400,
            block_size=block_size,
            device=device,
            temperature=0.9,
            top_k=40,
        )
        completion = tokenizer.decode(out[0].tolist())
        print(completion)


if __name__ == "__main__":
    main()
