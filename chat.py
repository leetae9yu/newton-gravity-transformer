import math
import os

import torch
import torch.nn.functional as F

from hgt_model import HierarchicalGravityTransformer


def build_causal_mask(seq_len, device):
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)


def encode(text, stoi, fallback_token):
    return [stoi.get(ch, fallback_token) for ch in text]


def decode(tokens, itos):
    return "".join(itos[i] for i in tokens)

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
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -1e9

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return idx


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with HGT model.")
    parser.add_argument(
        "--checkpoint-path", 
        type=str, 
        default=os.path.join("checkpoints", "shakespeare.pt"),
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    stoi = vocab["stoi"]
    itos = vocab["itos"]
    vocab_size = config["vocab_size"]
    hidden_dim = config["hidden_dim"]
    coord_dim = config["coord_dim"]

    model = HierarchicalGravityTransformer(
        num_tokens=vocab_size,
        hidden_dim=hidden_dim,
        coord_dim=coord_dim,
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
    ).to(device)
    state_dict = checkpoint["model_state"]

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

    block_size = config["max_seq_len"]
    fallback_token = stoi.get(" ", 0)

    print("Loaded checkpoint. Type /quit to exit.")
    while True:
        prompt = input("prompt> ")
        if prompt.strip() in {"/quit", "/exit"}:
            break

        if prompt == "":
            prompt = "\n"

        encoded = encode(prompt, stoi, fallback_token)
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
        completion = decode(out[0].tolist(), itos)
        print(completion)


if __name__ == "__main__":
    main()
