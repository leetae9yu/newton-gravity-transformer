import argparse
import math
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from common import build_causal_mask
from ngt_model import NewtonGravityTransformer
from prepare_data import ensure_data
from tokenizer_utils import build_tokenizer, load_tokenizer


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def get_batch(data, block_size, batch_size, device):
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            f"Dataset too small for block_size={block_size} (len={len(data)})."
        )
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def compute_repulsion_loss(z, mass=None, alpha=2.0, min_dist=1e-3):
    if z.dim() == 2:
        z = z.unsqueeze(0)

    dists = torch.cdist(z, z, p=2).clamp_min(min_dist)
    denom = dists.pow(alpha)

    if mass is None:
        mass_tensor = torch.ones(z.shape[:-1], device=z.device, dtype=z.dtype)
    else:
        mass_tensor = mass.to(z.device, dtype=z.dtype)
        if mass_tensor.dim() == 1:
            mass_tensor = mass_tensor.unsqueeze(0)
        if mass_tensor.dim() == 3 and mass_tensor.size(-1) == 1:
            mass_tensor = mass_tensor.squeeze(-1)

    mass_products = mass_tensor.unsqueeze(-1) * mass_tensor.unsqueeze(-2)
    energy = mass_products / denom

    seq_len = z.size(1)
    pair_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=z.device, dtype=torch.bool), diagonal=1
    ).unsqueeze(0)
    pair_mask = pair_mask.expand_as(energy)

    pairwise = energy.masked_select(pair_mask)
    if pairwise.numel() == 0:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    return pairwise.mean()


PROBE_WORDS = {
    "royalty": ["KING", "QUEEN", "PRINCE", "PRINCESS"],
    "family": ["MOTHER", "FATHER", "SON", "DAUGHTER"],
    "antonyms": ["LOVE", "HATE", "WAR", "PEACE"],
}


def encode_probe_words(stoi):
    encoded = {}
    for group, words in PROBE_WORDS.items():
        valid_words = []
        for word in words:
            if all(ch in stoi for ch in word):
                valid_words.append((word, encode(word, stoi)))
        if valid_words:
            encoded[group] = valid_words
    return encoded


def record_probe_snapshot(x, z, m, probe_words, itos, step, path):
    records = []
    batch, seq_len = x.size()
    for b in range(batch):
        seq = x[b]
        for group, word_list in probe_words.items():
            for word, token_ids in word_list:
                word_len = token_ids.numel()
                if word_len == 0 or word_len > seq_len:
                    continue
                token_ids = token_ids.to(seq.device)
                for start in range(seq_len - word_len + 1):
                    window = seq[start : start + word_len]
                    if torch.equal(window, token_ids):
                        coords = z[b, start : start + word_len].detach().cpu()
                        masses = m[b, start : start + word_len].detach().cpu().squeeze(-1)
                        chars = [
                            itos[int(idx)] if int(idx) in itos else "<unk>"
                            for idx in window
                        ]
                        for idx_in_word in range(word_len):
                            records.append(
                                {
                                    "group": group,
                                    "word": word,
                                    "char": chars[idx_in_word],
                                    "position": idx_in_word,
                                    "coord": coords[idx_in_word].tolist(),
                                    "mass": float(masses[idx_in_word]),
                                }
                            )
    if not records:
        return

    snapshot = {"step": step, "records": records}
    # Append-only: write single snapshot to avoid O(n²) I/O
    with open(path, "ab") as f:
        pickle.dump(snapshot, f)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, mask, criterion, eval_iters):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        logits = model(x, mask=mask)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


DEFAULT_CONFIG = {
    "data_path": os.path.join("data", "input.txt"),
    "checkpoint_path": os.path.join("checkpoints", "shakespeare.pt"),
    "resume": False,
    "batch_size": 64,
    "block_size": 256,
    "max_steps": 5000,
    "eval_interval": 100,
    "eval_iters": 50,
    "vis_interval": None,
    "learning_rate": 3e-4,
    "grad_clip": 1.0,
    "hidden_dim": 256,
    "coord_dim": 32,
    "num_layers": 6,
    "num_heads": 8,
    "mlp_dim": 1024,
    "dropout": 0.1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train NGT on TinyShakespeare.")
    parser.add_argument("--data-path", default=DEFAULT_CONFIG["data_path"])
    parser.add_argument("--checkpoint-path", default=DEFAULT_CONFIG["checkpoint_path"])
    parser.add_argument("--resume", action="store_true", default=DEFAULT_CONFIG["resume"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--block-size", type=int, default=DEFAULT_CONFIG["block_size"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_CONFIG["eval_interval"])
    parser.add_argument("--eval-iters", type=int, default=DEFAULT_CONFIG["eval_iters"])
    parser.add_argument("--vis-interval", type=int, default=DEFAULT_CONFIG["vis_interval"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_CONFIG["grad_clip"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--coord-dim", type=int, default=DEFAULT_CONFIG["coord_dim"])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--num-heads", type=int, default=DEFAULT_CONFIG["num_heads"])
    parser.add_argument("--mlp-dim", type=int, default=DEFAULT_CONFIG["mlp_dim"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--no-radius-cutoff", action="store_true", default=False,
                        help="Disable radius cutoff in gravity attention")
    parser.add_argument("--no-repulsion", action="store_true", default=False,
                        help="Disable repulsion loss during training")
    parser.add_argument("--use-rsqrt", action="store_true", default=False,
                        help="Use rsqrt instead of division for gravity score (faster)")
    parser.add_argument("--mass-in-value", action="store_true", default=False,
                        help="Apply mass to value weighting instead of attention scores")
    parser.add_argument("--use-soft-cutoff", action="store_true", default=False,
                        help="Use smooth ReLU-style radius cutoff instead of hard masking")
    parser.add_argument("--lambda-repulsion", type=float, default=0.05,
                        help="Weight for repulsion loss (default: 0.05)")
    parser.add_argument("--repulsion-interval", type=int, default=1,
                        help="Compute repulsion loss every N steps (default: 1, i.e. every step)")
    parser.add_argument("--use-amp", action="store_true", default=False,
                        help="Enable Automatic Mixed Precision (FP16) for faster training on CUDA")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Number of linear warmup steps for LR scheduler (0 = no warmup)")
    parser.add_argument("--use-cosine-schedule", action="store_true", default=False,
                        help="Use cosine annealing LR schedule (with optional warmup)")
    parser.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe", "tiktoken"],
                        help="Tokenizer type: char (default), bpe, or tiktoken")
    parser.add_argument("--bpe-vocab-size", type=int, default=4000,
                        help="BPE vocabulary size (only used with --tokenizer bpe)")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = ensure_data(args.data_path)
    checkpoint_path = args.checkpoint_path
    best_checkpoint_path = f"{checkpoint_path}_best.pt"
    last_checkpoint_path = f"{checkpoint_path}_last.pt"
    gravity_evolution_path = os.path.join("checkpoints", "gravity_evolution.pkl")
    resume = args.resume

    batch_size = args.batch_size
    block_size = args.block_size
    max_steps = args.max_steps
    eval_interval = args.eval_interval
    eval_iters = args.eval_iters
    vis_interval = args.vis_interval if args.vis_interval is not None else eval_interval
    learning_rate = args.learning_rate
    grad_clip = args.grad_clip
    
    # Validate intervals to prevent ZeroDivisionError
    if eval_interval <= 0:
        raise ValueError(f"--eval-interval must be > 0, got {eval_interval}")
    if vis_interval <= 0:
        print(f"Warning: --vis-interval={vis_interval} is invalid, disabling visualization.")
        vis_interval = None
    lambda_repulsion = args.lambda_repulsion
    repulsion_interval = args.repulsion_interval

    hidden_dim = args.hidden_dim
    coord_dim = args.coord_dim
    num_layers = args.num_layers
    num_heads = args.num_heads
    mlp_dim = args.mlp_dim
    dropout = args.dropout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = read_text(data_path)

    # --- Tokenizer setup (may be overridden by checkpoint on resume) ---
    tokenizer = build_tokenizer(text, args.tokenizer, args.bpe_vocab_size)
    vocab_size = tokenizer.vocab_size
    data = tokenizer.encode_to_tensor(text)

    # Probe words only make sense for char-level tokenizer
    use_char_tokenizer = args.tokenizer == "char"
    if use_char_tokenizer:
        _, stoi, itos = build_vocab(text)
        probe_word_ids = encode_probe_words(stoi)
    else:
        stoi, itos, probe_word_ids = None, None, {}

    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    use_radius_cutoff = not args.no_radius_cutoff
    use_repulsion = not args.no_repulsion
    use_rsqrt = args.use_rsqrt
    mass_in_value = args.mass_in_value
    use_soft_cutoff = args.use_soft_cutoff
    use_amp = args.use_amp and torch.cuda.is_available()
    if args.use_amp and not torch.cuda.is_available():
        print("Warning: --use-amp ignored (CUDA not available)")

    model = NewtonGravityTransformer(
        num_tokens=vocab_size,
        hidden_dim=hidden_dim,
        coord_dim=coord_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        max_seq_len=block_size,
        dropout=dropout,
        use_radius_cutoff=use_radius_cutoff,
        use_rsqrt=use_rsqrt,
        mass_in_value=mass_in_value,
        use_soft_cutoff=use_soft_cutoff,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    mask = build_causal_mask(block_size, device)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    writer = SummaryWriter(log_dir="runs/ngt_experiment")

    # LR scheduler setup
    scheduler = None
    if args.use_cosine_schedule:
        warmup_steps = args.warmup_steps

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    best_val = float("inf")

    if resume:
        for candidate in (last_checkpoint_path, best_checkpoint_path, checkpoint_path):
            if os.path.exists(candidate):
                checkpoint = torch.load(candidate, map_location=device, weights_only=False)
                # Restore tokenizer from checkpoint to avoid retraining BPE
                tokenizer = load_tokenizer(checkpoint["vocab"])
                vocab_size = tokenizer.vocab_size
                data = tokenizer.encode_to_tensor(text)
                split_idx = int(0.9 * len(data))
                train_data = data[:split_idx]
                val_data = data[split_idx:]
                # Rebuild model with restored vocab size
                model = NewtonGravityTransformer(
                    num_tokens=vocab_size,
                    hidden_dim=hidden_dim,
                    coord_dim=coord_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    max_seq_len=block_size,
                    dropout=dropout,
                    use_radius_cutoff=use_radius_cutoff,
                    use_rsqrt=use_rsqrt,
                    mass_in_value=mass_in_value,
                    use_soft_cutoff=use_soft_cutoff,
                ).to(device)
                # Restore ablation flags from checkpoint config
                ckpt_config = checkpoint.get("config", {})
                use_rsqrt = ckpt_config.get("use_rsqrt", use_rsqrt)
                mass_in_value = ckpt_config.get("mass_in_value", mass_in_value)
                use_soft_cutoff = ckpt_config.get("use_soft_cutoff", use_soft_cutoff)
                use_radius_cutoff = ckpt_config.get("use_radius_cutoff", use_radius_cutoff)
                model.load_state_dict(checkpoint["model_state"], strict=False)
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                start_step = checkpoint.get("iter", 0)
                best_val = checkpoint.get("best_val", best_val)
                mask = build_causal_mask(block_size, device)
                break

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    gravity_dir = os.path.dirname(gravity_evolution_path)
    if gravity_dir:
        os.makedirs(gravity_dir, exist_ok=True)

    t0 = time.time()
    for step in range(start_step, max_steps):
        x, y = get_batch(train_data, block_size, batch_size, device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            compute_rep = use_repulsion and (step % repulsion_interval == 0)
            if use_repulsion:
                logits, z, m = model(x, mask=mask, return_last_coords=True)
                task_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                if compute_rep:
                    rep_loss = compute_repulsion_loss(z, mass=m)
                    rep_loss_item = float(rep_loss.detach())
                    loss = task_loss + lambda_repulsion * rep_loss
                else:
                    rep_loss_item = 0.0
                    loss = task_loss
            else:
                logits = model(x, mask=mask)
                if isinstance(logits, tuple):
                    logits = logits[0]
                task_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                rep_loss_item = 0.0
                loss = task_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        if vis_interval and vis_interval > 0 and (step + 1) % vis_interval == 0 and use_repulsion:
            flat_z = z.detach().reshape(-1, z.size(-1))
            tokens_flat = x.detach().reshape(-1)

            # Sample embeddings to prevent event file bloat
            max_embed_samples = 1024
            total_tokens = flat_z.size(0)
            if total_tokens > max_embed_samples:
                indices = torch.randperm(total_tokens)[:max_embed_samples]
                flat_z = flat_z[indices]
                tokens_flat = tokens_flat[indices]

            tokens_text = [tokenizer.decode([int(tok)]) or "<unk>" for tok in tokens_flat]
            writer.add_embedding(
                mat=flat_z,
                metadata=tokens_text,
                global_step=step + 1,
            )
            if probe_word_ids and use_char_tokenizer:
                record_probe_snapshot(
                    x.cpu(),
                    z.cpu(),
                    m.cpu(),
                    probe_word_ids,
                    itos,
                    step + 1,
                    gravity_evolution_path,
                )

        if (step + 1) % eval_interval == 0:
            train_loss = estimate_loss(
                model,
                train_data,
                block_size,
                batch_size,
                device,
                mask,
                criterion,
                eval_iters,
            )
            val_loss = estimate_loss(
                model,
                val_data,
                block_size,
                batch_size,
                device,
                mask,
                criterion,
                eval_iters,
            )
            # Compute rep_loss at evaluation time (not from last training batch)
            if use_repulsion:
                with torch.no_grad():
                    eval_x, _ = get_batch(val_data, block_size, batch_size, device)
                    _, eval_z, eval_m = model(eval_x, mask=mask, return_last_coords=True)
                    eval_rep_loss = compute_repulsion_loss(eval_z, mass=eval_m).item()
            else:
                eval_rep_loss = 0.0
            elapsed = time.time() - t0
            print(
                f"step {step + 1}/{max_steps} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"rep_loss={eval_rep_loss:.4f} "
                f"elapsed={elapsed:.1f}s"
            )
            writer.add_scalar("loss/train", train_loss, step + 1)
            writer.add_scalar("loss/val", val_loss, step + 1)
            writer.add_scalar("loss/repulsion", eval_rep_loss, step + 1)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "iter": step + 1,
                        "best_val": best_val,
                        "config": {
                            "hidden_dim": hidden_dim,
                            "coord_dim": coord_dim,
                            "num_layers": num_layers,
                            "num_heads": num_heads,
                            "mlp_dim": mlp_dim,
                            "max_seq_len": block_size,
                            "dropout": dropout,
                            "vocab_size": vocab_size,
                            "use_radius_cutoff": use_radius_cutoff,
                            "use_rsqrt": use_rsqrt,
                            "mass_in_value": mass_in_value,
                            "use_soft_cutoff": use_soft_cutoff,
                        },
                        "vocab": tokenizer.save_state(),
                    },
                    best_checkpoint_path,
                )

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iter": max_steps,
            "best_val": best_val,
            "config": {
                "hidden_dim": hidden_dim,
                "coord_dim": coord_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "mlp_dim": mlp_dim,
                "max_seq_len": block_size,
                "dropout": dropout,
                "vocab_size": vocab_size,
                "use_radius_cutoff": use_radius_cutoff,
                "use_rsqrt": use_rsqrt,
                "mass_in_value": mass_in_value,
                "use_soft_cutoff": use_soft_cutoff,
            },
            "vocab": tokenizer.save_state(),
        },
        last_checkpoint_path,
    )
    writer.close()


if __name__ == "__main__":
    main()
