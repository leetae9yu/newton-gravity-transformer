import argparse
import math
import os
import pickle
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from common import build_causal_mask
from ngt_model import NewtonGravityTransformer
from data_utils import load_dataset
from prepare_data import ensure_data
from tokenizer_utils import (
    build_tokenizer,
    load_tokenizer,
    load_tokenizer_from_path,
    save_tokenizer_to_path,
    train_bpe_tokenizer_from_iterator,
)


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
    x = x.to(device)
    y = y.to(device)
    # wikitext103 caches may be stored as int32 for memory efficiency; Embedding expects int64.
    if x.dtype != torch.long:
        x = x.long()
    if y.dtype != torch.long:
        y = y.long()
    return x, y


def compute_repulsion_loss(z, mass=None, alpha=2.0, min_dist=1e-3, max_samples=64):
    if z.dim() == 2:
        z = z.unsqueeze(0)

    # Sample tokens to avoid O(L^2) full pairwise computation
    seq_len = z.size(1)
    if seq_len > max_samples:
        idx = torch.randperm(seq_len, device=z.device)[:max_samples]
        z = z[:, idx, :]
        if mass is not None:
            mass = mass[:, idx, :]
        seq_len = max_samples

    # Squared distances directly (skip sqrt for speed)
    diff = z.unsqueeze(2) - z.unsqueeze(1)
    squared_dist = (diff * diff).sum(-1).clamp_min(min_dist * min_dist)
    # alpha=2 on Euclidean dist = alpha=1 on squared dist
    denom = squared_dist if alpha == 2.0 else squared_dist.pow(alpha / 2.0)

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

_WIKITEXT103_BASE = {
    "data_path": "data",
    "batch_size": 32,
    "block_size": 512,
    "max_steps": 100000,
    "eval_interval": 500,
    "eval_iters": 100,
    "learning_rate": 3e-4,
    "grad_clip": 1.0,
    "hidden_dim": 512,
    "num_layers": 8,
    "num_heads": 8,
    "mlp_dim": 2048,
    "dropout": 0.1,
}

WIKITEXT103_CONFIG = {
    **_WIKITEXT103_BASE,
    "coord_dim": 64,
    "checkpoint_path": os.path.join("checkpoints", "ngt_wikitext103.pt"),
}


def _apply_preset(args, preset, *default_dicts):
    """Apply preset defaults for arguments not explicitly set on the CLI.

    ``default_dicts`` are additional dicts whose values are treated as
    defaults (i.e. if the current attribute value equals any of them,
    it will be overwritten by the preset).
    """
    for key, value in preset.items():
        attr = key.replace("-", "_")
        current = getattr(args, attr, None)
        if current is None:
            setattr(args, attr, value)
            continue
        # Overwrite if current value matches any known default
        for d in (DEFAULT_CONFIG, *default_dicts):
            if current == d.get(key):
                setattr(args, attr, value)
                break
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Train NGT on TinyShakespeare.")
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        choices=["shakespeare", "wikitext103"],
                        help="Dataset to train on (default: shakespeare)")
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
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to tokenizer state JSON to load/save for reproducibility (recommended for wikitext103)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom TensorBoard run directory name (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of gradient accumulation steps (default: 1)")
    args = parser.parse_args()
    if args.dataset == "wikitext103":
        _apply_preset(args, WIKITEXT103_CONFIG)
    return args


def main():
    args = parse_args()

    # Seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.dataset == "shakespeare":
        data_path = ensure_data(args.data_path)
    else:
        data_path = args.data_path
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

    accum_steps = args.gradient_accumulation_steps

    # --- Tokenizer setup (may be overridden by checkpoint on resume) ---
    tokenizer = None
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        tokenizer = load_tokenizer_from_path(args.tokenizer_path)
        tok_type = tokenizer.save_state().get("type")
        if tok_type != args.tokenizer:
            print(f"Warning: --tokenizer={args.tokenizer} but tokenizer file is type={tok_type}; using tokenizer file.")

    if tokenizer is None:
        if args.dataset == "wikitext103":
            if args.tokenizer == "bpe":
                from datasets import load_dataset as hf_load_dataset
                print("Loading WikiText-103 text for BPE training...")
                ds = hf_load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
                iterator = (line for line in ds["text"] if line.strip())
                tokenizer = train_bpe_tokenizer_from_iterator(iterator, args.bpe_vocab_size, args.tokenizer_path)
                del ds
            else:
                tokenizer = build_tokenizer("", args.tokenizer, args.bpe_vocab_size)
            if args.tokenizer == "char":
                print(f"Warning: wikitext103 is best used with --tokenizer bpe or tiktoken (got {args.tokenizer})")
        else:
            text = read_text(data_path)
            tokenizer = build_tokenizer(text, args.tokenizer, args.bpe_vocab_size)

        if args.tokenizer_path and args.tokenizer != "bpe":
            save_tokenizer_to_path(tokenizer, args.tokenizer_path)
            print(f"Saved tokenizer to {args.tokenizer_path}")

    splits = load_dataset(args.dataset, tokenizer, data_path)
    train_data, val_data = splits["train"], splits["val"]
    text = splits.get("text")  # None for wikitext103
    vocab_size = tokenizer.vocab_size

    # Probe words only make sense for char-level tokenizer with shakespeare
    use_char_tokenizer = args.tokenizer == "char" and text is not None
    if use_char_tokenizer:
        _, stoi, itos = build_vocab(text)
        probe_word_ids = encode_probe_words(stoi)
    else:
        stoi, itos, probe_word_ids = None, None, {}

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

    # Auto-generate TensorBoard run directory name
    if args.run_name:
        run_dir = os.path.join("runs", args.run_name)
    else:
        ablation_parts = []
        if args.no_radius_cutoff:
            ablation_parts.append("no_radius")
        if args.no_repulsion:
            ablation_parts.append("no_repulsion")
        if args.use_rsqrt:
            ablation_parts.append("rsqrt")
        if args.mass_in_value:
            ablation_parts.append("mass_val")
        if args.use_soft_cutoff:
            ablation_parts.append("soft_cutoff")
        ablation_str = "_".join(ablation_parts) if ablation_parts else "default"
        seed_str = f"_s{args.seed}" if args.seed is not None else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", f"ngt_{args.dataset}_{ablation_str}{seed_str}_{timestamp}")
    writer = SummaryWriter(log_dir=run_dir)

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
                splits = load_dataset(args.dataset, tokenizer, data_path)
                train_data, val_data = splits["train"], splits["val"]
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
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(start_step * accum_steps, max_steps * accum_steps):
        x, y = get_batch(train_data, block_size, batch_size, device)
        effective_step = (micro_step + 1) // accum_steps  # 1-based
        is_accum_boundary = (micro_step + 1) % accum_steps == 0

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            compute_rep = use_repulsion and (effective_step % repulsion_interval == 0) if is_accum_boundary else False
            if use_repulsion:
                logits, z, m = model(x, mask=mask, return_last_coords=True)
                task_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                if compute_rep:
                    rep_loss = compute_repulsion_loss(z, mass=m)
                    rep_loss_item = float(rep_loss.detach())
                    loss = (task_loss + lambda_repulsion * rep_loss) / accum_steps
                else:
                    rep_loss_item = 0.0
                    loss = task_loss / accum_steps
            else:
                logits = model(x, mask=mask)
                if isinstance(logits, tuple):
                    logits = logits[0]
                task_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                rep_loss_item = 0.0
                loss = task_loss / accum_steps

        scaler.scale(loss).backward()

        if not is_accum_boundary:
            continue

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        step = effective_step - 1  # 0-based for compatibility with eval checks

        if vis_interval and vis_interval > 0 and effective_step % vis_interval == 0 and use_repulsion:
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
                global_step=effective_step,
            )
            if probe_word_ids and use_char_tokenizer:
                record_probe_snapshot(
                    x.cpu(),
                    z.cpu(),
                    m.cpu(),
                    probe_word_ids,
                    itos,
                    effective_step,
                    gravity_evolution_path,
                )

        if effective_step % eval_interval == 0:
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
                f"step {effective_step}/{max_steps} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"rep_loss={eval_rep_loss:.4f} "
                f"elapsed={elapsed:.1f}s"
            )
            writer.add_scalar("loss/train", train_loss, effective_step)
            writer.add_scalar("loss/val", val_loss, effective_step)
            writer.add_scalar("loss/repulsion", eval_rep_loss, effective_step)

            # Learning rate and gradient norm
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", current_lr, effective_step)
            writer.add_scalar("grad_norm", grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, effective_step)

            # NGT-specific: gamma and radius distributions
            gammas = []
            radii = []
            for layer in model.layers:
                if hasattr(layer.attn, "gamma"):
                    gammas.append(torch.nn.functional.softplus(layer.attn.gamma).detach())
                if hasattr(layer.attn, "radius_param"):
                    radii.append(torch.nn.functional.softplus(layer.attn.radius_param).detach())
            if gammas:
                all_gamma = torch.cat([g.flatten() for g in gammas])
                writer.add_scalar("ngt/gamma_mean", all_gamma.mean().item(), effective_step)
                writer.add_scalar("ngt/gamma_std", all_gamma.std().item(), effective_step)
            if radii:
                all_radius = torch.stack(radii)
                writer.add_scalar("ngt/radius_mean", all_radius.mean().item(), effective_step)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "iter": effective_step,
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
                            "dataset": args.dataset,
                            "use_radius_cutoff": use_radius_cutoff,
                            "use_rsqrt": use_rsqrt,
                            "mass_in_value": mass_in_value,
                            "use_soft_cutoff": use_soft_cutoff,
                            "seed": args.seed,
                            "model_type": "ngt",
                            "gradient_accumulation_steps": accum_steps,
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
                "dataset": args.dataset,
                "use_radius_cutoff": use_radius_cutoff,
                "use_rsqrt": use_rsqrt,
                "mass_in_value": mass_in_value,
                "use_soft_cutoff": use_soft_cutoff,
                "seed": args.seed,
                "model_type": "ngt",
                "gradient_accumulation_steps": accum_steps,
            },
            "vocab": tokenizer.save_state(),
        },
        last_checkpoint_path,
    )

    # Log hyperparameters and final metrics for TensorBoard HParams tab
    final_train_loss = estimate_loss(
        model, train_data, block_size, batch_size, device, mask, criterion, eval_iters
    )
    hparam_dict = {
        "dataset": args.dataset,
        "hidden_dim": hidden_dim,
        "coord_dim": coord_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "mlp_dim": mlp_dim,
        "block_size": block_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "use_radius_cutoff": use_radius_cutoff,
        "use_repulsion": use_repulsion,
        "use_rsqrt": use_rsqrt,
        "mass_in_value": mass_in_value,
        "use_soft_cutoff": use_soft_cutoff,
        "use_cosine_schedule": args.use_cosine_schedule,
        "warmup_steps": args.warmup_steps,
        "tokenizer": args.tokenizer,
        "seed": args.seed if args.seed is not None else -1,
        "gradient_accumulation_steps": accum_steps,
    }
    metric_dict = {
        "best_val_loss": best_val,
        "final_train_loss": final_train_loss,
    }
    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()


if __name__ == "__main__":
    main()
