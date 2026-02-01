import argparse
import math
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from vanilla_model import VanillaTransformer
from data_utils import load_dataset
from prepare_data import ensure_data
from tokenizer_utils import build_tokenizer, load_tokenizer
from train_shakespeare import (
    read_text,
    build_vocab,
    encode,
    build_causal_mask,
    get_batch,
    estimate_loss,
    DEFAULT_CONFIG,
    _WIKITEXT103_BASE,
    _apply_preset,
)


WIKITEXT103_VANILLA_CONFIG = {
    **_WIKITEXT103_BASE,
    "checkpoint_path": os.path.join("checkpoints", "vanilla_wikitext103.pt"),
}


VANILLA_DEFAULTS = {
    "checkpoint_path": os.path.join("checkpoints", "vanilla_shakespeare.pt"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vanilla Transformer on TinyShakespeare.")
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        choices=["shakespeare", "wikitext103"],
                        help="Dataset to train on (default: shakespeare)")
    parser.add_argument("--data-path", default=DEFAULT_CONFIG["data_path"])
    parser.add_argument("--checkpoint-path", default=os.path.join("checkpoints", "vanilla_shakespeare.pt"))
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--block-size", type=int, default=DEFAULT_CONFIG["block_size"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_CONFIG["eval_interval"])
    parser.add_argument("--eval-iters", type=int, default=DEFAULT_CONFIG["eval_iters"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_CONFIG["grad_clip"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--num-heads", type=int, default=DEFAULT_CONFIG["num_heads"])
    parser.add_argument("--mlp-dim", type=int, default=DEFAULT_CONFIG["mlp_dim"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
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
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom TensorBoard run directory name (default: auto-generated)")
    args = parser.parse_args()
    if args.dataset == "wikitext103":
        _apply_preset(args, WIKITEXT103_VANILLA_CONFIG, VANILLA_DEFAULTS)
    return args


def main():
    args = parse_args()

    if args.dataset == "shakespeare":
        data_path = ensure_data(args.data_path)
    else:
        data_path = args.data_path
    checkpoint_path = args.checkpoint_path
    best_checkpoint_path = f"{checkpoint_path}_best.pt"
    last_checkpoint_path = f"{checkpoint_path}_last.pt"

    batch_size = args.batch_size
    block_size = args.block_size
    max_steps = args.max_steps
    eval_interval = args.eval_interval
    eval_iters = args.eval_iters
    learning_rate = args.learning_rate
    grad_clip = args.grad_clip

    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    num_heads = args.num_heads
    mlp_dim = args.mlp_dim
    dropout = args.dropout

    use_amp = args.use_amp and torch.cuda.is_available()
    if args.use_amp and not torch.cuda.is_available():
        print("Warning: --use-amp ignored (CUDA not available)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Tokenizer setup ---
    if args.dataset == "wikitext103":
        tokenizer = build_tokenizer("", args.tokenizer, args.bpe_vocab_size)
        if args.tokenizer != "tiktoken":
            print(f"Warning: wikitext103 is best used with --tokenizer tiktoken (got {args.tokenizer})")
    else:
        text = read_text(data_path)
        tokenizer = build_tokenizer(text, args.tokenizer, args.bpe_vocab_size)

    splits = load_dataset(args.dataset, tokenizer, data_path)
    train_data, val_data = splits["train"], splits["val"]
    vocab_size = tokenizer.vocab_size

    model = VanillaTransformer(
        num_tokens=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        max_seq_len=block_size,
        dropout=dropout,
    ).to(device)

    ngt_params = None
    ngt_best = os.path.join("checkpoints", "shakespeare.pt_best.pt")
    if os.path.exists(ngt_best):
        ckpt = torch.load(ngt_best, map_location="cpu", weights_only=False)
        if "model_state" in ckpt:
            ngt_params = sum(p.numel() for p in ckpt["model_state"].values())
    vanilla_params = sum(p.numel() for p in model.parameters())
    print(f"Vanilla Transformer parameters: {vanilla_params:,}")
    if ngt_params:
        print(f"NGT parameters (from checkpoint): {ngt_params:,}")
    print(f"Config: hidden_dim={hidden_dim}, num_layers={num_layers}, "
          f"num_heads={num_heads}, mlp_dim={mlp_dim}, block_size={block_size}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    mask = build_causal_mask(block_size, device)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Auto-generate TensorBoard run directory name
    if args.run_name:
        run_dir = os.path.join("runs", args.run_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", f"vanilla_{args.dataset}_{timestamp}")
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

    if args.resume:
        for candidate in (last_checkpoint_path, best_checkpoint_path, checkpoint_path):
            if os.path.exists(candidate):
                checkpoint = torch.load(candidate, map_location=device, weights_only=False)
                # Restore tokenizer from checkpoint to avoid retraining BPE
                tokenizer = load_tokenizer(checkpoint["vocab"])
                vocab_size = tokenizer.vocab_size
                splits = load_dataset(args.dataset, tokenizer, data_path)
                train_data, val_data = splits["train"], splits["val"]
                # Rebuild model with restored vocab size
                model = VanillaTransformer(
                    num_tokens=vocab_size,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    max_seq_len=block_size,
                    dropout=dropout,
                ).to(device)
                model.load_state_dict(checkpoint["model_state"])
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                start_step = checkpoint.get("iter", 0)
                best_val = checkpoint.get("best_val", best_val)
                mask = build_causal_mask(block_size, device)
                print(f"Resumed from {candidate} at step {start_step}")
                break

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    t0 = time.time()
    for step in range(start_step, max_steps):
        x, y = get_batch(train_data, block_size, batch_size, device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x, mask=mask)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        if (step + 1) % eval_interval == 0:
            train_loss = estimate_loss(
                model, train_data, block_size, batch_size, device, mask, criterion, eval_iters
            )
            val_loss = estimate_loss(
                model, val_data, block_size, batch_size, device, mask, criterion, eval_iters
            )
            elapsed = time.time() - t0
            print(
                f"step {step + 1}/{max_steps} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"elapsed={elapsed:.1f}s"
            )
            writer.add_scalar("loss/train", train_loss, step + 1)
            writer.add_scalar("loss/val", val_loss, step + 1)

            # Learning rate and gradient norm
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", current_lr, step + 1)
            writer.add_scalar("grad_norm", grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, step + 1)

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
                            "num_layers": num_layers,
                            "num_heads": num_heads,
                            "mlp_dim": mlp_dim,
                            "max_seq_len": block_size,
                            "dropout": dropout,
                            "vocab_size": vocab_size,
                            "dataset": args.dataset,
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
                "num_layers": num_layers,
                "num_heads": num_heads,
                "mlp_dim": mlp_dim,
                "max_seq_len": block_size,
                "dropout": dropout,
                "vocab_size": vocab_size,
                "dataset": args.dataset,
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
        "num_layers": num_layers,
        "num_heads": num_heads,
        "mlp_dim": mlp_dim,
        "block_size": block_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "use_cosine_schedule": args.use_cosine_schedule,
        "warmup_steps": args.warmup_steps,
        "tokenizer": args.tokenizer,
    }
    metric_dict = {
        "best_val_loss": best_val,
        "final_train_loss": final_train_loss,
    }
    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()
    print(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
