import argparse
import math
import os
import time
from datetime import datetime

VANILLA_DEFAULT_CONFIG = {
    "data_path": "data",
    "checkpoint_path": os.path.join("checkpoints", "vanilla_wikitext2_bpe_8192.pt"),
    "batch_size": 64,
    "block_size": 256,
    "max_steps": 5000,
    "eval_interval": 100,
    "eval_iters": 50,
    "learning_rate": 3e-4,
    "grad_clip": 1.0,
    "hidden_dim": 256,
    "num_layers": 6,
    "num_heads": 8,
    "mlp_dim": 1024,
    "dropout": 0.1,
    "bpe_vocab_size": 8192,
    "tokenizer_path": os.path.join("data", "tokenizer_wikitext2_bpe_8192.json"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train vanilla Transformer.")
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2"], help="Dataset to train on (fixed: wikitext2)")
    parser.add_argument("--data-path", default=VANILLA_DEFAULT_CONFIG["data_path"])
    parser.add_argument("--checkpoint-path", default=VANILLA_DEFAULT_CONFIG["checkpoint_path"])
    parser.add_argument("--batch-size", type=int, default=VANILLA_DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--block-size", type=int, default=VANILLA_DEFAULT_CONFIG["block_size"])
    parser.add_argument("--max-steps", type=int, default=VANILLA_DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--eval-interval", type=int, default=VANILLA_DEFAULT_CONFIG["eval_interval"])
    parser.add_argument("--eval-iters", type=int, default=VANILLA_DEFAULT_CONFIG["eval_iters"])
    parser.add_argument("--learning-rate", type=float, default=VANILLA_DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--grad-clip", type=float, default=VANILLA_DEFAULT_CONFIG["grad_clip"])
    parser.add_argument("--hidden-dim", type=int, default=VANILLA_DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=VANILLA_DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--num-heads", type=int, default=VANILLA_DEFAULT_CONFIG["num_heads"])
    parser.add_argument("--mlp-dim", type=int, default=VANILLA_DEFAULT_CONFIG["mlp_dim"])
    parser.add_argument("--dropout", type=float, default=VANILLA_DEFAULT_CONFIG["dropout"])
    parser.add_argument("--use-amp", action="store_true", default=False, help="Enable Automatic Mixed Precision (FP16) for faster training on CUDA")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of linear warmup steps for LR scheduler (0 = no warmup)")
    parser.add_argument("--use-cosine-schedule", action="store_true", default=False, help="Use cosine annealing LR schedule (with optional warmup)")
    parser.add_argument("--bpe-vocab-size", type=int, default=VANILLA_DEFAULT_CONFIG["bpe_vocab_size"], help="BPE vocabulary size")
    parser.add_argument("--tokenizer-path", type=str, default=VANILLA_DEFAULT_CONFIG["tokenizer_path"], help="Path to BPE tokenizer state JSON to load/save for reproducibility")
    parser.add_argument("--run-name", type=str, default=None, help="Custom TensorBoard run directory name (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of gradient accumulation steps (default: 1)")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    from common import build_causal_mask
    from data_utils import load_dataset
    from tokenizer_utils import load_tokenizer_from_path, train_bpe_tokenizer_from_iterator
    from train import estimate_loss, get_batch
    from vanilla_model import VanillaTransformer

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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

    if batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {batch_size}")
    if block_size <= 0:
        raise ValueError(f"--block-size must be > 0, got {block_size}")
    if max_steps <= 0:
        raise ValueError(f"--max-steps must be > 0, got {max_steps}")
    if hidden_dim <= 0:
        raise ValueError(f"--hidden-dim must be > 0, got {hidden_dim}")
    if num_layers <= 0:
        raise ValueError(f"--num-layers must be > 0, got {num_layers}")
    if num_heads <= 0:
        raise ValueError(f"--num-heads must be > 0, got {num_heads}")
    if mlp_dim <= 0:
        raise ValueError(f"--mlp-dim must be > 0, got {mlp_dim}")
    if eval_interval <= 0:
        raise ValueError(f"--eval-interval must be > 0, got {eval_interval}")
    if eval_iters <= 0:
        raise ValueError(f"--eval-iters must be > 0, got {eval_iters}")
    if learning_rate <= 0:
        raise ValueError(f"--learning-rate must be > 0, got {learning_rate}")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"--dropout must be in [0, 1), got {dropout}")
    if grad_clip <= 0:
        raise ValueError(f"--grad-clip must be > 0, got {grad_clip}")
    if hidden_dim % num_heads != 0:
        raise ValueError(f"--hidden-dim must be divisible by --num-heads, got hidden_dim={hidden_dim}, num_heads={num_heads}")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError(f"--gradient-accumulation-steps must be > 0, got {args.gradient_accumulation_steps}")
    if args.bpe_vocab_size <= 0:
        raise ValueError(f"--bpe-vocab-size must be > 0, got {args.bpe_vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.use_amp and torch.cuda.is_available()
    if args.use_amp and not torch.cuda.is_available():
        print("Warning: --use-amp ignored (CUDA not available)")

    tokenizer = None
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        tokenizer = load_tokenizer_from_path(args.tokenizer_path)

    if tokenizer is None:
        from datasets import load_dataset as hf_load_dataset

        hf_variant = "wikitext-2-raw-v1"
        print(f"Loading {hf_variant} text for BPE training...")
        ds = hf_load_dataset("wikitext", hf_variant, split="train")
        iterator = (line for line in ds["text"] if line.strip())
        tokenizer = train_bpe_tokenizer_from_iterator(iterator, args.bpe_vocab_size, args.tokenizer_path)
        del ds

    splits = load_dataset(args.dataset, tokenizer, args.data_path)
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

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    mask = build_causal_mask(block_size, device)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    checkpoint_path = args.checkpoint_path
    best_checkpoint_path = f"{checkpoint_path}_best.pt"
    last_checkpoint_path = f"{checkpoint_path}_last.pt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if args.run_name:
        run_dir = os.path.join("runs", args.run_name)
    else:
        seed_str = f"_s{args.seed}" if args.seed is not None else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", f"vanilla_{args.dataset}{seed_str}_{timestamp}")
    writer = SummaryWriter(log_dir=run_dir)

    scheduler = None
    if args.use_cosine_schedule:
        warmup_steps = args.warmup_steps

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    accum_steps = args.gradient_accumulation_steps
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for micro_step in range(max_steps * accum_steps):
        x, y = get_batch(train_data, block_size, batch_size, device)
        effective_step = (micro_step + 1) // accum_steps
        is_accum_boundary = (micro_step + 1) % accum_steps == 0

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x, mask=mask)
            task_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
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

        if effective_step % eval_interval == 0:
            train_loss = estimate_loss(model, train_data, block_size, batch_size, device, mask, criterion, eval_iters)
            val_loss = estimate_loss(model, val_data, block_size, batch_size, device, mask, criterion, eval_iters)
            elapsed = time.time() - t0
            print(f"step {effective_step}/{max_steps} train_loss={train_loss:.4f} val_loss={val_loss:.4f} elapsed={elapsed:.1f}s")
            writer.add_scalar("loss/train", train_loss, effective_step)
            writer.add_scalar("loss/val", val_loss, effective_step)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", current_lr, effective_step)
            writer.add_scalar("grad_norm", grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, effective_step)

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
                            "num_layers": num_layers,
                            "num_heads": num_heads,
                            "mlp_dim": mlp_dim,
                            "max_seq_len": block_size,
                            "dropout": dropout,
                            "vocab_size": vocab_size,
                            "dataset": args.dataset,
                            "tokenizer": "bpe",
                            "bpe_vocab_size": args.bpe_vocab_size,
                            "seed": args.seed,
                            "model_type": "vanilla",
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
                "num_layers": num_layers,
                "num_heads": num_heads,
                "mlp_dim": mlp_dim,
                "max_seq_len": block_size,
                "dropout": dropout,
                "vocab_size": vocab_size,
                "dataset": args.dataset,
                "tokenizer": "bpe",
                "bpe_vocab_size": args.bpe_vocab_size,
                "seed": args.seed,
                "model_type": "vanilla",
                "gradient_accumulation_steps": accum_steps,
            },
            "vocab": tokenizer.save_state(),
        },
        last_checkpoint_path,
    )

    writer.close()
    print(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
