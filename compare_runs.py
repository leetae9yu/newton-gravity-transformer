"""
Compare training runs by reading checkpoint files.

Usage:
    python compare_runs.py checkpoints/*.pt
    python compare_runs.py checkpoints/*.pt --csv results.csv
"""

import argparse
import csv
import sys

import torch


def parse_checkpoint(path):
    """Extract metadata from a checkpoint file."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Warning: could not load {path}: {e}", file=sys.stderr)
        return None

    config = ckpt.get("config", {})

    # Determine model type from checkpoint contents
    model_type = "unknown"
    if "coord_dim" in config or any("coord" in k for k in (ckpt.get("model_state") or {}).keys()):
        model_type = "ngt"
    else:
        model_type = "vanilla"

    # Collect ablation flags for NGT
    ablation_flags = []
    if model_type == "ngt":
        if not config.get("use_radius_cutoff", True):
            ablation_flags.append("no_radius")
        if config.get("use_rsqrt", False):
            ablation_flags.append("rsqrt")
        if config.get("mass_in_value", False):
            ablation_flags.append("mass_val")
        if config.get("use_soft_cutoff", False):
            ablation_flags.append("soft_cutoff")

    return {
        "path": path,
        "model_type": model_type,
        "dataset": config.get("dataset", "unknown"),
        "hidden_dim": config.get("hidden_dim", "?"),
        "num_layers": config.get("num_layers", "?"),
        "num_heads": config.get("num_heads", "?"),
        "ablation_flags": ",".join(ablation_flags) if ablation_flags else "default",
        "best_val_loss": ckpt.get("best_val", None),
        "total_steps": ckpt.get("iter", None),
    }


COLUMNS = [
    "model_type",
    "dataset",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "ablation_flags",
    "best_val_loss",
    "total_steps",
    "path",
]


def print_table(rows):
    """Print a formatted table to stdout."""
    if not rows:
        print("No valid checkpoints found.")
        return

    # Sort by best_val_loss (ascending), None values last
    rows.sort(key=lambda r: (r["best_val_loss"] is None, r["best_val_loss"] or float("inf")))

    # Compute column widths
    widths = {col: len(col) for col in COLUMNS}
    for row in rows:
        for col in COLUMNS:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            widths[col] = max(widths[col], len(str(val)))

    # Header
    header = " | ".join(col.ljust(widths[col]) for col in COLUMNS)
    separator = "-+-".join("-" * widths[col] for col in COLUMNS)
    print(header)
    print(separator)

    # Rows
    for row in rows:
        cells = []
        for col in COLUMNS:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            cells.append(str(val).ljust(widths[col]))
        print(" | ".join(cells))


def write_csv(rows, path):
    """Write rows to a CSV file."""
    rows.sort(key=lambda r: (r["best_val_loss"] is None, r["best_val_loss"] or float("inf")))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in COLUMNS})
    print(f"CSV saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare training runs by reading checkpoint files."
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="Checkpoint .pt files to compare",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Save results to CSV file",
    )
    args = parser.parse_args()

    rows = []
    for path in args.checkpoints:
        result = parse_checkpoint(path)
        if result is not None:
            rows.append(result)

    print_table(rows)

    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
