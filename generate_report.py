#!/usr/bin/env python3
"""Generate an ablation study report from NGT experiment checkpoints.

Usage:
    python generate_report.py checkpoints/*_best.pt
    python generate_report.py checkpoints/*_best.pt --output results/report.md
"""

import argparse
import glob
import os
import sys

import torch


def load_checkpoint_info(path):
    """Extract config and metrics from a checkpoint file."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Warning: could not load {path}: {e}", file=sys.stderr)
        return None

    config = ckpt.get("config", {})
    best_val = ckpt.get("best_val", None)
    step = ckpt.get("iter", None)

    # Determine model type
    model_type = config.get("model_type", None)
    if model_type is None:
        # Infer from config keys
        if "coord_dim" in config:
            model_type = "ngt"
        else:
            model_type = "vanilla"

    # Build ablation flags description
    flags = []
    if model_type == "ngt":
        if not config.get("use_radius_cutoff", True):
            flags.append("no-radius-cutoff")
        if config.get("use_rsqrt", False):
            flags.append("rsqrt")
        if config.get("mass_in_value", False):
            flags.append("mass-in-value")
        if config.get("use_soft_cutoff", False):
            flags.append("soft-cutoff")

    # Check if repulsion was disabled (inferred from checkpoint path or run name)
    # Since no_repulsion isn't stored in config, check the path
    basename = os.path.basename(path).lower()
    if "no_repulsion" in basename:
        flags.append("no-repulsion")

    flags_str = ", ".join(flags) if flags else "(default)"

    return {
        "path": path,
        "model_type": model_type,
        "best_val_loss": best_val,
        "step": step,
        "config": config,
        "flags": flags,
        "flags_str": flags_str,
        "dataset": config.get("dataset", "unknown"),
        "seed": config.get("seed", None),
    }


def find_run(runs, model_type, flags=None, flags_str=None):
    """Find a run matching model_type and flags."""
    for r in runs:
        if r["model_type"] != model_type:
            continue
        if flags_str is not None and r["flags_str"] == flags_str:
            return r
        if flags is not None and sorted(r["flags"]) == sorted(flags):
            return r
    return None


def format_delta(base_val, compare_val):
    """Format the delta between two val losses."""
    if base_val is None or compare_val is None:
        return "N/A"
    delta = compare_val - base_val
    pct = (delta / base_val) * 100 if base_val != 0 else 0
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f} ({sign}{pct:.1f}%)"


def generate_report(runs, output_path=None):
    """Generate markdown report from collected run info."""
    lines = []
    lines.append("# NGT Ablation Study Report\n")

    if not runs:
        lines.append("No checkpoint data found.\n")
        report = "\n".join(lines)
        _write_report(report, output_path)
        return report

    # Group by dataset
    datasets = sorted(set(r["dataset"] for r in runs))

    for dataset in datasets:
        ds_runs = [r for r in runs if r["dataset"] == dataset]
        lines.append(f"## Dataset: {dataset}\n")

        # Summary table sorted by val_loss
        ds_runs_sorted = sorted(ds_runs, key=lambda r: r["best_val_loss"] if r["best_val_loss"] is not None else float("inf"))

        lines.append("### Results Summary\n")
        lines.append("| # | Model | Flags | Best Val Loss | Step |")
        lines.append("|---|-------|-------|--------------|------|")
        for i, r in enumerate(ds_runs_sorted, 1):
            val_str = f"{r['best_val_loss']:.4f}" if r["best_val_loss"] is not None else "N/A"
            step_str = str(r["step"]) if r["step"] is not None else "N/A"
            lines.append(f"| {i} | {r['model_type']} | {r['flags_str']} | {val_str} | {step_str} |")
        lines.append("")

        # Find key runs for comparison
        vanilla = find_run(ds_runs, "vanilla", flags_str="(default)")
        ngt_default = find_run(ds_runs, "ngt", flags_str="(default)")
        ngt_bare = find_run(ds_runs, "ngt", flags=["no-repulsion", "no-radius-cutoff"])
        ngt_no_repulsion = find_run(ds_runs, "ngt", flags=["no-repulsion"])
        ngt_no_radius = find_run(ds_runs, "ngt", flags=["no-radius-cutoff"])
        ngt_mass_val = find_run(ds_runs, "ngt", flags=["mass-in-value"])
        ngt_soft = find_run(ds_runs, "ngt", flags=["soft-cutoff"])

        # Interpretation framework
        lines.append("### Ablation Analysis\n")

        comparisons = []

        if vanilla and ngt_default:
            comparisons.append((
                "NGT vs Vanilla",
                "Does NGT outperform Vanilla?",
                vanilla["best_val_loss"],
                ngt_default["best_val_loss"],
                "lower is better for NGT",
            ))

        if ngt_default and ngt_bare:
            comparisons.append((
                "NGT default vs bare gravity",
                "Total contribution of repulsion + radius cutoff",
                ngt_bare["best_val_loss"],
                ngt_default["best_val_loss"],
                "improvement from adding both components",
            ))

        if ngt_default and ngt_no_repulsion:
            comparisons.append((
                "NGT default vs no-repulsion",
                "Repulsion contribution (removed from default)",
                ngt_default["best_val_loss"],
                ngt_no_repulsion["best_val_loss"],
                "positive delta = repulsion helps",
            ))

        if ngt_default and ngt_no_radius:
            comparisons.append((
                "NGT default vs no-radius-cutoff",
                "Radius cutoff contribution (removed from default)",
                ngt_default["best_val_loss"],
                ngt_no_radius["best_val_loss"],
                "positive delta = radius cutoff helps",
            ))

        if ngt_bare and ngt_no_repulsion:
            comparisons.append((
                "bare gravity vs bare+radius",
                "Incremental effect of adding radius to bare gravity",
                ngt_bare["best_val_loss"],
                ngt_no_repulsion["best_val_loss"],
                "negative delta = radius helps",
            ))

        if ngt_bare and ngt_no_radius:
            comparisons.append((
                "bare gravity vs bare+repulsion",
                "Incremental effect of adding repulsion to bare gravity",
                ngt_bare["best_val_loss"],
                ngt_no_radius["best_val_loss"],
                "negative delta = repulsion helps",
            ))

        if ngt_default and ngt_mass_val:
            comparisons.append((
                "NGT default vs mass-in-value",
                "Mass placement strategy comparison",
                ngt_default["best_val_loss"],
                ngt_mass_val["best_val_loss"],
                "positive delta = default (mass in attention) is better",
            ))

        if ngt_default and ngt_soft:
            comparisons.append((
                "NGT default vs soft-cutoff",
                "Hard vs soft radius masking",
                ngt_default["best_val_loss"],
                ngt_soft["best_val_loss"],
                "positive delta = hard cutoff is better",
            ))

        if comparisons:
            lines.append("| Comparison | Question | Base | Compare | Delta | Note |")
            lines.append("|-----------|----------|------|---------|-------|------|")
            for name, question, base_val, cmp_val, note in comparisons:
                base_str = f"{base_val:.4f}" if base_val is not None else "N/A"
                cmp_str = f"{cmp_val:.4f}" if cmp_val is not None else "N/A"
                delta_str = format_delta(base_val, cmp_val)
                lines.append(f"| {name} | {question} | {base_str} | {cmp_str} | {delta_str} | {note} |")
            lines.append("")
        else:
            lines.append("Not enough runs to perform ablation comparisons.\n")

        # Config details
        if ds_runs_sorted:
            sample = ds_runs_sorted[0]["config"]
            lines.append("### Training Config\n")
            lines.append(f"- hidden_dim: {sample.get('hidden_dim', 'N/A')}")
            lines.append(f"- num_layers: {sample.get('num_layers', 'N/A')}")
            lines.append(f"- num_heads: {sample.get('num_heads', 'N/A')}")
            lines.append(f"- mlp_dim: {sample.get('mlp_dim', 'N/A')}")
            lines.append(f"- block_size: {sample.get('max_seq_len', 'N/A')}")
            lines.append(f"- vocab_size: {sample.get('vocab_size', 'N/A')}")
            lines.append(f"- seed: {sample.get('seed', 'N/A')}")
            if "coord_dim" in sample:
                lines.append(f"- coord_dim: {sample.get('coord_dim', 'N/A')}")
            lines.append("")

    report = "\n".join(lines)
    _write_report(report, output_path)
    return report


def _write_report(report, output_path):
    """Write report to file and stdout."""
    print(report)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ablation study report from checkpoints")
    parser.add_argument("checkpoints", nargs="*", default=[],
                        help="Checkpoint files (supports glob patterns)")
    parser.add_argument("--output", "-o", type=str, default="results/report.md",
                        help="Output markdown file path (default: results/report.md)")
    args = parser.parse_args()

    # Collect checkpoint paths
    paths = []
    if args.checkpoints:
        for pattern in args.checkpoints:
            expanded = glob.glob(pattern)
            if expanded:
                paths.extend(expanded)
            elif os.path.exists(pattern):
                paths.append(pattern)
    else:
        # Auto-discover best checkpoints
        paths = glob.glob("checkpoints/*_best.pt")

    if not paths:
        print("No checkpoint files found. Pass paths or ensure checkpoints/*_best.pt exist.",
              file=sys.stderr)
        sys.exit(1)

    paths = sorted(set(paths))
    print(f"Loading {len(paths)} checkpoint(s)...\n")

    runs = []
    for p in paths:
        info = load_checkpoint_info(p)
        if info is not None:
            runs.append(info)

    generate_report(runs, args.output)


if __name__ == "__main__":
    main()
