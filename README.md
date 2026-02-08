# Newton Gravity Transformer (NGT)

<a id="top"></a>

**[English](README.md)** | **[Korean](README_KO.md)**

### *"Words are Particles, Attention is Gravity"*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

NGT is an experimental Transformer variant where tokens behave like particles: each token has a learned **mass** and **coordinates**, and attention is a learnable **gravity kernel** over distances in a latent space.

This repo includes end-to-end training, logging (TensorBoard), checkpointing (`*_best.pt` / `*_last.pt`), and interactive coordinate visualizations.

---

## Project focus: WikiText-103 (~25M)

Current focus is WikiText-103 with BPE-8192 and ~25M parameter scale.

- Minimal reproduction script (15k screening): `run_wikitext103_25m.sh`
- Minimal summary: `reports/w3_25m_summary.md`
- Full screening artifacts: `w3_25m_results/results/w3_25m/Summary.md`
- Pretrained checkpoints (w3_25m): `https://huggingface.co/leetae9yu/newton-gravity-transformer/tree/main/checkpoints/w3_25m`

Shakespeare dataset/checkpoints are legacy and no longer actively used in this project.

### Project trajectory (TinyShakespeare -> WikiText-103)

- Initial phase used TinyShakespeare (char-level) as a fast prototyping sandbox.
- In archived 5k-step TinyShakespeare checkpoints, best validation losses reached about `1.70` and later about `1.55`.
- After that, the project moved to larger-scale screening on WikiText-103 (~25M parameter scale).
- Going forward, the plan is to keep scaling model capacity and training budget step by step.

### Latest screening snapshot (w3_25m, seed=42, max_steps=15000)

Validation loss is cross-entropy; perplexity is `exp(loss)`.

| run | config | val loss @15000 | ppl @15000 | best val loss (step) |
|---|---|---:|---:|---:|
| vanilla | baseline | 4.5554 | 95.14 | 4.5524 (13500) |
| ngt_mass_in_value | `--mass-in-value --use-rsqrt` | 4.6635 | 106.01 | 4.6451 (13000) |
| ngt_no_repulsion | `--no-repulsion --use-rsqrt` | 4.7214 | 112.33 | 4.7214 (15000) |
| ngt_repulsion_interval_8 | `--repulsion-interval 8 --use-rsqrt` | 4.7889 | 120.17 | 4.7748 (13000) |
| ngt_default | `--use-rsqrt` | 4.7915 | 120.48 | 4.7762 (13000) |
| ngt_no_radius | `--no-radius-cutoff --use-rsqrt` | 4.7940 | 120.78 | 4.7772 (13000) |

Throughput on the same settings (`batch=16`, `accum=2`, `block=512`):

- vanilla: ~4.964 steps/s
- ngt_mass_in_value: ~0.852 steps/s
- ngt_no_radius: ~0.855 steps/s
- ngt_default / ngt_no_repulsion / ngt_repulsion_interval_8: ~0.829-0.830 steps/s

This is a budget-constrained screening run (15k steps, roughly about 2 epochs depending on tokenized train-set size), so treat results as directional.

---

## What is NGT (mechanism overview)

Standard Transformers compute attention via dot products between query/key vectors.

NGT introduces a geometric stream:

- Each token has a hidden state `h` (semantic stream) and coordinate `z` (geometric stream)
- Each token has a learned mass `m` (kept positive via `Softplus`)
- Attention scores depend on distance in `z` space (and mass interaction), not dot products
- Optional radius cutoff provides learned sparsity (hard or soft)
- A mass-based repulsion regularizer discourages coordinate collapse

---

## Installation, quickstart, and checkpoints

Install:

```bash
pip install -r requirements.txt
```

Quickstart (WikiText-103, 15k screening):

```bash
# Download/cache WikiText-103 via HuggingFace datasets
python prepare_data.py --dataset wikitext103

# Run vanilla + NGT (mass-in-value) at 15k steps
bash run_wikitext103_25m.sh

# Chat (NGT / Vanilla auto-detected from checkpoint config)
python chat.py --checkpoint-path checkpoints/w3_25m/ngt_mass_in_value.pt_best.pt
python chat.py --checkpoint-path checkpoints/w3_25m/vanilla_25m.pt_best.pt
```

Checkpoint policy:

- If you pass `--checkpoint-path checkpoints/foo.pt`, training writes:
- Best validation model: `checkpoints/foo.pt_best.pt`
- Final model state: `checkpoints/foo.pt_last.pt`
- `--resume` loads in this order: `*_last.pt` -> `*_best.pt` -> base path

Python 3.11+ is recommended. CUDA is strongly recommended for training.

---

## Training (NGT)

See `python train_shakespeare.py --help` for the full list.

Common flags:

- Dataset: `--dataset {shakespeare,wikitext103}`, `--data-path ...`
- Tokenizers: `--tokenizer {char,bpe,tiktoken}`
- BPE option: `--bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json`
- Regularization: `--lambda-repulsion`, `--repulsion-interval`, `--no-repulsion`
- Sparsity: `--no-radius-cutoff` or `--use-soft-cutoff`
- Performance: `--use-rsqrt`, `--use-amp`, `--gradient-accumulation-steps`
- Schedule: `--use-cosine-schedule --warmup-steps N`

Example:

```bash
python train_shakespeare.py --dataset wikitext103 --data-path data \
  --tokenizer bpe --bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json \
  --hidden-dim 512 --coord-dim 64 --num-layers 8 --num-heads 8 --mlp-dim 2048 \
  --block-size 512 --batch-size 16 --gradient-accumulation-steps 2 \
  --use-amp --use-cosine-schedule --warmup-steps 2000 \
  --checkpoint-path checkpoints/w3_ngt.pt
```

---

## Artifacts and visualization links

Summary/report artifacts:

- [Minimal summary (`reports/w3_25m_summary.md`)](reports/w3_25m_summary.md)
- [Full summary (`w3_25m_results/results/w3_25m/Summary.md`)](w3_25m_results/results/w3_25m/Summary.md)
- [Ablation report (`w3_25m_results/results/w3_25m/report.md`)](w3_25m_results/results/w3_25m/report.md)
- [Results CSV (`w3_25m_results/results/w3_25m/results.csv`)](w3_25m_results/results/w3_25m/results.csv)

Interactive HTML visualizations (Plotly 3D PCA):

- [coords_ngt_default.html](w3_25m_results_latest/results/w3_25m/coords_ngt_default.html)
- [coords_ngt_mass_in_value.html](w3_25m_results_latest/results/w3_25m/coords_ngt_mass_in_value.html)
- [coords_ngt_no_radius.html](w3_25m_results_latest/results/w3_25m/coords_ngt_no_radius.html)
- [coords_ngt_no_repulsion.html](w3_25m_results_latest/results/w3_25m/coords_ngt_no_repulsion.html)
- [coords_ngt_repulsion_interval_8.html](w3_25m_results_latest/results/w3_25m/coords_ngt_repulsion_interval_8.html)

TensorBoard:

```bash
tensorboard --logdir runs
```

---

## Security note

Checkpoints are loaded via `torch.load(..., weights_only=False)`, which uses Python pickle. Do not load untrusted `.pt` files.

---

## About

Hi! I'm **Taegyu Lee**, an undergraduate student in Korea with strong interest in AI.

I started this project to build practical personal research experience while preparing for graduate school. Since this is still undergraduate-level work, there may be many things to improve. PRs and issues are always welcome.

Contact: `mjrror@korea.ac.kr`

---

## License

MIT (see `LICENSE`).

---

<div align="center">

**[Back to Top](#top)**

</div>
