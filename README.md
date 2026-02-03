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

- Minimal reproduction script (vanilla + NGT mass-in-value @15k): `run_wikitext103_25m.sh`
- Latest tracked experiment summary: `reports/w3_25m_summary.md`

### Latest screening snapshot (w3_25m, seed=42, max_steps=15000)

Validation loss is cross-entropy; perplexity is `exp(loss)`.

| model | config | val loss @15000 | ppl |
|---|---|---:|---:|
| vanilla | baseline | 4.5554 | 95.14 |
| NGT | `--mass-in-value` | 4.6635 | 106.01 |

Throughput on the same settings (B=16, accum=2, block=512):
- vanilla: ~4.97 steps/s
- NGT: ~0.83-0.86 steps/s (~6x slower)

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

## Installation

```bash
pip install -r requirements.txt
```

Python 3.11+ is recommended. CUDA is strongly recommended for training.

---

## Quickstart (WikiText-103, 15k)

```bash
# Download/caches WikiText-103 via HuggingFace datasets
python prepare_data.py --dataset wikitext103

# Run vanilla + NGT (mass-in-value) at 15k steps
bash run_wikitext103_25m.sh

# Chat (NGT / Vanilla auto-detected from checkpoint config)
python chat.py --checkpoint-path checkpoints/w3_25m/ngt_mass_in_value.pt_best.pt
python chat.py --checkpoint-path checkpoints/w3_25m/vanilla_25m.pt_best.pt
```

---

## Checkpoints and resume

When you pass `--checkpoint-path checkpoints/foo.pt`, training writes:

- Best validation model: `checkpoints/foo.pt_best.pt`
- Final model state: `checkpoints/foo.pt_last.pt`

`--resume` attempts to load in this order: `*_last.pt` -> `*_best.pt` -> the base path.

---

## Training (NGT)

See `python train_shakespeare.py --help` for the full list. Common flags:

- Dataset: `--dataset {shakespeare,wikitext103}`, `--data-path ...`
- Tokenizers: `--tokenizer {char,bpe,tiktoken}`
  - BPE: `--bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json`
- Regularization: `--lambda-repulsion`, `--repulsion-interval`, `--no-repulsion`
- Sparsity: `--no-radius-cutoff` or `--use-soft-cutoff`
- Performance: `--use-rsqrt`, `--use-amp`, `--gradient-accumulation-steps`
- Schedule: `--use-cosine-schedule --warmup-steps N`

Example (WikiText-103):

```bash
python train_shakespeare.py --dataset wikitext103 --data-path data \
  --tokenizer bpe --bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json \
  --hidden-dim 512 --coord-dim 64 --num-layers 8 --num-heads 8 --mlp-dim 2048 \
  --block-size 512 --batch-size 16 --gradient-accumulation-steps 2 \
  --use-amp --use-cosine-schedule --warmup-steps 2000 \
  --checkpoint-path checkpoints/w3_ngt.pt
```

---

## TensorBoard and coordinate visualization

TensorBoard:

```bash
tensorboard --logdir runs
```

Coordinate visualization (3D PCA to a Plotly HTML):

```bash
python visualize_coords.py --checkpoint-path checkpoints/shakespeare.pt_best.pt --output coords.html
```

---

## Security note

Checkpoints are loaded via `torch.load(..., weights_only=False)`, which uses Python pickle. Do not load untrusted `.pt` files.

---

## About

Hi! I'm **Taegyu Lee**, an undergraduate student exploring physics-inspired attention mechanisms and geometric interpretability in language models.

---

## License

MIT (see `LICENSE`).

---

<div align="center">

**[Back to Top](#top)**

</div>
