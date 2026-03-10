# Newton Gravity Transformer (NGT)

<a id="top"></a>

**[English](README.md)** | **[Korean](README_KO.md)**

### *"Words are Particles, Attention is Gravity"*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

NGT is an experimental Transformer variant where tokens behave like particles: each token has a learned **mass** and **coordinates**, and attention is a learnable **gravity kernel** over distances in a latent space.

This repo currently focuses on end-to-end training, logging (TensorBoard), and checkpointing (`*_best.pt` / `*_last.pt`).

---

## Project focus: current ~6M WikiText-2 path

The active code path now targets a small `~6M`-scale setup on WikiText-2 with a fixed BPE tokenizer. The goal of the current branch is fast iteration, stability debugging, and clean baseline comparison against a vanilla transformer under the same data path.

Shakespeare dataset/checkpoints are legacy and no longer used in this project.

### Project trajectory (TinyShakespeare -> current WikiText-2)

- Initial phase used TinyShakespeare as a fast prototyping sandbox.
- The current branch is intentionally refocused on a much smaller `~6M` WikiText-2 setup for faster debugging and tighter ablation loops.
- The main near-term goal is to understand whether the current NGT formulation can stay competitive against a vanilla transformer baseline at this smaller scale.

### Current 6M snapshot (WikiText-2, ~20 epochs)

Current small-scale comparison on the active branch:

- Dataset: `wikitext2`
- Tokenizer: `BPE-8192`
- Context length: `256`
- Batch / accumulation: `16 x 4`
- Schedule: cosine with `warmup_steps=100`
- Learning rate: `5e-5`
- Training horizon: `3340` steps (roughly `~20` epochs)

| model | config | final val loss | final train loss | elapsed |
|---|---|---:|---:|---:|
| vanilla | current ~6M baseline | 6.2320 | 6.2048 | 645.3s |
| new-NGT | current ~6M branch | 7.9497 | 7.9810 | 3733.5s |

At the moment, the current `~6M` NGT branch is roughly `5.8x` slower than the matched vanilla baseline and underperforms it by `+1.7177` validation-loss points.

---

## What is NGT (mechanism overview)

Standard Transformers compute attention via dot products between query/key vectors.

NGT introduces a geometric stream:

- Each token has a hidden state `h` (semantic stream) and coordinate `z` (geometric stream)
- Each token has a learned mass `m` (kept positive via `Softplus`)
- Attention scores depend on distance in `z` space (and mass interaction), not dot products
- Optional radius cutoff provides learned sparsity
- A mass-based repulsion regularizer discourages coordinate collapse

---

## Installation, quickstart, and checkpoints

Install:

```bash
pip install -r requirements.txt
```

Quickstart (WikiText-2, current ~6M benchmark path):

```bash
# Download/cache WikiText-2 via HuggingFace datasets
python prepare_data.py

# Run NGT training (defaults: WikiText-2 + BPE-8192, ~6M path)
python train.py --data-path data \
  --checkpoint-path checkpoints/ngt_wikitext2_bpe_8192.pt

```

The current training path is fixed to WikiText-2 and tuned around the current ~6M model scale.

Checkpoint policy:

- If you pass `--checkpoint-path checkpoints/foo.pt`, training writes:
- Best validation model: `checkpoints/foo.pt_best.pt`
- Final model state: `checkpoints/foo.pt_last.pt`

Python 3.11+ is recommended. CUDA is strongly recommended for training.

---

## Training (current branch)

See `python train.py --help` for the full NGT training options and `python train_vanilla.py --help` for the matched vanilla baseline path.

Common flags:

- Dataset: `--dataset wikitext2`, `--data-path ...`
- Tokenizer: fixed BPE path with `--bpe-vocab-size` and `--tokenizer-path`
- Regularization: `--repulsion`, `--lambda-repulsion`, `--repulsion-interval` (`4` by default when enabled)
- Performance: gravity scoring uses the rsqrt-based path, plus `--use-amp`, `--gradient-accumulation-steps`
- Schedule: `--use-cosine-schedule --warmup-steps N`

Current baseline comparison work uses:

- NGT: `python train.py ...`
- Vanilla baseline: `python train_vanilla.py ...`

Example:

```bash
python train.py --data-path data \
  --checkpoint-path checkpoints/ngt_wikitext2_bpe_8192.pt
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
