# Newton Gravity Transformer (NGT)

<a id="top"></a>

**[English](README.md)** | **[Korean](README_KO.md)**

### *"Words are Particles, Attention is Gravity"*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

NGT explores a physics-inspired alternative to dot-product attention: tokens are particles with **mass** and **coordinates**, and attention is a learnable **gravity kernel** over distances in a latent space.

## WikiText-103 Status (w3_25m)

This repo currently focuses on WikiText-103 experiments (BPE-8192, ~25M params). Latest screening summary:
- `reports/w3_25m_summary.md`

**Screening snapshot (seed=42, max_steps=15000):**
- Best overall (final @15000): Vanilla val loss `4.5554` (ppl `95.14`)
- Best NGT (final @15000): `--mass-in-value` val loss `4.6635` (ppl `106.01`)
- Throughput on the same settings: Vanilla ~`4.97` steps/s vs NGT ~`0.83–0.86` steps/s (~`6x` slower)

For how to run/monitor/download on RunPod, see:
- `GUIDE.md` (commands)
- `FUTURE.md` (handoff notes / continue training)

## Highlights

- Gravity attention with learnable per-head strength (gamma) and bias (beta)
- Mass embedding (`Softplus`) and coordinate embedding (`z`)
- Coordinate evolution across layers + learnable radius cutoff (hard or soft)
- Repulsion regularizer (mass-based, distance-clamped for stability)
- Tokenizers: `char`, `bpe` (HF `tokenizers`), `tiktoken`
- TensorBoard scalars + Projector embeddings
- Checkpoint safety: `*_best.pt` and `*_last.pt` + robust `--resume`
- Inference compatibility for legacy checkpoints in `chat.py`

---

## About

Hi! I'm **Taegyu Lee**, an undergraduate student exploring physics-inspired attention mechanisms and geometric interpretability in language models.

This project started from a simple question: *"What if semantic relationships followed something like motion + gravity?"* NGT is a personal research-style implementation to test that idea end-to-end (training, logging, visualization, and checkpoint management).

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.11+ is recommended. Training is much faster on CUDA GPUs; CPU training works but is slow.

---

## Quickstart (TinyShakespeare)

```bash
# Download TinyShakespeare
python prepare_data.py

# Train NGT (default config)
python train_shakespeare.py --max-steps 5000 --checkpoint-path checkpoints/shakespeare.pt

# Train Vanilla baseline
python train_shakespeare_vanilla.py --max-steps 5000 --checkpoint-path checkpoints/vanilla_shakespeare.pt

# Chat (NGT / Vanilla auto-detected from checkpoint config)
python chat.py --checkpoint-path checkpoints/shakespeare.pt_best.pt
python chat.py --checkpoint-path checkpoints/vanilla_shakespeare.pt_best.pt
```

---

## Checkpoints & Resume

When you pass `--checkpoint-path checkpoints/foo.pt`, training writes:

- Best validation model: `checkpoints/foo.pt_best.pt`
- Final model state: `checkpoints/foo.pt_last.pt`

`--resume` attempts to load in this order: `*_last.pt` → `*_best.pt` → the base path.

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

## RunPod Workflow (WikiText-103 ~25M)

- Run guide: `GUIDE.md`
- Session handoff / continuing notes: `FUTURE.md`
- Runner script: `run_wikitext103_25m.sh` (use `budget10` for screening + report generation)

If you cannot use `scp` or extra HTTP ports, prefer `runpodctl send/receive` to download results:

```bash
# on the pod
tar -cJf /tmp/w3_25m_results.tar.xz results/w3_25m
runpodctl send /tmp/w3_25m_results.tar.xz
```

```powershell
# on local Windows PowerShell
.\runpodctl.exe receive <CODE>
```

---

## TensorBoard (Scalars + Projector)

Training logs to `runs/...` by default.

```bash
tensorboard --logdir runs
```

To log embeddings for the Projector tab, set `--vis-interval` (defaults to `--eval-interval` when omitted).

---

## Coordinate Visualization (3D PCA)

```bash
python visualize_coords.py --checkpoint-path checkpoints/shakespeare.pt_best.pt --output coords.html
```

This produces an interactive Plotly HTML scatter where marker size/color encodes mass.

---

## Tests

```bash
pytest -q
```

Known issue: the default hard radius cutoff uses a boolean mask, so `radius_param` does not receive gradients; this currently makes `test_gradient_flow` fail in `test_ngt.py`.

---

## Security note

Checkpoints are loaded via `torch.load(..., weights_only=False)`, which uses Python pickle. Do not load untrusted `.pt` files.

---

## License

MIT (see `LICENSE`).

---

<div align="center">

**[Back to Top](#top)**

</div>
