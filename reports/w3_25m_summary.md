# w3_25m Experiment Summary (WikiText-103, ~25M)

This file summarizes the **w3_25m screening run set** (seed=42, max_steps=15000).

## Source artifacts (not tracked in git)

From the RunPod workspace (or your downloaded archive), expect:
- `results/w3_25m/report.md`
- `results/w3_25m/results.csv`
- `results/w3_25m/coords_*.html` (self-contained Plotly visualizations)
- `logs/w3_25m/w3_25m_screen_*.log` (training logs with `val_loss` + `elapsed=...`)

## What ran

- Dataset: `wikitext103`
- Tokenizer: BPE, vocab size `8192`
- Context length (`block_size`): `512`
- Model config: `hidden_dim=512`, `coord_dim=64`, `num_layers=8`, `num_heads=8`, `mlp_dim=2048`, `dropout=0.1`
- Optim: AdamW, cosine schedule + warmup (see `GUIDE.md` for the exact command used)
- Batch: `batch_size=16`, `gradient_accumulation_steps=2`
- Seed: `42`
- Max steps: `15000`

## Headline results (@15000)

Validation loss is cross-entropy; perplexity is `exp(loss)`.

- **Best overall (vanilla)**: `val_loss=4.5554` (ppl `95.14`)
- **Best NGT variant**: `--mass-in-value` -> `val_loss=4.6635` (ppl `106.01`)
  - Delta vs vanilla (@15000): `+0.1081` loss (~`+11.4%` ppl)

## Fixed-step comparison (@15000)

All runs reached `iter=15000` (via `*_last.pt`). Values below are from the per-run log line:
`step 15000/15000 ... val_loss=...`

| run | val loss @15000 |
|---|---:|
| screen_vanilla_25m | 4.5554 |
| screen_ngt_mass_in_value | 4.6635 |
| screen_ngt_no_repulsion | 4.7214 |
| screen_ngt_repulsion_interval_8 | 4.7889 |
| screen_ngt_default | 4.7915 |
| screen_ngt_no_radius | 4.7940 |

## Best vs last (per run)

`best` = lowest validation loss observed during training (`*_best.pt`).  
`last` = validation loss at the final step (15000) (`*_last.pt`).

| run | best val loss | best step | last val loss (@15000) |
|---|---:|---:|---:|
| screen_vanilla_25m | 4.5524 | 13500 | 4.5554 |
| screen_ngt_mass_in_value | 4.6451 | 13000 | 4.6635 |
| screen_ngt_no_repulsion | 4.7214 | 15000 | 4.7214 |
| screen_ngt_repulsion_interval_8 | 4.7748 | 13000 | 4.7889 |
| screen_ngt_default | 4.7762 | 13000 | 4.7915 |
| screen_ngt_no_radius | 4.7772 | 13000 | 4.7940 |

## Full ranking (best -> worst)

Best validation observed during training (may occur before 15000):

| rank | model | flags | best val loss | ppl (exp(loss)) | step | checkpoint |
|---:|---|---|---:|---:|---:|---|
| 1 | vanilla | default | 4.5524 | 94.86 | 13500 | `checkpoints/w3_25m/vanilla_25m.pt_best.pt` |
| 2 | ngt | rsqrt,mass_val | 4.6451 | 104.07 | 13000 | `checkpoints/w3_25m/ngt_mass_in_value.pt_best.pt` |
| 3 | ngt | rsqrt *(no_repulsion run)* | 4.7214 | 112.33 | 15000 | `checkpoints/w3_25m/ngt_no_repulsion.pt_best.pt` |
| 4 | ngt | rsqrt *(repulsion_interval_8 run)* | 4.7748 | 118.48 | 13000 | `checkpoints/w3_25m/ngt_repulsion_interval_8.pt_best.pt` |
| 5 | ngt | rsqrt *(default run)* | 4.7762 | 118.66 | 13000 | `checkpoints/w3_25m/ngt_default.pt_best.pt` |
| 6 | ngt | no_radius,rsqrt | 4.7772 | 118.77 | 13000 | `checkpoints/w3_25m/ngt_no_radius.pt_best.pt` |

## Throughput (screening)

Approx throughput normalized to the same training horizon (**step=500 -> 15000**) using `elapsed` from `logs/w3_25m/w3_25m_screen_*.log`:

| run | steps/sec | sec/step |
|---|---:|---:|
| w3_25m_screen_vanilla_25m.log | 4.964 | 0.20 |
| w3_25m_screen_ngt_default.log | 0.829 | 1.21 |
| w3_25m_screen_ngt_no_repulsion.log | 0.829 | 1.21 |
| w3_25m_screen_ngt_repulsion_interval_8.log | 0.830 | 1.21 |
| w3_25m_screen_ngt_mass_in_value.log | 0.852 | 1.17 |
| w3_25m_screen_ngt_no_radius.log | 0.855 | 1.17 |

Rule of thumb: NGT runs here are ~0.83-0.86 steps/sec vs vanilla ~4.96 steps/sec (~5.8-6.0x slower) under the same batch/accum/block settings.

## HTML visualizations (coords)

Each `coords_*.html` is a self-contained Plotly HTML (open locally in a browser).

Note: Hover labels may include tokenizer-specific whitespace markers (e.g., `Ġ` / `Ċ`, U+0120 / U+010A) indicating a leading space/newline. These are tokenizer artifacts, not literal characters from the raw text.

