# w3_25m Experiment Summary (WikiText-103, ~25M)

Source artifacts:
- Report: [report.md](./report.md)
- Raw results: [results.csv](./results.csv)
- 3D PCA coords (Plotly HTML):
  - [coords_ngt_default.html](./coords_ngt_default.html)
  - [coords_ngt_mass_in_value.html](./coords_ngt_mass_in_value.html)
  - [coords_ngt_no_radius.html](./coords_ngt_no_radius.html)
  - [coords_ngt_no_repulsion.html](./coords_ngt_no_repulsion.html)
  - [coords_ngt_repulsion_interval_8.html](./coords_ngt_repulsion_interval_8.html)

## What ran

- Dataset: `wikitext103`
- Vocab size: `8192`
- Context length (`block_size`): `512`
- Model config: `hidden_dim=512`, `num_layers=8`, `num_heads=8`, `mlp_dim=1536`
- Seed: `42`

## Headline results

This is a small screening set (see note in [report.md](./report.md)); use as directional, not definitive.

- **Final @15000 (overall best)**: `vanilla` - **val loss `4.5554`** (ppl `95.14`)
- **Final @15000 (best NGT)**: `ngt_mass_in_value` - **val loss `4.6635`** (ppl `106.01`)
  - Delta vs vanilla (@15000): `+0.1081` loss (~ `+11.4%` ppl)

## Fixed-step comparison (@15000)

All runs reached `iter=15000` (`*_last.pt`). The table below compares **validation loss at step=15000** from the per-run logs (`logs/w3_25m/w3_25m_screen_*.log`).

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
`last` = validation loss at the final step (15000) (`*_last.pt`, shown via the `step 15000/15000 ... val_loss=...` log line).

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

Approx throughput normalized to the same training horizon (**step=500 -> 15000**) using `elapsed` from `logs/w3_25m/*.log`:

| run | steps/sec | sec/step |
|---|---:|---:|
| w3_25m_screen_vanilla_25m.log | 4.964 | 0.20 |
| w3_25m_screen_ngt_default.log | 0.829 | 1.21 |
| w3_25m_screen_ngt_no_repulsion.log | 0.829 | 1.21 |
| w3_25m_screen_ngt_repulsion_interval_8.log | 0.830 | 1.21 |
| w3_25m_screen_ngt_mass_in_value.log | 0.852 | 1.17 |
| w3_25m_screen_ngt_no_radius.log | 0.855 | 1.17 |

Rule of thumb: NGT runs here are ~0.83-0.86 steps/sec vs vanilla ~4.97 steps/sec (~5.8-6.0x slower) under the same batch/accum/block settings.

## HTML visualizations (coords)

Each file below is a self-contained Plotly HTML (can be opened locally in a browser):

- `coords_ngt_default.html` -> `ngt_default` checkpoint
- `coords_ngt_mass_in_value.html` -> `ngt_mass_in_value` checkpoint
- `coords_ngt_no_repulsion.html` -> `ngt_no_repulsion` checkpoint
- `coords_ngt_repulsion_interval_8.html` -> `ngt_repulsion_interval_8` checkpoint
- `coords_ngt_no_radius.html` -> `ngt_no_radius` checkpoint

Note: Hover labels may include tokenizer-specific markers like `Ġ` (common in byte-level BPE) indicating a leading space; it is not a literal character from the input text.
