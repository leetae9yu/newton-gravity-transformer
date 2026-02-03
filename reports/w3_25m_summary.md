# w3_25m Experiment Summary (WikiText-103, ~25M)

This repo intentionally tracks a **minimal** report for the 15k-step snapshot comparing:
- vanilla baseline
- NGT with `--mass-in-value`

## Setup

- Dataset: `wikitext103` (HuggingFace `wikitext-103-raw-v1`)
- Tokenizer: BPE, vocab size `8192`
- Context length: `512`
- Seed: `42`
- Max steps: `15000`

Reproduce with:
```bash
python prepare_data.py --dataset wikitext103
bash run_wikitext103_25m.sh
```

## Results (@15000)

Validation loss is cross-entropy; perplexity is `exp(loss)`.

| model | config | val loss @15000 | ppl |
|---|---|---:|---:|
| vanilla | baseline | 4.5554 | 95.14 |
| NGT | `--mass-in-value` | 4.6635 | 106.01 |

## Notes

- Checkpoint naming: when training uses `--checkpoint-path checkpoints/foo.pt`, it writes `checkpoints/foo.pt_best.pt` and `checkpoints/foo.pt_last.pt`.
- Token hover text in downstream visualizations (if you generate any locally) may contain tokenizer whitespace markers like `Ġ` / `Ċ` (U+0120 / U+010A), indicating a leading space/newline. These are tokenizer artifacts, not literal input characters.

