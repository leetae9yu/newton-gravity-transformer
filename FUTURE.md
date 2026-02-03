# Handoff / Notes (2026-02-03)

This file is a lightweight "what happened + how to continue" log for the current RunPod WikiText-103 (~25M) experiment workflow.

## Current status

- Dataset: `wikitext103`
- Tokenizer: `bpe` with vocab `8192` (`data/tokenizer_bpe_8192.json`)
- Screening suite completed for the `w3_25m` run group:
  - All runs reached `iter=15000` (`checkpoints/w3_25m/*_last.pt` all show `iter=15000`).
  - Logs: `logs/budget15k.out`, per-run logs in `logs/w3_25m/`
  - TensorBoard: `runs/w3_25m/...`
  - Generated: `results/w3_25m/report.md`, `results/w3_25m/results.csv`, `results/w3_25m/coords_*.html`
- Best NGT variant in screening:
  - `ngt_mass_in_value` is top among NGT variants by both best-val and final @15000 (see `w3_25m_results/results/w3_25m/Summary.md`).
- Long run in progress:
  - Resume training `ngt_mass_in_value` from `iter=15000` to `max_steps=50000` using `--resume` + `nohup` (see "Resume 50k" below).

## What was done in this session

- Standardized file transfer on `runpodctl send/receive` (SSH `scp` via `ssh.runpod.io` failed with `Permission denied (publickey)`; exposing extra HTTP ports was not reliable without risky Pod edits).
- Built a local bundle of results (`w3_25m_results_latest/...`) and wrote a consolidated report:
  - `w3_25m_results/results/w3_25m/Summary.md`
    - Fixed-step comparison @15000 (final step).
    - Best vs last per run.
    - Throughput normalized over step=500->15000.
    - Note about tokenizer hover labels: byte-level BPE may show a "leading-space" marker (often rendered as `\u0120` / "G-with-dot").
- Verified the Plotly HTML structure by parsing `coords_ngt_mass_in_value.html`:
  - 3D scatter (`scatter3d`) uses `marker.color = mass` and `marker.size` derived from mass.
  - Token labels are tokenizer outputs, so markers like `\u0120W` can appear even if the raw text does not contain those literal characters.

## Resume 50k (NGT mass-in-value)

The long run is intended to be a true continuation (resume), not a fresh re-train.

### Confirm checkpoint state

```bash
cd /newton-gravity-transformer
python -c "import torch; ck=torch.load('checkpoints/w3_25m/ngt_mass_in_value.pt_last.pt', map_location='cpu', weights_only=False); print('iter=',ck.get('iter')); print(ck.get('config',{}))"
```

### Start (nohup)

```bash
cd /newton-gravity-transformer
mkdir -p logs/w3_25m
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
nohup python train_shakespeare.py \
  --dataset wikitext103 --data-path data \
  --tokenizer bpe --bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json \
  --block-size 512 --batch-size 16 --gradient-accumulation-steps 2 \
  --learning-rate 3e-4 --grad-clip 1.0 --dropout 0.1 \
  --eval-interval 500 --eval-iters 100 --seed 42 \
  --use-cosine-schedule --warmup-steps 2000 --use-amp --use-rsqrt \
  --repulsion-interval 4 --mass-in-value \
  --hidden-dim 512 --coord-dim 64 --num-layers 8 --num-heads 8 --mlp-dim 2048 \
  --max-steps 50000 --resume \
  --checkpoint-path checkpoints/w3_25m/ngt_mass_in_value.pt \
  --run-name w3_25m/final_ngt_mass_in_value_resume \
  > logs/w3_25m/final_ngt_mass_in_value_resume_50k.out 2>&1 &
echo $! > logs/w3_25m/final_ngt_mass_in_value_resume_50k.pid
```

### Monitor

```bash
tail -f logs/w3_25m/final_ngt_mass_in_value_resume_50k.out
```

Notes:
- `Ctrl+C` stops `tail -f` only; it does not stop the `nohup` training job.
- The script prints `step ... elapsed=...` at `eval_interval` boundaries only (so the first "step line" after resuming at 15000 appears at 15500 if `eval_interval=500`).
- `elapsed` is time since this run started (not total historical training time).

## How to extract key numbers from logs

### Final (step=15000) val loss per run

```bash
grep -a "^step 15000/15000" logs/w3_25m/w3_25m_screen_*.log
```

### Check actual training step in checkpoints

```bash
python - <<'PY'
import glob, torch
for p in sorted(glob.glob("checkpoints/w3_25m/*_last.pt")):
    ck=torch.load(p, map_location="cpu", weights_only=False)
    print(p.split("/")[-1], "iter=", ck.get("iter"))
PY
```

### Throughput (normalized to step=500 -> 15000)

```bash
python - <<'PY'
import glob, os, re
START=500; TARGET=15000
PAT=re.compile(r"^step (\\d+)/(\\d+).*elapsed=([0-9.]+)s")
def m(path):
    d={}
    for line in open(path,"r",errors="ignore"):
        g=PAT.match(line.strip())
        if g: d[int(g.group(1))]=float(g.group(3))
    return d
for p in sorted(glob.glob("logs/w3_25m/w3_25m_screen_*.log")):
    d=m(p)
    if START in d and TARGET in d and d[TARGET]>d[START]:
        sp=(TARGET-START)/(d[TARGET]-d[START])
        print(f"{os.path.basename(p)}\\t{sp:.3f} step/s\\t({1/sp:.2f} s/step)")
PY
```

## File transfer (no SSH / no extra ports)

### Pod -> local via `runpodctl`

On the Pod:

```bash
cd /newton-gravity-transformer
tar -cJf /tmp/w3_25m_results_latest.tar.xz results/w3_25m
runpodctl send /tmp/w3_25m_results_latest.tar.xz
```

On local Windows PowerShell:

```powershell
.\runpodctl.exe receive <CODE>
```

Then extract with 7-Zip (`.tar.xz` -> `.tar` -> folder).

## Pitfalls / gotchas

- `scp` to `ssh.runpod.io` typically fails unless you set up SSH keys correctly and RunPod allows it; prefer `runpodctl send/receive`.
- Port exposure may be restricted by template/account. Avoid "Edit Pod" if it warns about data loss.
- Plot hover labels are tokenized outputs; a leading-space marker (often rendered as `\u0120`) is expected for byte-level BPE tokenizers.
- The PyTorch warning about `lr_scheduler.step()` order is a warning (not an immediate crash).

