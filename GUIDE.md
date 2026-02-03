# Runpod First-Run Guide (WikiText-103 ~25M NGT Ablations)

This guide documents the RunPod + command-line setup used for the current experiment plan:

- Dataset: `wikitext103`
- Tokenizer: `bpe` (vocab 8192) with persistent tokenizer file
- Model size target: ~25M params
- Runner script: `run_wikitext103_25m.sh` in `budget10` mode
- Screening: `15k` steps per run
- Long run: `50k` steps on the selected NGT checkpoint (often via `--resume`)

---

## 1) Runpod UI Settings

Recommended (good defaults; adjust to your needs):

- **GPU**: whichever is available/affordable (A4500/A5000/3090 are fine)
- **Disk**: at least **50-200GB** (WikiText-103 cache + checkpoints)
- **Expose HTTP Port**: `6006` (TensorBoard), if your template/account allows it
- **Container**: any PyTorch/CUDA image that includes `python` + `pip`

Optional but recommended:

- **Persistent volume** for the repo or at least the `data/` directory so caches survive pod resets.

---

## 2) Pod Setup (Fresh Pod)

```bash
git clone https://github.com/leetae9yu/newton-gravity-transformer.git
cd newton-gravity-transformer

pip install -r requirements.txt
mkdir -p logs
```

If the repo already exists:

```bash
cd newton-gravity-transformer
git pull
```

---

## 3) Start TensorBoard (Port 6006)

```bash
nohup tensorboard --logdir runs --port 6006 --bind_all > logs/tensorboard.out 2>&1 &
echo $! > logs/tensorboard.pid
```

Check it is running:

```bash
ps -p $(cat logs/tensorboard.pid) -o pid,cmd
tail -n 30 logs/tensorboard.out
```

---

## 4) (Optional) One-Time Speed Tests

These help estimate cost/time on the current GPU. They reuse the same tokenizer + cached dataset.

### 4.1 Vanilla (500 steps)

```bash
python train_shakespeare_vanilla.py --dataset wikitext103 --data-path data \
  --tokenizer bpe --bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json \
  --max-steps 500 --eval-interval 500 --eval-iters 20 \
  --block-size 512 --batch-size 16 --gradient-accumulation-steps 2 \
  --use-amp --use-cosine-schedule --warmup-steps 2000 \
  --hidden-dim 512 --num-layers 8 --num-heads 8 --mlp-dim 1536 \
  --checkpoint-path checkpoints/_speedtest_vanilla_500.pt --run-name _speedtest_vanilla_500
```

### 4.2 NGT (500 steps)

```bash
python train_shakespeare.py --dataset wikitext103 --data-path data \
  --tokenizer bpe --bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json \
  --max-steps 500 --eval-interval 500 --eval-iters 20 \
  --block-size 512 --batch-size 16 --gradient-accumulation-steps 2 \
  --use-amp --use-cosine-schedule --warmup-steps 2000 --use-rsqrt \
  --hidden-dim 512 --coord-dim 64 --num-layers 8 --num-heads 8 --mlp-dim 2048 \
  --checkpoint-path checkpoints/_speedtest_ngt_500.pt --run-name _speedtest_ngt_500
```

---

## 5) Main Experiment (One Command)

This runs:

- Vanilla params-match baseline (once)
- NGT ablation set (budget10)
- Generates:
  - `results/w3_25m/report.md`
  - `results/w3_25m/results.csv`
  - `results/w3_25m/coords_*.html` (3D PCA visualization per run)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
MAX_STEPS_SCREEN=15000 MAX_STEPS_FINAL=50000 \
EVAL_ITERS_SCREEN=20 EVAL_ITERS_FINAL=50 \
REPULSION_INTERVAL=4 \
nohup bash run_wikitext103_25m.sh budget10 > logs/budget15k.out 2>&1 &
echo $! > logs/budget15k.pid
```

---

## 6) Monitoring

Top-level runner log:

```bash
tail -f logs/budget15k.out
```

Per-run logs:

```bash
ls -lt logs/w3_25m | head
tail -f logs/w3_25m/*.log
```

Quick "is it alive?" checks:

```bash
ps -p $(cat logs/budget15k.pid) -o pid,cmd
ps aux | egrep "train_shakespeare(_vanilla)?\\.py|run_wikitext103_25m\\.sh" | grep -v grep
nvidia-smi
```

Notes:
- `Ctrl+C` stops `tail -f` only; it does not stop any `nohup`-started training process.

### CUDA OOM quick fixes

If you see `torch.OutOfMemoryError` during backward (common on 20GB cards), try:

```bash
# 1) Kill other GPU jobs (most common cause)
nvidia-smi

# 2) Reduce per-step memory without changing effective batch too much
BATCH_SIZE=8 ACCUM_STEPS=4 bash run_wikitext103_25m.sh budget10

# 3) Lower context length (big memory lever)
BLOCK_SIZE=384 bash run_wikitext103_25m.sh budget10
```

Progress without relying on stdout (useful if output buffering happens):

```bash
ls -lt runs/w3_25m/screen_ngt_default | head
ls -lh checkpoints/w3_25m/*.pt_best.pt | head
```

---

## 7) Outputs (Where to Look)

- Checkpoints: `checkpoints/w3_25m/`
  - per run: `*_best.pt` and `*_last.pt`
- TensorBoard events: `runs/w3_25m/`
- Report + CSV + HTML visualizations: `results/w3_25m/`

---

## 8) Stop / Resume

Stop runner (safe):

```bash
kill $(cat logs/budget15k.pid) 2>/dev/null || true
```

Stop TensorBoard:

```bash
kill $(cat logs/tensorboard.pid) 2>/dev/null || true
```

Resume:

- The training scripts support `--resume`, but the runner script may not auto-add it in all modes.
- For long runs, prefer continuing from `*_last.pt` by passing `--resume` and the same `--checkpoint-path` base used originally.

---

## 9) Download Results (No SSH / No Extra Ports)

If `scp` is blocked and you cannot expose extra HTTP ports, use `runpodctl send/receive`.

### 9.1 Pod -> Local

On the RunPod pod:

```bash
cd /newton-gravity-transformer
tar -cJf /tmp/w3_25m_results_latest.tar.xz results/w3_25m
runpodctl send /tmp/w3_25m_results_latest.tar.xz
```

This prints a one-time `<CODE>`.

On local Windows PowerShell:

```powershell
.\runpodctl.exe receive <CODE>
```

Extract with 7-Zip (`.tar.xz` -> `.tar` -> folder).

### 9.2 Install `runpodctl` on Windows (PowerShell)

If `Invoke-WebRequest` fails (TLS / "connection closed"), `Start-BitsTransfer` is usually more reliable.

```powershell
Start-BitsTransfer -Source "https://github.com/runpod/runpodctl/releases/download/v1.14.3/runpodctl-windows-amd64.exe" -Destination ".\\runpodctl.exe"
Unblock-File .\\runpodctl.exe
.\\runpodctl.exe --help
```

Fallback:

```powershell
curl.exe -L --retry 5 --retry-delay 2 -o runpodctl.exe "https://github.com/runpod/runpodctl/releases/download/v1.14.3/runpodctl-windows-amd64.exe"
```

---

## 10) Quick Stats From Logs (Screening @15000)

All screening runs in `w3_25m` are expected to reach `iter=15000` (`*_last.pt`).

### 10.1 Final val loss @15000 (per run)

```bash
grep -a "^step 15000/15000" logs/w3_25m/w3_25m_screen_*.log
```

### 10.2 Check checkpoint `iter`

```bash
python - <<'PY'
import glob, torch
for p in sorted(glob.glob("checkpoints/w3_25m/*_last.pt")):
    ck=torch.load(p, map_location="cpu", weights_only=False)
    print(p.split("/")[-1], "iter=", ck.get("iter"))
PY
```

### 10.3 Throughput (normalized step=500 -> 15000)

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

---

## 11) Resume Top-1 NGT To 50k (Continue Training)

Example: resume `ngt_mass_in_value` from its `*_last.pt` (if present) to `max_steps=50000`.

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

Monitor:

```bash
tail -f logs/w3_25m/final_ngt_mass_in_value_resume_50k.out
```
