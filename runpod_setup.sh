#!/bin/bash
set -e

# ============================================================
#  NGT RunPod Environment Setup
#  - GPU-optimized PyTorch + dependencies
#  - Data pre-download (shakespeare + wikitext103)
#  - TensorBoard auto-start
# ============================================================

REPO_URL="https://github.com/leetae9yu/newton-gravity-transformer.git"
WORK_DIR="/workspace/ngt"

echo "============================================"
echo " NGT RunPod Setup"
echo "============================================"

# --- 1. Clone repo ---
if [ -d "$WORK_DIR" ]; then
    echo "[1/5] Repo exists, pulling latest..."
    cd "$WORK_DIR"
    git pull
else
    echo "[1/5] Cloning repo..."
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

# --- 2. Install dependencies ---
echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Ensure CUDA torch is installed (RunPod images usually have it, but just in case)
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "WARNING: CUDA not available. Reinstalling PyTorch with CUDA..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
fi

# --- 3. Prepare data ---
echo "[3/5] Preparing datasets..."
mkdir -p data checkpoints

# Shakespeare
python prepare_data.py --dataset shakespeare
# WikiText-103 (HuggingFace will cache it)
python prepare_data.py --dataset wikitext103

# --- 4. Pre-encode wikitext103 with tiktoken ---
echo "[4/5] Pre-encoding wikitext103 with tiktoken (one-time cost)..."
python -c "
from tokenizer_utils import build_tokenizer
from data_utils import load_dataset
tok = build_tokenizer('', 'tiktoken', 0)
splits = load_dataset('wikitext103', tok, 'data')
print(f'  train: {len(splits[\"train\"]):,} tokens')
print(f'  val:   {len(splits[\"val\"]):,} tokens')
print(f'  test:  {len(splits[\"test\"]):,} tokens')
print('Pre-encoding complete.')
"

# --- 5. Start TensorBoard in background ---
echo "[5/5] Starting TensorBoard on port 6006..."
mkdir -p runs
nohup tensorboard --logdir runs --bind_all --port 6006 > /tmp/tensorboard.log 2>&1 &
echo "  TensorBoard PID: $!"
echo "  Access: http://<your-pod-ip>:6006"

# --- Done ---
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
echo ""
echo "Quick start commands:"
echo ""
echo "  # Shakespeare baseline (NGT vs Vanilla)"
echo "  python train_shakespeare.py --use-cosine-schedule --warmup-steps 500"
echo "  python train_shakespeare_vanilla.py --use-cosine-schedule --warmup-steps 500"
echo ""
echo "  # WikiText-103 (full experiment)"
echo "  python train_shakespeare.py --dataset wikitext103 --tokenizer tiktoken --use-cosine-schedule --warmup-steps 2000 --use-amp"
echo "  python train_shakespeare_vanilla.py --dataset wikitext103 --tokenizer tiktoken --use-cosine-schedule --warmup-steps 2000 --use-amp"
echo ""
echo "  # Ablation examples"
echo "  python train_shakespeare.py --dataset wikitext103 --tokenizer tiktoken --use-cosine-schedule --warmup-steps 2000 --use-amp --no-radius-cutoff"
echo "  python train_shakespeare.py --dataset wikitext103 --tokenizer tiktoken --use-cosine-schedule --warmup-steps 2000 --use-amp --no-repulsion"
echo "  python train_shakespeare.py --dataset wikitext103 --tokenizer tiktoken --use-cosine-schedule --warmup-steps 2000 --use-amp --use-rsqrt"
echo ""
echo "  # Compare results after training"
echo "  python compare_runs.py checkpoints/*.pt"
echo ""
echo "  # Monitor TensorBoard"
echo "  # Open http://<your-pod-ip>:6006 in browser"
echo ""
