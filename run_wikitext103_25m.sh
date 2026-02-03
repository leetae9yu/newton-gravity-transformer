#!/usr/bin/env bash
set -euo pipefail

# Minimal runner: WikiText-103 (~25M) at 15k steps
# - vanilla baseline (params-match)
# - NGT (--mass-in-value)
#
# Usage:
#   bash run_wikitext103_25m.sh
#
# Overrides:
#   RUN_GROUP=w3_25m SEED=42 MAX_STEPS=15000 bash run_wikitext103_25m.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUN_GROUP="${RUN_GROUP:-w3_25m}"
DATA_PATH="${DATA_PATH:-data}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-15000}"

# Tokenizer
TOKENIZER="${TOKENIZER:-bpe}"
BPE_VOCAB_SIZE="${BPE_VOCAB_SIZE:-8192}"
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tokenizer_${TOKENIZER}_${BPE_VOCAB_SIZE}.json}"

# Batch / compute
BLOCK_SIZE="${BLOCK_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"

# Optim / eval
LR="${LR:-3e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
DROPOUT="${DROPOUT:-0.1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_ITERS="${EVAL_ITERS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"

# Model sizes (~25M-ish)
VAN_HIDDEN_DIM="${VAN_HIDDEN_DIM:-512}"
VAN_LAYERS="${VAN_LAYERS:-8}"
VAN_HEADS="${VAN_HEADS:-8}"
VAN_MLP_DIM="${VAN_MLP_DIM:-1536}"

NGT_HIDDEN_DIM="${NGT_HIDDEN_DIM:-512}"
NGT_COORD_DIM="${NGT_COORD_DIM:-64}"
NGT_LAYERS="${NGT_LAYERS:-8}"
NGT_HEADS="${NGT_HEADS:-8}"
NGT_MLP_DIM="${NGT_MLP_DIM:-2048}"

CKPT_DIR="checkpoints/${RUN_GROUP}"
LOG_DIR="logs/${RUN_GROUP}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${RUN_GROUP}}"

mkdir -p "$CKPT_DIR" "$LOG_DIR" runs

COMMON_ARGS=(--dataset wikitext103 --data-path "$DATA_PATH"
  --tokenizer "$TOKENIZER" --bpe-vocab-size "$BPE_VOCAB_SIZE" --tokenizer-path "$TOKENIZER_PATH"
  --block-size "$BLOCK_SIZE" --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$ACCUM_STEPS"
  --learning-rate "$LR" --grad-clip "$GRAD_CLIP" --dropout "$DROPOUT"
  --eval-interval "$EVAL_INTERVAL" --eval-iters "$EVAL_ITERS" --seed "$SEED"
  --use-cosine-schedule --warmup-steps "$WARMUP_STEPS" --use-amp)

run() {
  local name="$1"; shift
  local logfile="${LOG_DIR}/${name}.log"
  echo ""
  echo "========================================"
  echo "[RUN] ${name}"
  echo "[LOG] ${logfile}"
  echo "========================================"
  bash -lc "$*" 2>&1 | tee "$logfile"
}

# Vanilla baseline
run "${RUN_NAME_PREFIX}_screen_vanilla_25m" \
  "python train_shakespeare_vanilla.py ${COMMON_ARGS[*]} --max-steps ${MAX_STEPS} \
    --hidden-dim ${VAN_HIDDEN_DIM} --num-layers ${VAN_LAYERS} --num-heads ${VAN_HEADS} --mlp-dim ${VAN_MLP_DIM} \
    --checkpoint-path ${CKPT_DIR}/vanilla_25m.pt --run-name ${RUN_NAME_PREFIX}/screen_vanilla_25m"

# NGT mass-in-value
run "${RUN_NAME_PREFIX}_screen_ngt_mass_in_value" \
  "python train_shakespeare.py ${COMMON_ARGS[*]} --max-steps ${MAX_STEPS} --use-rsqrt --repulsion-interval 4 \
    --hidden-dim ${NGT_HIDDEN_DIM} --coord-dim ${NGT_COORD_DIM} --num-layers ${NGT_LAYERS} --num-heads ${NGT_HEADS} --mlp-dim ${NGT_MLP_DIM} \
    --mass-in-value \
    --checkpoint-path ${CKPT_DIR}/ngt_mass_in_value.pt --run-name ${RUN_NAME_PREFIX}/screen_ngt_mass_in_value"

echo ""
echo "========================================"
echo "[DONE] wikitext103 @${MAX_STEPS}"
echo "Checkpoints: ${CKPT_DIR}"
echo "Logs:        ${LOG_DIR}"
echo "========================================"

