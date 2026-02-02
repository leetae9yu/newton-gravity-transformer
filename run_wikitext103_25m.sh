#!/usr/bin/env bash
set -euo pipefail

# WikiText-103 (BPE 8192) ~25M screening + optional final run.
#
# Usage:
#   bash run_wikitext103_25m.sh screening
#   bash run_wikitext103_25m.sh final
#   bash run_wikitext103_25m.sh all
#
# Notes:
# - Expects you to expose TensorBoard port 6006 in Runpod if using TENSORBOARD=1.
# - Overrides via env vars (examples):
#     SEED=42 MAX_STEPS_SCREEN=12000 MAX_STEPS_FINAL=50000 bash run_wikitext103_25m.sh all
#     BATCH_SIZE=32 ACCUM_STEPS=1 bash run_wikitext103_25m.sh screening

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-all}" # screening|final|all

RUN_GROUP="${RUN_GROUP:-w3_25m}"
DATA_PATH="${DATA_PATH:-data}"
SEED="${SEED:-42}"

# Training budget
MAX_STEPS_SCREEN="${MAX_STEPS_SCREEN:-12000}"
MAX_STEPS_FINAL="${MAX_STEPS_FINAL:-50000}"

# Evaluation + logging
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_ITERS_SCREEN="${EVAL_ITERS_SCREEN:-50}"
EVAL_ITERS_FINAL="${EVAL_ITERS_FINAL:-100}"
VIS_INTERVAL_SCREEN="${VIS_INTERVAL_SCREEN:-0}"   # 0 disables projector logging
VIS_INTERVAL_FINAL="${VIS_INTERVAL_FINAL:-2000}"  # set 0 to disable

# Batch / compute
BLOCK_SIZE="${BLOCK_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"

# Optim
LR="${LR:-3e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
DROPOUT="${DROPOUT:-0.1}"

# Schedule / AMP
USE_COSINE="${USE_COSINE:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
USE_AMP="${USE_AMP:-1}"
USE_RSQRT="${USE_RSQRT:-1}"

# Model (NGT ~25.6M, Vanilla params-match ~25.5M)
NGT_HIDDEN_DIM="${NGT_HIDDEN_DIM:-512}"
NGT_COORD_DIM="${NGT_COORD_DIM:-64}"
NGT_LAYERS="${NGT_LAYERS:-8}"
NGT_HEADS="${NGT_HEADS:-8}"
NGT_MLP_DIM="${NGT_MLP_DIM:-2048}"

VAN_HIDDEN_DIM="${VAN_HIDDEN_DIM:-512}"
VAN_LAYERS="${VAN_LAYERS:-8}"
VAN_HEADS="${VAN_HEADS:-8}"
VAN_MLP_DIM="${VAN_MLP_DIM:-1536}"

# Tokenizer
TOKENIZER="${TOKENIZER:-bpe}"
BPE_VOCAB_SIZE="${BPE_VOCAB_SIZE:-8192}"

# Visualization (3D PCA) input text for wikitext-like token labels
VIS_TEXT="${VIS_TEXT:-$'== WikiText-style snippet ==\nThis is a small snippet to visualize coordinate space.\nThe quick brown fox jumps over the lazy dog.\n'}"

CKPT_DIR="checkpoints/${RUN_GROUP}"
LOG_DIR="logs/${RUN_GROUP}"
RESULTS_DIR="results/${RUN_GROUP}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${RUN_GROUP}}"

mkdir -p "$CKPT_DIR" "$LOG_DIR" "$RESULTS_DIR" runs

if [[ "${TENSORBOARD:-0}" == "1" ]]; then
  if [[ ! -f "${LOG_DIR}/tensorboard.pid" ]] || ! kill -0 "$(cat "${LOG_DIR}/tensorboard.pid" 2>/dev/null)" 2>/dev/null; then
    nohup tensorboard --logdir runs --port 6006 --bind_all > "${LOG_DIR}/tensorboard.out" 2>&1 &
    echo $! > "${LOG_DIR}/tensorboard.pid"
    echo "[TB] started (pid=$(cat "${LOG_DIR}/tensorboard.pid")): port 6006"
  else
    echo "[TB] already running (pid=$(cat "${LOG_DIR}/tensorboard.pid"))"
  fi
fi

COMMON_ARGS=(--dataset wikitext103 --data-path "$DATA_PATH" --tokenizer "$TOKENIZER" --bpe-vocab-size "$BPE_VOCAB_SIZE"
  --block-size "$BLOCK_SIZE" --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$ACCUM_STEPS"
  --learning-rate "$LR" --grad-clip "$GRAD_CLIP" --dropout "$DROPOUT"
  --eval-interval "$EVAL_INTERVAL" --seed "$SEED")

if [[ "$USE_COSINE" == "1" ]]; then COMMON_ARGS+=(--use-cosine-schedule --warmup-steps "$WARMUP_STEPS"); fi
if [[ "$USE_AMP" == "1" ]]; then COMMON_ARGS+=(--use-amp); fi
if [[ "$USE_RSQRT" == "1" ]]; then COMMON_ARGS+=(--use-rsqrt); fi

run() {
  local name="$1"; shift
  local logfile="${LOG_DIR}/${name}.log"
  echo ""
  echo "========================================"
  echo "[RUN] ${name}"
  echo "[CMD] $*"
  echo "[LOG] ${logfile}"
  echo "========================================"
  if bash -lc "$*" 2>&1 | tee "$logfile"; then
    echo "[OK] ${name}"
    return 0
  else
    echo "[FAIL] ${name}"
    return 1
  fi
}

viz_ngt_checkpoint() {
  local ckpt_best="$1"
  local out_html="$2"
  if [[ ! -f "$ckpt_best" ]]; then
    echo "[VIZ] skip (missing): $ckpt_best"
    return 0
  fi
  python visualize_coords.py --checkpoint-path "$ckpt_best" --text "$VIS_TEXT" --num-tokens "$BLOCK_SIZE" --output "$out_html" \
    2>&1 | tee "${LOG_DIR}/viz_$(basename "$out_html" .html).log" || true
}

screening() {
  local screen_common=("${COMMON_ARGS[@]}" --max-steps "$MAX_STEPS_SCREEN" --eval-iters "$EVAL_ITERS_SCREEN" --vis-interval "$VIS_INTERVAL_SCREEN")

  # Vanilla params-match baseline (run once)
  local van_ckpt="${CKPT_DIR}/vanilla_25m.pt"
  run "${RUN_NAME_PREFIX}_screen_vanilla_25m" \
    "python train_shakespeare_vanilla.py ${screen_common[*]} --hidden-dim ${VAN_HIDDEN_DIM} --num-layers ${VAN_LAYERS} --num-heads ${VAN_HEADS} --mlp-dim ${VAN_MLP_DIM} --checkpoint-path ${van_ckpt} --run-name ${RUN_NAME_PREFIX}/screen_vanilla_25m"

  # NGT base
  local ngt_base="python train_shakespeare.py ${screen_common[*]} --hidden-dim ${NGT_HIDDEN_DIM} --coord-dim ${NGT_COORD_DIM} --num-layers ${NGT_LAYERS} --num-heads ${NGT_HEADS} --mlp-dim ${NGT_MLP_DIM}"

  run "${RUN_NAME_PREFIX}_screen_ngt_default" \
    "${ngt_base} --checkpoint-path ${CKPT_DIR}/ngt_default.pt --run-name ${RUN_NAME_PREFIX}/screen_ngt_default"
  run "${RUN_NAME_PREFIX}_screen_ngt_no_repulsion" \
    "${ngt_base} --no-repulsion --checkpoint-path ${CKPT_DIR}/ngt_no_repulsion.pt --run-name ${RUN_NAME_PREFIX}/screen_ngt_no_repulsion"
  run "${RUN_NAME_PREFIX}_screen_ngt_no_radius" \
    "${ngt_base} --no-radius-cutoff --checkpoint-path ${CKPT_DIR}/ngt_no_radius.pt --run-name ${RUN_NAME_PREFIX}/screen_ngt_no_radius"
  run "${RUN_NAME_PREFIX}_screen_ngt_mass_in_value" \
    "${ngt_base} --mass-in-value --checkpoint-path ${CKPT_DIR}/ngt_mass_in_value.pt --run-name ${RUN_NAME_PREFIX}/screen_ngt_mass_in_value"
  run "${RUN_NAME_PREFIX}_screen_ngt_soft_cutoff" \
    "${ngt_base} --use-soft-cutoff --checkpoint-path ${CKPT_DIR}/ngt_soft_cutoff.pt --run-name ${RUN_NAME_PREFIX}/screen_ngt_soft_cutoff"

  # Visualize each NGT best checkpoint to interactive HTML
  viz_ngt_checkpoint "${CKPT_DIR}/ngt_default.pt_best.pt" "${RESULTS_DIR}/coords_ngt_default.html"
  viz_ngt_checkpoint "${CKPT_DIR}/ngt_no_repulsion.pt_best.pt" "${RESULTS_DIR}/coords_ngt_no_repulsion.html"
  viz_ngt_checkpoint "${CKPT_DIR}/ngt_no_radius.pt_best.pt" "${RESULTS_DIR}/coords_ngt_no_radius.html"
  viz_ngt_checkpoint "${CKPT_DIR}/ngt_mass_in_value.pt_best.pt" "${RESULTS_DIR}/coords_ngt_mass_in_value.html"
  viz_ngt_checkpoint "${CKPT_DIR}/ngt_soft_cutoff.pt_best.pt" "${RESULTS_DIR}/coords_ngt_soft_cutoff.html"

  # Summary + report (screening stage)
  python compare_runs.py "${CKPT_DIR}"/*_best.pt --csv "${RESULTS_DIR}/results.csv" 2>/dev/null | tee "${LOG_DIR}/compare_runs.log" || true
  python generate_report.py "${CKPT_DIR}"/*_best.pt --output "${RESULTS_DIR}/report.md" 2>/dev/null | tee "${LOG_DIR}/report.log" || true

  echo ""
  echo "========================================"
  echo "[DONE] screening"
  echo "Checkpoints: ${CKPT_DIR}"
  echo "Results:     ${RESULTS_DIR}"
  echo "Logs:        ${LOG_DIR}"
  echo "========================================"
}

pick_best_ngt_ckpt() {
  python - "$CKPT_DIR" <<'PY'
import glob
import os
import sys
import torch

ckpt_dir = sys.argv[1]
paths = sorted(glob.glob(os.path.join(ckpt_dir, "ngt_*.pt_best.pt")))
best = None
for p in paths:
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        val = ckpt.get("best_val", None)
        if val is None:
            continue
        if best is None or val < best[0]:
            best = (val, p)
    except Exception:
        continue
if best is None:
    print("")
else:
    print(best[1])
PY
}

final_run() {
  local best_ckpt
  best_ckpt="$(pick_best_ngt_ckpt)"
  if [[ -z "$best_ckpt" ]]; then
    echo "[FINAL] No NGT *_best.pt found in ${CKPT_DIR}. Run screening first (or set BEST_NGT_CKPT)."
    exit 1
  fi

  local best_name
  best_name="$(basename "$best_ckpt" .pt_best.pt)"
  echo "[FINAL] picked best: ${best_name} (${best_ckpt})"

  local final_common=("${COMMON_ARGS[@]}" --max-steps "$MAX_STEPS_FINAL" --eval-iters "$EVAL_ITERS_FINAL" --vis-interval "$VIS_INTERVAL_FINAL")
  local out_ckpt="${CKPT_DIR}/final_${best_name}.pt"

  # Re-run with the same ablation flags implied by the best checkpoint name.
  # (Keeps this script single-file; no stateful parsing beyond the name.)
  local extra_flags=()
  case "$best_name" in
    ngt_no_repulsion) extra_flags+=(--no-repulsion) ;;
    ngt_no_radius) extra_flags+=(--no-radius-cutoff) ;;
    ngt_mass_in_value) extra_flags+=(--mass-in-value) ;;
    ngt_soft_cutoff) extra_flags+=(--use-soft-cutoff) ;;
    ngt_default) : ;;
    *) echo "[FINAL] Unknown best checkpoint name pattern: ${best_name}"; exit 1 ;;
  esac

  run "${RUN_NAME_PREFIX}_final_${best_name}" \
    "python train_shakespeare.py ${final_common[*]} --hidden-dim ${NGT_HIDDEN_DIM} --coord-dim ${NGT_COORD_DIM} --num-layers ${NGT_LAYERS} --num-heads ${NGT_HEADS} --mlp-dim ${NGT_MLP_DIM} ${extra_flags[*]} --checkpoint-path ${out_ckpt} --run-name ${RUN_NAME_PREFIX}/final_${best_name}"

  viz_ngt_checkpoint "${out_ckpt}_best.pt" "${RESULTS_DIR}/coords_final_${best_name}.html"

  python compare_runs.py "${CKPT_DIR}"/*_best.pt --csv "${RESULTS_DIR}/results.csv" 2>/dev/null | tee "${LOG_DIR}/compare_runs.log" || true
  python generate_report.py "${CKPT_DIR}"/*_best.pt --output "${RESULTS_DIR}/report.md" 2>/dev/null | tee "${LOG_DIR}/report.log" || true

  echo ""
  echo "========================================"
  echo "[DONE] final"
  echo "Final checkpoint base: ${out_ckpt}"
  echo "Results:              ${RESULTS_DIR}"
  echo "========================================"
}

case "$MODE" in
  screening) screening ;;
  final) final_run ;;
  all) screening; final_run ;;
  *)
    echo "Unknown mode: $MODE (expected: screening|final|all)"
    exit 2
    ;;
esac

