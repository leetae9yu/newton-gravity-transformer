#!/bin/bash
set -e

# ============================================================
#  NGT Full Experiment Suite
#  Run all experiments sequentially on a single GPU.
#  Checkpoints + TensorBoard logs are saved per-run.
#
#  Usage:
#    bash run_experiments.sh                  # all experiments
#    bash run_experiments.sh shakespeare      # shakespeare only
#    bash run_experiments.sh wikitext103      # wikitext103 only
# ============================================================

cd /workspace/ngt

FILTER="${1:-all}"

# Common flags
COMMON_SHAKESPEARE="--use-cosine-schedule --warmup-steps 500"
COMMON_WIKI="--dataset wikitext103 --tokenizer tiktoken --use-cosine-schedule --warmup-steps 2000 --use-amp"

run() {
    local name="$1"
    shift
    echo ""
    echo "========================================"
    echo " Experiment: $name"
    echo " Command: $@"
    echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    time "$@"
    echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
}

# ---- Shakespeare experiments ----
if [ "$FILTER" = "all" ] || [ "$FILTER" = "shakespeare" ]; then
    echo "=== Phase 1: Shakespeare ==="

    run "NGT Shakespeare (default)" \
        python train_shakespeare.py $COMMON_SHAKESPEARE

    run "Vanilla Shakespeare (baseline)" \
        python train_shakespeare_vanilla.py $COMMON_SHAKESPEARE

    run "NGT Shakespeare (no radius cutoff)" \
        python train_shakespeare.py $COMMON_SHAKESPEARE --no-radius-cutoff

    run "NGT Shakespeare (no repulsion)" \
        python train_shakespeare.py $COMMON_SHAKESPEARE --no-repulsion

    run "NGT Shakespeare (rsqrt)" \
        python train_shakespeare.py $COMMON_SHAKESPEARE --use-rsqrt

    run "NGT Shakespeare (mass in value)" \
        python train_shakespeare.py $COMMON_SHAKESPEARE --mass-in-value

    run "NGT Shakespeare (soft cutoff)" \
        python train_shakespeare.py $COMMON_SHAKESPEARE --use-soft-cutoff
fi

# ---- WikiText-103 experiments ----
if [ "$FILTER" = "all" ] || [ "$FILTER" = "wikitext103" ]; then
    echo "=== Phase 2: WikiText-103 ==="

    run "NGT WikiText-103 (default)" \
        python train_shakespeare.py $COMMON_WIKI

    run "Vanilla WikiText-103 (baseline)" \
        python train_shakespeare_vanilla.py $COMMON_WIKI

    run "NGT WikiText-103 (no radius cutoff)" \
        python train_shakespeare.py $COMMON_WIKI --no-radius-cutoff

    run "NGT WikiText-103 (no repulsion)" \
        python train_shakespeare.py $COMMON_WIKI --no-repulsion

    run "NGT WikiText-103 (rsqrt)" \
        python train_shakespeare.py $COMMON_WIKI --use-rsqrt

    run "NGT WikiText-103 (mass in value)" \
        python train_shakespeare.py $COMMON_WIKI --mass-in-value

    run "NGT WikiText-103 (soft cutoff)" \
        python train_shakespeare.py $COMMON_WIKI --use-soft-cutoff
fi

# ---- Summary ----
echo ""
echo "============================================"
echo " All experiments complete!"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "Results:"
python compare_runs.py checkpoints/*.pt 2>/dev/null || echo "(no checkpoints found)"
echo ""
echo "CSV export:"
python compare_runs.py checkpoints/*.pt --csv results.csv 2>/dev/null || true
echo ""
echo "TensorBoard: http://<your-pod-ip>:6006"
