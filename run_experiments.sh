#!/bin/bash
set -e

# ============================================================
#  NGT Full Experiment Suite
#
#  - NGT ablations: 3 seeds each for statistical significance
#  - Vanilla baseline: 1 run per dataset
#  - WikiText-103: 50k steps (sufficient for convergence trends)
#
#  Usage:
#    bash run_experiments.sh                  # all experiments
#    bash run_experiments.sh shakespeare      # shakespeare only
#    bash run_experiments.sh wikitext103      # wikitext103 only
# ============================================================

cd /workspace/ngt

FILTER="${1:-all}"
SEEDS="42 123 7"

# Common flags
COMMON_SHK="--use-cosine-schedule --warmup-steps 500"
COMMON_WIKI="--dataset wikitext103 --tokenizer tiktoken --use-cosine-schedule --warmup-steps 2000 --use-amp --max-steps 50000"

STARTED=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL=0
FAILED=0

run() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "========================================"
    echo " [$TOTAL] $name"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo " > $@"
    echo ""
    if time "$@"; then
        echo " [OK] $name"
    else
        echo " [FAIL] $name"
        FAILED=$((FAILED + 1))
    fi
    echo "========================================"
}

# ============================================================
#  Shakespeare (5k steps, fast)
# ============================================================
if [ "$FILTER" = "all" ] || [ "$FILTER" = "shakespeare" ]; then
    echo ""
    echo "########################################"
    echo "#  Phase 1: Shakespeare                #"
    echo "########################################"

    # --- Vanilla baseline (1 run) ---
    run "Vanilla Shakespeare" \
        python train_shakespeare_vanilla.py $COMMON_SHK --seed 42

    # --- NGT ablations (3 seeds each) ---
    for SEED in $SEEDS; do
        run "NGT Shakespeare default (seed=$SEED)" \
            python train_shakespeare.py $COMMON_SHK --seed $SEED
    done

    for SEED in $SEEDS; do
        run "NGT Shakespeare no_radius (seed=$SEED)" \
            python train_shakespeare.py $COMMON_SHK --seed $SEED --no-radius-cutoff
    done

    for SEED in $SEEDS; do
        run "NGT Shakespeare no_repulsion (seed=$SEED)" \
            python train_shakespeare.py $COMMON_SHK --seed $SEED --no-repulsion
    done

    for SEED in $SEEDS; do
        run "NGT Shakespeare rsqrt (seed=$SEED)" \
            python train_shakespeare.py $COMMON_SHK --seed $SEED --use-rsqrt
    done

    for SEED in $SEEDS; do
        run "NGT Shakespeare mass_val (seed=$SEED)" \
            python train_shakespeare.py $COMMON_SHK --seed $SEED --mass-in-value
    done

    for SEED in $SEEDS; do
        run "NGT Shakespeare soft_cutoff (seed=$SEED)" \
            python train_shakespeare.py $COMMON_SHK --seed $SEED --use-soft-cutoff
    done
fi

# ============================================================
#  WikiText-103 (50k steps)
# ============================================================
if [ "$FILTER" = "all" ] || [ "$FILTER" = "wikitext103" ]; then
    echo ""
    echo "########################################"
    echo "#  Phase 2: WikiText-103 (50k steps)   #"
    echo "########################################"

    # --- Vanilla baseline (1 run) ---
    run "Vanilla WikiText-103" \
        python train_shakespeare_vanilla.py $COMMON_WIKI --seed 42

    # --- NGT ablations (3 seeds each) ---
    for SEED in $SEEDS; do
        run "NGT WikiText-103 default (seed=$SEED)" \
            python train_shakespeare.py $COMMON_WIKI --seed $SEED
    done

    for SEED in $SEEDS; do
        run "NGT WikiText-103 no_radius (seed=$SEED)" \
            python train_shakespeare.py $COMMON_WIKI --seed $SEED --no-radius-cutoff
    done

    for SEED in $SEEDS; do
        run "NGT WikiText-103 no_repulsion (seed=$SEED)" \
            python train_shakespeare.py $COMMON_WIKI --seed $SEED --no-repulsion
    done

    for SEED in $SEEDS; do
        run "NGT WikiText-103 rsqrt (seed=$SEED)" \
            python train_shakespeare.py $COMMON_WIKI --seed $SEED --use-rsqrt
    done

    for SEED in $SEEDS; do
        run "NGT WikiText-103 mass_val (seed=$SEED)" \
            python train_shakespeare.py $COMMON_WIKI --seed $SEED --mass-in-value
    done

    for SEED in $SEEDS; do
        run "NGT WikiText-103 soft_cutoff (seed=$SEED)" \
            python train_shakespeare.py $COMMON_WIKI --seed $SEED --use-soft-cutoff
    done
fi

# ============================================================
#  Summary
# ============================================================
echo ""
echo "============================================"
echo " All experiments complete!"
echo " Started: $STARTED"
echo " Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
echo " Total:   $TOTAL runs ($FAILED failed)"
echo "============================================"
echo ""
echo "Results:"
python compare_runs.py checkpoints/*.pt 2>/dev/null || echo "(no checkpoints found)"
echo ""
python compare_runs.py checkpoints/*.pt --csv results.csv 2>/dev/null && echo "Saved: results.csv" || true
echo ""
echo "TensorBoard: http://<your-pod-ip>:6006"
echo ""
echo "To download results:"
echo "  tar czf results_bundle.tar.gz runs/ checkpoints/ results.csv"
