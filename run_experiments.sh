#!/bin/bash
set -e

# ============================================================
#  NGT Scale-Up Ablation Suite
#
#  WikiText-103 + BPE 8192 + Full Ablation (8 runs)
#  A5000 24GB VRAM, seed=42 fixed
#
#  Usage:
#    bash run_experiments.sh              # all (sanity + full)
#    bash run_experiments.sh sanity       # Phase 0 only (5K steps)
#    bash run_experiments.sh full         # Phase 1 only (50K steps)
# ============================================================

cd /workspace/ngt

FILTER="${1:-all}"
SEED=42

# Common flags for all WikiText-103 BPE experiments
COMMON="--dataset wikitext103 --tokenizer bpe --bpe-vocab-size 8192 \
  --batch-size 16 --gradient-accumulation-steps 2 \
  --use-cosine-schedule --warmup-steps 2000 --use-amp --use-rsqrt --seed $SEED"

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
#  Phase 0: Sanity Check (5K steps × 2)
# ============================================================
if [ "$FILTER" = "all" ] || [ "$FILTER" = "sanity" ]; then
    echo ""
    echo "########################################"
    echo "#  Phase 0: Sanity Check (5K steps)    #"
    echo "########################################"

    run "Sanity: Vanilla 5K" \
        python train_shakespeare_vanilla.py $COMMON \
        --max-steps 5000 --eval-interval 500 \
        --checkpoint-path checkpoints/sanity_vanilla.pt \
        --run-name sanity_vanilla

    run "Sanity: NGT default 5K" \
        python train_shakespeare.py $COMMON \
        --max-steps 5000 --eval-interval 500 \
        --checkpoint-path checkpoints/sanity_ngt.pt \
        --run-name sanity_ngt_default

    echo ""
    echo "Phase 0 complete. Check TensorBoard for loss curves before proceeding."
    echo "If losses are decreasing and no OOM, proceed with: bash run_experiments.sh full"
    echo ""

    if [ "$FILTER" = "sanity" ]; then
        echo "============================================"
        echo " Sanity check done. Total: $TOTAL runs ($FAILED failed)"
        echo "============================================"
        exit 0
    fi
fi

# ============================================================
#  Phase 1: Full Ablation (50K steps × 8 runs)
# ============================================================
if [ "$FILTER" = "all" ] || [ "$FILTER" = "full" ]; then
    echo ""
    echo "########################################"
    echo "#  Phase 1: Full Ablation (50K steps)  #"
    echo "########################################"

    FULL="$COMMON --max-steps 50000 --eval-interval 500"

    # --- Run 1: Vanilla baseline ---
    run "1/8 Vanilla" \
        python train_shakespeare_vanilla.py $FULL \
        --checkpoint-path checkpoints/vanilla_wiki_bpe.pt \
        --run-name exp1_vanilla

    # --- Run 2: NGT default (full features) ---
    run "2/8 NGT default" \
        python train_shakespeare.py $FULL \
        --checkpoint-path checkpoints/ngt_wiki_bpe_default.pt \
        --run-name exp2_ngt_default

    # --- Run 3: NGT bare gravity (no repulsion, no radius cutoff) ---
    run "3/8 NGT bare gravity" \
        python train_shakespeare.py $FULL \
        --no-repulsion --no-radius-cutoff \
        --checkpoint-path checkpoints/ngt_wiki_bpe_bare.pt \
        --run-name exp3_ngt_bare

    # --- Run 4: NGT no repulsion (radius cutoff only) ---
    run "4/8 NGT no-repulsion" \
        python train_shakespeare.py $FULL \
        --no-repulsion \
        --checkpoint-path checkpoints/ngt_wiki_bpe_no_repulsion.pt \
        --run-name exp4_ngt_no_repulsion

    # --- Run 5: NGT no radius cutoff (repulsion only) ---
    run "5/8 NGT no-radius-cutoff" \
        python train_shakespeare.py $FULL \
        --no-radius-cutoff \
        --checkpoint-path checkpoints/ngt_wiki_bpe_no_radius.pt \
        --run-name exp5_ngt_no_radius

    # --- Run 6: NGT mass-in-value ---
    run "6/8 NGT mass-in-value" \
        python train_shakespeare.py $FULL \
        --mass-in-value \
        --checkpoint-path checkpoints/ngt_wiki_bpe_mass_val.pt \
        --run-name exp6_ngt_mass_val

    # --- Run 7: NGT soft-cutoff ---
    run "7/8 NGT soft-cutoff" \
        python train_shakespeare.py $FULL \
        --use-soft-cutoff \
        --checkpoint-path checkpoints/ngt_wiki_bpe_soft_cutoff.pt \
        --run-name exp7_ngt_soft_cutoff

    # --- Run 8: Best combination (placeholder: adjust after Phase 1 results) ---
    # Default: bare gravity + soft cutoff (example combo; update based on results)
    run "8/8 NGT best combo" \
        python train_shakespeare.py $FULL \
        --no-repulsion --use-soft-cutoff \
        --checkpoint-path checkpoints/ngt_wiki_bpe_best_combo.pt \
        --run-name exp8_ngt_best_combo
fi

# ============================================================
#  Generate Report
# ============================================================
echo ""
echo "Generating ablation report..."
python generate_report.py checkpoints/*_best.pt --output results/report.md 2>/dev/null || \
    echo "(report generation failed or no checkpoints found)"

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
echo "Report: results/report.md"
echo "TensorBoard: http://<your-pod-ip>:6006"
echo ""
echo "To download results:"
echo "  tar czf results_bundle.tar.gz runs/ checkpoints/ results/ results.csv"
echo ""
