#!/usr/bin/env bash
# Train vanilla 7×50 PINN at Re=100 (μ=0.0005, tmax=2.0s, period=1.0s).
# Phase 3A stress-test baseline — expected to fail (smeared Kármán wake).
#
# Usage:
#   bash scripts/train_re100_vanilla.sh                    # default
#   bash scripts/train_re100_vanilla.sh --device cuda      # force CUDA
#   bash scripts/train_re100_vanilla.sh --exp-name re100_vanilla

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXP_NAME="re100_vanilla"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done
RUN_ROOT="results/phase3_re100_stress/$EXP_NAME"

echo "=== Starting Re=100 vanilla PINN training (exp: $EXP_NAME) ==="
echo "    Adam 10k (lr=5e-4) + L-BFGS 100k function evals"
echo "    Re=100: mu=0.0005, tmax=2.0s, period=1.0s"
echo "    β: wall=5 inlet=5 outlet=1 ic=1"
echo ""

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/logs" "$RUN_ROOT/figures"

python3 -u src/unsteady.py \
    --exp-name "$EXP_NAME" \
    --adam-iters 10000 \
    --learning-rate 5e-4 \
    --lbfgs-iters 100000 \
    --lbfgs-save-every 500 \
    --save-best \
    --mu 0.0005 \
    --tmax 2.0 \
    --period 1.0 \
    --checkpoint "$RUN_ROOT/checkpoints/latest.pt" \
    --best-checkpoint "$RUN_ROOT/checkpoints/best.pt" \
    --lbfgs-checkpoint "$RUN_ROOT/checkpoints/latest_lbfgs.pt" \
    --loss-history "$RUN_ROOT/logs/loss_history.pkl" \
    --output-dir "$RUN_ROOT/figures" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$RUN_ROOT/logs/train.log"

echo ""
echo "=== Training complete. Run bench: bash scripts/bench_re100.sh ==="
