#!/usr/bin/env bash
# Resume unsteady L-BFGS training from latest_lbfgs.pt checkpoint.
#
# Use this to continue a run that was interrupted or to add more function
# evaluations on top of an existing checkpoint.
#
# NOTE on --lbfgs-iters: counts *function evaluations* (same unit as scipy's
# maxfun=100000 in the original paper). ~20 evals per true L-BFGS iteration.
#
# Current state (2026-05-26):
#   ~55k function evals done; target is 100k (paper-faithful).
#   Run with --lbfgs-iters 50000 to reach ~105k total.
#
# Usage:
#   bash scripts/resume_unsteady.sh                       # default 50k more evals
#   bash scripts/resume_unsteady.sh --lbfgs-iters 30000  # custom budget
#   bash scripts/resume_unsteady.sh --device cuda         # force CUDA

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p results/checkpoints results/logs results/figures

echo "=== Resuming unsteady PINN L-BFGS ==="
echo "    Loading: results/checkpoints/latest_lbfgs.pt"
echo "    Additional L-BFGS function evals: 50k (default)"
echo ""

python3 -u src/unsteady.py \
    --adam-iters 0 \
    --lbfgs-iters 50000 \
    --lbfgs-save-every 500 \
    --save-best \
    --load-checkpoint results/checkpoints/latest_lbfgs.pt \
    --resume \
    --checkpoint      results/checkpoints/latest.pt \
    --best-checkpoint results/checkpoints/best.pt \
    --lbfgs-checkpoint results/checkpoints/latest_lbfgs.pt \
    --loss-history    results/logs/loss_history.pkl \
    --output-dir      results/figures \
    "$@" \
    2>&1 | tee results/logs/train_resume.log

echo ""
echo "=== Resume complete. Run bench: bash scripts/bench_unsteady.sh ==="
