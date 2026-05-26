#!/usr/bin/env bash
# Resume unsteady L-BFGS training from an experiment's latest_lbfgs.pt checkpoint.
#
# Usage:
#   bash scripts/resume_unsteady.sh                              # vanilla, default 50k more evals
#   bash scripts/resume_unsteady.sh --exp-name vanilla           # explicit experiment name
#   bash scripts/resume_unsteady.sh --lbfgs-iters 30000         # custom budget
#   bash scripts/resume_unsteady.sh --device cuda                # force CUDA

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXP_NAME="vanilla"

echo "=== Resuming unsteady PINN L-BFGS (exp: $EXP_NAME) ==="
echo "    Loading: results/checkpoints/$EXP_NAME/latest_lbfgs.pt"
echo "    Additional L-BFGS function evals: 50k (default)"
echo ""

mkdir -p "results/checkpoints/$EXP_NAME" "results/logs/$EXP_NAME" "results/figures/$EXP_NAME"

python3 -u src/unsteady.py \
    --exp-name "$EXP_NAME" \
    --adam-iters 0 \
    --lbfgs-iters 50000 \
    --lbfgs-save-every 500 \
    --save-best \
    --load-checkpoint "results/checkpoints/$EXP_NAME/latest_lbfgs.pt" \
    --resume \
    "$@" \
    2>&1 | tee "results/logs/$EXP_NAME/train_resume.log"

echo ""
echo "=== Resume complete. Run bench: bash scripts/bench_unsteady.sh ==="
