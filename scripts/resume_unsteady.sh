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

EXP_NAME="vanilla_pytorch"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)
            EXP_NAME="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done
RUN_ROOT="results/phase1_reproduction/$EXP_NAME"

echo "=== Resuming unsteady PINN L-BFGS (exp: $EXP_NAME) ==="
echo "    Loading: $RUN_ROOT/checkpoints/latest_lbfgs.pt"
echo "    Additional L-BFGS function evals: 50k (default)"
echo ""

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/logs" "$RUN_ROOT/figures"

python3 -u src/unsteady.py \
    --exp-name "$EXP_NAME" \
    --adam-iters 0 \
    --lbfgs-iters 50000 \
    --lbfgs-save-every 500 \
    --save-best \
    --load-checkpoint "$RUN_ROOT/checkpoints/latest_lbfgs.pt" \
    --checkpoint "$RUN_ROOT/checkpoints/latest.pt" \
    --best-checkpoint "$RUN_ROOT/checkpoints/best.pt" \
    --lbfgs-checkpoint "$RUN_ROOT/checkpoints/latest_lbfgs.pt" \
    --loss-history "$RUN_ROOT/logs/loss_history.pkl" \
    --output-dir "$RUN_ROOT/figures" \
    --resume \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$RUN_ROOT/logs/train_resume.log"

echo ""
echo "=== Resume complete. Run bench: bash scripts/bench_unsteady.sh ==="
