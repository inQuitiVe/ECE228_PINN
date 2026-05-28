#!/usr/bin/env bash
# Train PINN at Re=10 with a pluggable loss balancer.
#
# Usage:
#   bash scripts/train_balanced.sh                          # grad_norm (default)
#   bash scripts/train_balanced.sh --balancer none          # fixed β baseline
#   bash scripts/train_balanced.sh --balancer grad_norm     # Wang 2020 Alg.1
#   bash scripts/train_balanced.sh --device cuda
#   bash scripts/train_balanced.sh --exp-name my_run
#
# Defaults: Re=10 (mu=0.005), Adam 10k + L-BFGS 50k evals, grad_norm balancer.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Default balancer; override with --balancer {none,grad_norm}
BALANCER="grad_norm"

# Parse --balancer early to name the experiment; remaining args passed through.
EXP_NAME=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --balancer)
            BALANCER="$2"; shift 2 ;;
        --exp-name)
            EXP_NAME="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="$BALANCER"
fi
RUN_ROOT="results/phase4a_loss_balancing/$EXP_NAME"

echo "=== Starting balanced PINN training (exp: $EXP_NAME, balancer: $BALANCER) ==="
echo "    Adam 10k (lr=5e-4) + L-BFGS 50k evals"
echo "    Re=10: mu=0.005, tmax=0.5s, period=1.0s"
echo ""

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/logs" "$RUN_ROOT/figures"

python3 -u src/unsteady_balanced.py \
    --exp-name "$EXP_NAME" \
    --balancer "$BALANCER" \
    --adam-iters 10000 \
    --learning-rate 5e-4 \
    --lbfgs-iters 50000 \
    --lbfgs-save-every 500 \
    --save-best \
    --mu 0.005 \
    --tmax 0.5 \
    --period 1.0 \
    --checkpoint "$RUN_ROOT/checkpoints/latest.pt" \
    --best-checkpoint "$RUN_ROOT/checkpoints/best.pt" \
    --lbfgs-checkpoint "$RUN_ROOT/checkpoints/latest_lbfgs.pt" \
    --loss-history "$RUN_ROOT/logs/loss_history.pkl" \
    --output-dir "$RUN_ROOT/figures" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$RUN_ROOT/logs/train.log"

echo ""
echo "=== Training complete. Run bench: bash scripts/bench_balanced.sh --exp-name $EXP_NAME ==="
