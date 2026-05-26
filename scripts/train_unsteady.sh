#!/usr/bin/env bash
# Full unsteady PINN training: Adam 10k → L-BFGS 100k function evaluations.
# Paper-faithful settings (Rao 2020, arXiv:2002.10558):
#   β = 5/5/1/1,  IC = u+v+p = 0,  N_g = 100k LHS,  Re = 10 (μ = 0.005)
#
# NOTE on --lbfgs-iters: the counter printed as "LBFGS N" and the max_eval
# limit both count *function evaluations* (closure calls), not true L-BFGS
# quasi-Newton iterations. Strong-Wolfe line search uses ~20 evals per iter,
# so 100k evals ≈ 5k real iterations. The original paper uses maxfun=100000
# in scipy L-BFGS-B (same unit), so --lbfgs-iters 100000 is paper-faithful.
#
# Usage:
#   bash scripts/train_unsteady.sh                         # vanilla experiment
#   bash scripts/train_unsteady.sh --exp-name my_run       # custom name
#   bash scripts/train_unsteady.sh --device cuda           # force CUDA
#
# Outputs (all under results/{exp_name}/):
#   checkpoints/{exp_name}/best.pt          — Adam best
#   checkpoints/{exp_name}/latest.pt        — Adam final
#   checkpoints/{exp_name}/latest_lbfgs.pt  — L-BFGS checkpoint (every 500 steps)
#   logs/{exp_name}/loss_history.pkl        — full loss history
#   figures/{exp_name}/                     — visualisations

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXP_NAME="vanilla"

echo "=== Starting unsteady PINN training (exp: $EXP_NAME) ==="
echo "    Adam 10k (lr=5e-4) + L-BFGS 100k function evals"
echo "    β: wall=5 inlet=5 outlet=1 ic=1"
echo "    IC: u=v=p=0"
echo "    N_g: 100k LHS collocation points"
echo ""

mkdir -p "results/checkpoints/$EXP_NAME" "results/logs/$EXP_NAME" "results/figures/$EXP_NAME"

python3 -u src/unsteady.py \
    --exp-name "$EXP_NAME" \
    --adam-iters 10000 \
    --learning-rate 5e-4 \
    --lbfgs-iters 100000 \
    --lbfgs-save-every 500 \
    --save-best \
    "$@" \
    2>&1 | tee "results/logs/$EXP_NAME/train.log"

echo ""
echo "=== Training complete. Run bench: bash scripts/bench_unsteady.sh ==="
