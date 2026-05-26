#!/usr/bin/env bash
# Full unsteady PINN training: Adam 10k → L-BFGS 50k.
# Paper-faithful settings (Rao 2020, arXiv:2002.10558):
#   β = 5/5/1/1,  IC = u+v+p = 0,  N_g = 100k LHS,  Re = 10 (μ = 0.005)
#
# Usage:
#   bash scripts/train_unsteady.sh                  # auto device, fresh start
#   bash scripts/train_unsteady.sh --device cuda    # force CUDA
#   bash scripts/train_unsteady.sh --device mps     # force Apple MPS
#   bash scripts/train_unsteady.sh --device cpu     # CPU (slow, for testing)
#
# Outputs (all under results/unsteady/):
#   checkpoints/best.pt          — Adam best
#   checkpoints/latest.pt        — Adam final
#   checkpoints/latest_lbfgs.pt  — L-BFGS checkpoint (every 500 steps)
#   logs/train.log               — training stdout
#   logs/loss_history.pkl        — full loss history

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p results/unsteady/checkpoints results/unsteady/logs results/unsteady/figures

echo "=== Starting unsteady PINN training ==="
echo "    Adam 10k (lr=5e-4) + L-BFGS 50k"
echo "    β: wall=5 inlet=5 outlet=1 ic=1"
echo "    IC: u=v=p=0"
echo "    N_g: 100k LHS collocation points"
echo ""

python3 -u src/pinn_laminar_flow/unsteady.py \
    --adam-iters 10000 \
    --learning-rate 5e-4 \
    --lbfgs-iters 50000 \
    --lbfgs-save-every 500 \
    --save-best \
    --checkpoint      results/unsteady/checkpoints/latest.pt \
    --best-checkpoint results/unsteady/checkpoints/best.pt \
    --lbfgs-checkpoint results/unsteady/checkpoints/latest_lbfgs.pt \
    --loss-history    results/unsteady/logs/loss_history.pkl \
    --output-dir      results/unsteady/figures \
    "$@" \
    2>&1 | tee results/unsteady/logs/train.log

echo ""
echo "=== Training complete. Run bench: bash scripts/bench_unsteady.sh ==="
