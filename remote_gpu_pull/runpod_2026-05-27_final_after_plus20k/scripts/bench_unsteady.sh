#!/usr/bin/env bash
# Evaluate an unsteady PINN checkpoint against the Re=10 reference CFD.
#
# Usage:
#   bash scripts/bench_unsteady.sh                              # evaluate latest_lbfgs.pt
#   bash scripts/bench_unsteady.sh --checkpoint <path/to.pt>   # custom checkpoint
#   bash scripts/bench_unsteady.sh --device cpu                # force CPU
#
# Outputs (results/unsteady/bench/):
#   l2_vs_time.png               — per-snapshot L2 error plot
#   field_comparison_t*.png      — reference vs PINN at t=0.3/0.4/0.5 s
#   probe_pressures.png          — probe pressure histories (paper Fig 8)
#
# Go/No-go: Mean L2_u ≤ 10% AND Mean L2_v ≤ 10% (developed flow, t ≥ 0.1 s)

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Default to latest_lbfgs.pt if it exists, otherwise best.pt
DEFAULT_CKPT="results/unsteady/checkpoints/latest_lbfgs.pt"
if [ ! -f "$DEFAULT_CKPT" ]; then
    DEFAULT_CKPT="results/unsteady/checkpoints/best.pt"
fi

mkdir -p results/unsteady/bench

echo "=== Evaluating unsteady PINN checkpoint ==="

python3 -u src/pinn_laminar_flow/bench_unsteady.py \
    --checkpoint "$DEFAULT_CKPT" \
    --reference  data/reference/unsteady_reference.mat \
    --output-dir results/unsteady/bench \
    "$@"
