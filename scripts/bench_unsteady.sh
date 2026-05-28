#!/usr/bin/env bash
# Evaluate an unsteady PINN checkpoint against the Re=10 reference CFD.
#
# Usage:
#   bash scripts/bench_unsteady.sh                              # vanilla experiment
#   bash scripts/bench_unsteady.sh --exp-name vanilla           # explicit name
#   bash scripts/bench_unsteady.sh --checkpoint path/to.pt     # custom checkpoint
#   bash scripts/bench_unsteady.sh --device cpu                 # force CPU
#
# Outputs (results/phase1_reproduction/<exp_name>/benchmarks/latest_lbfgs/):
#   snapshot_metrics.csv         — per-snapshot L2_u, L2_v, L2_p
#   l2_vs_time.png               — per-snapshot L2 error plot
#   field_comparison_t*.png      — reference vs PINN at t=0.3/0.4/0.5 s
#   probe_pressures.png          — probe pressure histories (paper Fig 8)
#
# Go/No-go: Mean L2_u ≤ 10% AND Mean L2_v ≤ 10% (developed flow, t ≥ 0.1 s)

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
CHECKPOINT="$RUN_ROOT/checkpoints/latest_lbfgs.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
    CHECKPOINT="$RUN_ROOT/checkpoints/best.pt"
fi

echo "=== Evaluating unsteady PINN checkpoint (exp: $EXP_NAME) ==="

python3 -u src/bench_unsteady.py \
    --exp-name "$EXP_NAME" \
    --checkpoint "$CHECKPOINT" \
    --reference data/reference/unsteady_reference.mat \
    --output-dir "$RUN_ROOT/benchmarks/latest_lbfgs" \
    "${EXTRA_ARGS[@]}"
