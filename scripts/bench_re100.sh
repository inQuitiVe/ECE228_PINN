#!/usr/bin/env bash
# Evaluate a Re=100 PINN checkpoint against the Re=100 reference (t ∈ [0, 2.0] s).
#
# Usage:
#   bash scripts/bench_re100.sh                              # re100_vanilla experiment
#   bash scripts/bench_re100.sh --exp-name re100_vanilla
#   bash scripts/bench_re100.sh --checkpoint path/to.pt      # custom checkpoint
#   bash scripts/bench_re100.sh --device cpu
#
# Outputs (results/phase3_re100_stress/<exp_name>/benchmarks/latest_lbfgs/):
#   snapshot_metrics.csv         — per-snapshot L2_u, L2_v, L2_p
#   l2_vs_time.png               — per-snapshot L2 error plot
#   field_comparison_t*.png      — reference vs PINN at t=1.0/1.5/2.0 s
#   probe_pressures.png          — probe pressure histories
#
# Go/No-go: Mean L2_u ≤ 10% AND Mean L2_v ≤ 10% (developed flow, t ≥ 0.5 s)

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXP_NAME="re100_vanilla"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)
            EXP_NAME="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done
RUN_ROOT="results/phase3_re100_stress/$EXP_NAME"
CHECKPOINT="$RUN_ROOT/checkpoints/latest_lbfgs.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
    CHECKPOINT="$RUN_ROOT/checkpoints/best.pt"
fi

echo "=== Evaluating Re=100 PINN checkpoint (exp: $EXP_NAME) ==="

python3 -u src/bench_unsteady.py \
    --exp-name "$EXP_NAME" \
    --checkpoint "$CHECKPOINT" \
    --reference data/reference/unsteady_reference_re100_t2.mat \
    --output-dir "$RUN_ROOT/benchmarks/latest_lbfgs" \
    --tmax 2.0 \
    --t-developed 0.5 \
    --field-times 1.0 1.5 2.0 \
    --diagnostic-times 0.5 1.0 1.5 2.0 \
    "${EXTRA_ARGS[@]}"
