#!/usr/bin/env bash
# Evaluate a balanced-PINN checkpoint against the Re=10 reference.
#
# Usage:
#   bash scripts/bench_balanced.sh                          # exp-name=grad_norm
#   bash scripts/bench_balanced.sh --exp-name none
#   bash scripts/bench_balanced.sh --exp-name grad_norm --checkpoint path/to.pt
#   bash scripts/bench_balanced.sh --device cpu
#
# Outputs (results/phase4a_loss_balancing/<exp_name>/benchmarks/latest_lbfgs/):
#   snapshot_metrics.csv, l2_vs_time.png, field_comparison_t*.png, probe_pressures.png
#
# Go/No-go (Re=10 developed flow t≥0.1s): Mean L2_u ≤ 10% AND Mean L2_v ≤ 10%.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXP_NAME="grad_norm"

# Parse --exp-name early; remaining args forwarded to bench_unsteady.py.
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)
            EXP_NAME="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== Evaluating balanced PINN checkpoint (exp: $EXP_NAME) ==="
RUN_ROOT="results/phase4a_loss_balancing/$EXP_NAME"
CHECKPOINT="$RUN_ROOT/checkpoints/latest_lbfgs.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
    CHECKPOINT="$RUN_ROOT/checkpoints/best.pt"
fi

python3 -u src/bench_unsteady.py \
    --exp-name "$EXP_NAME" \
    --checkpoint "$CHECKPOINT" \
    --reference data/reference/unsteady_reference.mat \
    --output-dir "$RUN_ROOT/benchmarks/latest_lbfgs" \
    --tmax 0.5 \
    --t-developed 0.1 \
    --field-times 0.3 0.4 0.5 \
    --diagnostic-times 0.1 0.3 0.5 \
    "${EXTRA_ARGS[@]}"
