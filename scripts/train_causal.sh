#!/usr/bin/env bash
# Train the causal PINN at Re=10 with the recommended scheme:
#   higher LR (1e-3) + plateau scheduler + early-stop (true-loss, gated on min_w>0.9)
#   + frozen-causal-weight L-BFGS polish.
#
# Usage:
#   bash scripts/train_causal.sh --causal-eps 30
#   bash scripts/train_causal.sh --causal-eps 30 --exp-name causal_eps30 --device cuda
#   bash scripts/train_causal.sh --causal-eps 100 --adam-iters 80000   # override anything
#
# exp-name defaults to causal_eps<EPS>. Any extra flags pass through to the python entrypoint.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EPS=30
EXP_NAME=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --causal-eps) EPS="$2"; shift 2 ;;
        --exp-name)   EXP_NAME="$2"; shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done
if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="causal_eps${EPS}"
fi

echo "=== Causal training (exp: $EXP_NAME, eps: $EPS) ==="
echo "    LR 1e-3 + plateau scheduler + early-stop(true-loss, min_w>0.9) + frozen L-BFGS"
echo ""

mkdir -p "results/checkpoints/$EXP_NAME" "results/logs/$EXP_NAME" "results/figures/$EXP_NAME"

python3 -u src/unsteady_causal.py \
    --exp-name "$EXP_NAME" \
    --causal-eps "$EPS" --causal-bins 32 --causal-bin-mode duration \
    --causal-lbfgs frozen \
    --learning-rate 1e-3 \
    --adam-iters 50000 \
    --scheduler plateau --scheduler-gamma 0.5 --scheduler-plateau-patience 500 --scheduler-min-lr 1e-6 \
    --early-stop-patience 2000 --early-stop-min-delta 0 --early-stop-warmup 5000 \
    --causality-delta 0.9 \
    --lbfgs-iters 50000 --lbfgs-save-every 1000 \
    --snapshot-iters 10000,20000,30000 \
    --save-best --device cuda \
    --mu 0.005 --tmax 0.5 --period 1.0 \
    --print-every 100 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "results/logs/$EXP_NAME/train.log"

echo ""
echo "=== Done. Bench:"
echo "    python3 src/bench_unsteady.py --exp-name $EXP_NAME \\"
echo "        --checkpoint results/checkpoints/$EXP_NAME/latest_lbfgs.pt \\"
echo "        --reference data/reference/unsteady_reference.mat --tmax 0.5 --t-developed 0.1 ==="
