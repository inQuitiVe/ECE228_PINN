#!/bin/bash
# Sequential parameter sweep driver for Steady_adpative.py
# One-parameter-at-a-time off the baseline (nseed=5, acceptq=0.95, nexplore=1000).
# Each variant runs in its own subdir; final loss + L2 errors collected to summary.csv.
set -u

BASE=/root/sweep
REF=$BASE/FluentReferenceMu002/FluentSol.mat
SUMMARY=$BASE/summary.csv
export TF_FORCE_GPU_ALLOW_GROWTH=true
export SWEEP_LBFGS_MAXITER=4000   # capped L-BFGS budget so the 10-run sweep finishes

echo "variant,nseed,acceptq,nexplore,loss_after_lbfgs,err_u,err_v,err_p" > "$SUMMARY"

run_variant () {
  name=$1; nseed=$2; acceptq=$3; nexplore=$4
  d=$BASE/$name
  mkdir -p "$d/FluentReferenceMu002"
  ln -sf "$REF" "$d/FluentReferenceMu002/FluentSol.mat"
  cp "$BASE/Steady_adpative.py" "$d/"
  echo "[$(date +%H:%M:%S)] START $name (nseed=$nseed acceptq=$acceptq nexplore=$nexplore)"
  ( cd "$d" && SWEEP_NSEED=$nseed SWEEP_ACCEPTQ=$acceptq SWEEP_NEXPLORE=$nexplore \
      python3 -u Steady_adpative.py > "$d/run.log" 2>&1 )
  loss=$(grep "Loss after  L-BFGS" "$d/run.log" | tail -1 | awk '{print $NF}')
  eu=$(grep -E "^ +u +:" "$d/run.log" | tail -1 | awk '{print $3}')
  ev=$(grep -E "^ +v +:" "$d/run.log" | tail -1 | awk '{print $3}')
  ep=$(grep -E "^ +p +:" "$d/run.log" | tail -1 | awk '{print $3}')
  echo "$name,$nseed,$acceptq,$nexplore,${loss:-NA},${eu:-NA},${ev:-NA},${ep:-NA}" >> "$SUMMARY"
  echo "[$(date +%H:%M:%S)] DONE  $name -> loss=${loss:-NA} eu=${eu:-NA} ev=${ev:-NA} ep=${ep:-NA}"
}

# --- n_seed_neighbors (acceptq=0.95, nexplore=1000) ---
run_variant nseed_1  1  0.95 1000
run_variant nseed_2  2  0.95 1000
run_variant nseed_10 10 0.95 1000
# --- accept_quantile (nseed=5, nexplore=1000) ---
run_variant acceptq_090 5 0.90 1000
run_variant acceptq_097 5 0.97 1000
run_variant acceptq_099 5 0.99 1000
# --- n_exploration_pts (nseed=5, acceptq=0.95) ---
run_variant nexplore_500   5 0.95 500
run_variant nexplore_2000  5 0.95 2000
run_variant nexplore_5000  5 0.95 5000
run_variant nexplore_10000 5 0.95 10000

echo "ALL_DONE" >> "$SUMMARY"
echo "[$(date +%H:%M:%S)] SWEEP COMPLETE"
