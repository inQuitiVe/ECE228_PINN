#!/bin/bash
# Parallel fast sweep: all 10 variants concurrently on one A100 (allow_growth).
# Each variant writes its own run.log and, on completion, a result.csv line.
# Designed to be polled+copied out continuously so a pod death can't wipe results.
set -u
BASE=/root/sweep
REF=$BASE/FluentReferenceMu002/FluentSol.mat
export TF_FORCE_GPU_ALLOW_GROWTH=true
export SWEEP_LBFGS_MAXITER=1500   # reduced budget so all 10 finish fast

run_variant () {
  name=$1; nseed=$2; acceptq=$3; nexplore=$4
  d=$BASE/$name
  mkdir -p "$d/FluentReferenceMu002"
  ln -sf "$REF" "$d/FluentReferenceMu002/FluentSol.mat"
  cp "$BASE/Steady_adpative.py" "$d/"
  ( cd "$d" && SWEEP_NSEED=$nseed SWEEP_ACCEPTQ=$acceptq SWEEP_NEXPLORE=$nexplore \
      python3 -u Steady_adpative.py > "$d/run.log" 2>&1
    loss=$(grep "Loss after  L-BFGS" "$d/run.log" | tail -1 | awk '{print $NF}')
    eu=$(grep -E "^ +u +:" "$d/run.log" | tail -1 | awk '{print $3}')
    ev=$(grep -E "^ +v +:" "$d/run.log" | tail -1 | awk '{print $3}')
    ep=$(grep -E "^ +p +:" "$d/run.log" | tail -1 | awk '{print $3}')
    echo "$name,$nseed,$acceptq,$nexplore,${loss:-NA},${eu:-NA},${ev:-NA},${ep:-NA}" > "$d/result.csv" )
}

# Stagger launches by 4s so 10 TF inits don't spike GPU memory simultaneously.
run_variant nseed_1      1  0.95 1000   & sleep 4
run_variant nseed_2      2  0.95 1000   & sleep 4
run_variant nseed_10     10 0.95 1000   & sleep 4
run_variant acceptq_090  5  0.90 1000   & sleep 4
run_variant acceptq_097  5  0.97 1000   & sleep 4
run_variant acceptq_099  5  0.99 1000   & sleep 4
run_variant nexplore_500   5 0.95 500   & sleep 4
run_variant nexplore_2000  5 0.95 2000  & sleep 4
run_variant nexplore_5000  5 0.95 5000  & sleep 4
run_variant nexplore_10000 5 0.95 10000 &
wait
echo "ALL_DONE"
