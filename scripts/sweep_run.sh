#!/bin/bash
# Generic sweep driver: runs only the variant names passed as arguments,
# in parallel, each writing its own result.csv on completion.
# Usage: bash sweep_run.sh nseed_1 nseed_2 acceptq_090 ...
set -u
BASE=/root/sweep
REF=$BASE/FluentReferenceMu002/FluentSol.mat
export TF_FORCE_GPU_ALLOW_GROWTH=true
export SWEEP_LBFGS_MAXITER=${SWEEP_LBFGS_MAXITER:-1500}

params_for () {
  case "$1" in
    baseline)       echo "5 0.95 1000";;
    nseed_1)        echo "1 0.95 1000";;
    nseed_2)        echo "2 0.95 1000";;
    nseed_10)       echo "10 0.95 1000";;
    acceptq_090)    echo "5 0.90 1000";;
    acceptq_097)    echo "5 0.97 1000";;
    acceptq_099)    echo "5 0.99 1000";;
    nexplore_500)   echo "5 0.95 500";;
    nexplore_2000)  echo "5 0.95 2000";;
    nexplore_5000)  echo "5 0.95 5000";;
    nexplore_10000) echo "5 0.95 10000";;
    *) echo "ERR";;
  esac
}

run_variant () {
  name=$1; read nseed acceptq nexplore <<< "$(params_for "$name")"
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

for v in "$@"; do run_variant "$v" & sleep 4; done
wait
echo ALL_DONE
