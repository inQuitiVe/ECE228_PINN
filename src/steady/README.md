# Steady-flow PINN experiments (TensorFlow 2)

Adaptive-collocation study for the **steady** mixed-form PINN (flow past a
cylinder, `Re` from the original paper). This is separate from the PyTorch
*unsteady* code in `../` — it is the TF2 line of work used for the steady
residual-based-resampling ablation reported alongside the project.

| File | What it is |
|------|------------|
| `steady_org.py` | Straight TF2 port of the original steady baseline. |
| `Steady_ref.py` | Reference/baseline run (headless `Agg` backend). |
| `Steady_adpative.py` | Adaptive residual-based collocation resampling. Tunable via env vars `SWEEP_NSEED`, `SWEEP_ACCEPTQ`, `SWEEP_NEXPLORE`, `SWEEP_LBFGS_MAXITER`. |
| `Steady_loss_anneal.py` | Learning-rate-annealing loss-weighting variant. |
| `eval_l2.py` | Loads checkpointed weights and reports relative L2 error vs the Fluent reference. Imports the modules above by name. |

## Reference data

These scripts read the Ansys Fluent steady reference as
`FluentReferenceMu002/FluentSol.mat` **relative to the current working
directory** (the layout used on the GPU pod). The reference `.mat` is tracked
in this repo at:

```
results/reference/pinn_laminar_flow_original/fluent_reference/FluentSol.mat
```

To run a script, symlink that directory into the CWD first, e.g. from the repo root:

```bash
ln -s results/reference/pinn_laminar_flow_original/fluent_reference FluentReferenceMu002
PYTHONPATH=src/steady python3 src/steady/eval_l2.py Steady_ref <weights.pickle>
```

## Parameter sweep + plots

The adaptive-collocation sweep over `Steady_adpative.py` is driven by the
shell scripts in [`../../scripts/`](../../scripts):

- `sweep_driver.sh` — sequential one-parameter-at-a-time sweep,
- `sweep_parallel.sh` — all variants concurrently on one GPU,
- `sweep_run.sh` — runs only the named variants.

Lightweight evidence (per-variant `*.csv`, `summary.csv`, and run logs) is
archived under
[`../../results/experiments/steady_adaptive_sweep/`](../../results/experiments/steady_adaptive_sweep).
Heavy artifacts (loss/weight pickles, figures) are **not** tracked — regenerate
the figures with `scripts/plot_sweep.py` and `scripts/plot_loss_curve.py` after
placing the run pickles under
`results/experiments/steady_adaptive_sweep/logs/`.
