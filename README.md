# ECE228 PINN

Tracked PyTorch PINN code, reference data, experiment scripts, reports, and archived results for incompressible laminar flow around a cylinder.

## Setup

```bash
pip install -r requirements.txt
```

Or use the tracked setup script:

```bash
bash scripts/setup_env.sh
```

Use `--device auto`, `--device cpu`, `--device cuda`, or `--device mps` on scripts that forward device selection to Python.

## Tracked Entry Points

### Training

- `scripts/setup_env.sh`
  - Installs tracked Python dependencies and creates the `pyDOE` shim expected by the code.

- `scripts/train_unsteady.sh`
  - Re=10 vanilla baseline training.
  - Writes to `results/phase1_reproduction/<exp_name>/`.

```bash
bash scripts/train_unsteady.sh --device cuda
```

- `scripts/resume_unsteady.sh`
  - Resumes the Re=10 vanilla baseline from `latest_lbfgs.pt`.

```bash
bash scripts/resume_unsteady.sh --device cuda
```

- `scripts/train_balanced.sh`
  - Re=10 loss-balancer training with `none`, `grad_norm`, or `strict_grad_norm`.
  - Writes to `results/phase4a_loss_balancing/<exp_name>/`.

```bash
bash scripts/train_balanced.sh --balancer strict_grad_norm --exp-name strict_grad_norm_test --device cuda
```

- `scripts/train_re100_vanilla.sh`
  - Re=100 stress-test baseline.
  - Writes to `results/phase3_re100_stress/<exp_name>/`.

```bash
bash scripts/train_re100_vanilla.sh --device cuda
```

### Evaluation

- `scripts/bench_unsteady.sh`
  - Evaluates a Re=10 checkpoint against `data/reference/unsteady_reference.mat`.

```bash
bash scripts/bench_unsteady.sh --device cuda
```

- `scripts/bench_balanced.sh`
  - Evaluates a Phase 4A loss-balancer run against the Re=10 reference.

```bash
bash scripts/bench_balanced.sh --exp-name strict_grad_norm_test --device cuda
```

- `scripts/bench_re100.sh`
  - Evaluates a Re=100 checkpoint against `data/reference/unsteady_reference_re100_t2.mat`.

```bash
bash scripts/bench_re100.sh --device cuda
```

## Tracked Source Files

- `src/unsteady.py`
  - Main Re=10/Re=100 transient mixed-form PINN trainer.
- `src/unsteady_strict_reproduce.py`
  - Strict reproduction trainer with SciPy-style reproduction choices.
- `src/unsteady_balanced.py`
  - Loss-balancer trainer using `src/loss_balancers.py`.
- `src/loss_balancers.py`
  - `NoneBalancer`, `GradNormBalancer`, and `StrictGradNormBalancer`.
- `src/bench_unsteady.py`
  - Shared benchmark harness for Re=10 and Re=100 checkpoints.

## Tracked Data

- `data/reference/unsteady_reference.mat`
  - Re=10 CFD reference data.
- `data/reference/unsteady_reference_re100_t2.mat`
  - Re=100 CFD reference data.
- `data/reference/figures/`
  - Tracked reference animations and frame exports.
- `data/reference/*.py`
  - Scripts used to generate reference datasets and figures.

See `data/reference/README.md` for dataset details.

## Tracked Results

Results are phase-scoped under `results/`:

- `results/phase1_reproduction/`
  - Vanilla and strict reproduction baselines.
- `results/phase3_re100_reference/`
  - Re=100 reference-generation logs.
- `results/phase4a_loss_balancing/`
  - Fixed-beta, GradNorm, strict GradNorm, and smoke-test artifacts.
- `results/reference/`
  - Copied papers, reference reports, and original-paper artifacts from `ref/`.

See `results/README.md` for the directory convention.

## Tracked Docs

- `docs/plan.md`
  - Phase plan and method roadmap.
- `docs/todo.md`
  - Current project status and task list.
- `docs/user/current_status_and_postmortem.md`
  - Current reproduction status and failure analysis.
- `docs/user/initial_handover.pdf`
  - Team handover PDF.

## Notes

- `ref/`, `remote_gpu_pull/`, `tmp/`, and other untracked local working directories are not part of this README.
- For new ad hoc experiments, prefer `results/experiments/<exp_name>/` unless the run belongs to a named phase.
