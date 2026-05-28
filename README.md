# PINN Laminar Flow

PyTorch implementation of physics-informed neural networks (PINNs) for incompressible laminar flow around a cylinder. The project includes steady and transient training flows, checkpoint/resume support, and saved experiment artifacts.

## Reference

This repository follows the mixed-form PINN setup from:

[Chengping Rao, Hao Sun and Yang Liu. Physics-informed deep learning for incompressible laminar flows.](https://arxiv.org/abs/2002.10558)

## Repository Layout

- `src/pinn_laminar_flow/`
  - PyTorch source code for steady/transient PINN models and plotting utilities.
- `train.py`
  - Root training entrypoint for steady and transient experiments.
- `data/reference/`
  - Reference CFD data used for steady-flow comparison.
- `results/`
  - Checkpoints, figures, logs, and archived outputs from experiments.

## Setup

```bash
pip install -r requirements.txt
```

Device selection is supported with `--device {auto,cpu,cuda,mps}`. Use `--device mps` on Apple Silicon, `--device cuda` on NVIDIA GPUs, and `--device cpu` for CPU-only runs.

## Training

Run the Re=10 baseline reproduction:

```bash
bash scripts/train_unsteady.sh --device mps
```

Resume the baseline L-BFGS stage:

```bash
bash scripts/resume_unsteady.sh --device mps
```

Run the Re=10 benchmark:

```bash
bash scripts/bench_unsteady.sh --device mps
```

Run the loss-balancer experiments:

```bash
bash scripts/train_balanced.sh --balancer none --exp-name fixed_beta_none
bash scripts/train_balanced.sh --balancer strict_grad_norm --exp-name strict_grad_norm_test
```

## Checkpoints

Both training flows save PyTorch `.pt` checkpoints with:

- model state
- optimizer state
- scheduler state
- loss history
- resume metadata (`iteration`, `best_loss`, `stale_steps`)

Useful flags:

- `--resume` continues from the checkpoint iteration.
- `--save-every N` writes a checkpoint every `N` Adam iterations.
- `--save-best` writes the best model by total loss.

## Results

Artifacts are organized by project phase:

- `results/phase1_reproduction/vanilla_pytorch/` — original PyTorch Re=10 baseline.
- `results/phase1_reproduction/strict_reproduce_scipy/` — best SciPy L-BFGS-B reproduction baseline.
- `results/phase3_re100_reference/` — generated Re=100 reference-data logs.
- `results/phase4a_loss_balancing/` — fixed-beta, GradNorm, strict GradNorm, and smoke-test runs.
- `results/reference/` — copied reference papers, analysis reports, and original-paper output figures.

Each experiment directory uses the same internal layout when applicable:

- `checkpoints/`
- `logs/`
- `figures/`
- `benchmarks/`
