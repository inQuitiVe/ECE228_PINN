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

Run steady training:

```bash
python3 train.py steady --device mps
```

Run transient training:

```bash
python3 train.py unsteady --device mps
```

Resume steady training:

```bash
python3 train.py steady \
  --device mps \
  --adam-iters 30000 \
  --load-checkpoint results/steady/checkpoints/latest.pt \
  --resume \
  --save-every 200 \
  --save-best \
  --checkpoint results/steady/checkpoints/latest.pt \
  --best-checkpoint results/steady/checkpoints/best.pt \
  --loss-history results/steady/logs/loss_history.pkl \
  --output-figure results/steady/figures/field_comparison.png
```

Resume transient training:

```bash
python3 train.py unsteady \
  --device mps \
  --adam-iters 10000 \
  --load-checkpoint results/unsteady/checkpoints/latest.pt \
  --resume \
  --save-every 200 \
  --save-best
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

## Plotting

Plot steady total loss from a checkpoint:

```bash
PYTHONPATH=src python3 -m pinn_laminar_flow.plotting \
  --input results/steady/checkpoints/latest.pt \
  --output results/steady/figures/loss_curve.png
```

## Results

Steady artifacts:

- `results/steady/checkpoints/`
- `results/steady/figures/`
- `results/steady/logs/`

Transient artifacts:

- `results/unsteady/checkpoints/`
- `results/unsteady/figures/`
- `results/unsteady/logs/`

Representative steady outputs:

![](results/steady/figures/field_comparison.png)

![](results/steady/figures/loss_curve.png)
