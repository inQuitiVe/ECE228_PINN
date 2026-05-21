# PINN Laminar Flow

Physics-informed neural network experiments for incompressible laminar flow around a cylinder, with PyTorch training flows for both steady and transient cases.

## Reference

This repository follows the mixed-form PINN setup from:

[Chengping Rao, Hao Sun and Yang Liu. Physics-informed deep learning for incompressible laminar flows.](https://arxiv.org/abs/2002.10558)

## Repository Layout

- `data/reference/`
  - Reference CFD data used by the steady case.
- `scripts/`
  - Top-level entrypoints for training and plotting.
- `PINN_steady/`
  - Steady-flow implementation and steady experiment outputs.
- `PINN_unsteady/`
  - Transient-flow implementation and transient experiment outputs.

## Main Entry Points

- `scripts/train_steady.py`
  - Run or resume the steady PyTorch model.
- `scripts/train_unsteady.py`
  - Run or resume the transient PyTorch model.
- `scripts/plot_steady_loss.py`
  - Plot `iter` vs `total loss` from a checkpoint or loss-history pickle.

## Runtime

Install dependencies:

```bash
pip install -r requirements.txt
```

Run steady training:

```bash
python3 scripts/train_steady.py --device mps
```

Run transient training:

```bash
python3 scripts/train_unsteady.py --device mps
```

Resume steady training from a checkpoint:

```bash
python3 scripts/train_steady.py \
  --device mps \
  --adam-iters 30000 \
  --load-checkpoint PINN_steady/result/steady_new.pt \
  --resume \
  --save-every 200 \
  --save-best \
  --checkpoint PINN_steady/result/steady_new.pt \
  --best-checkpoint PINN_steady/result/steady_new_best.pt \
  --loss-history PINN_steady/result/steady_new_loss.pkl \
  --output-figure PINN_steady/result/steady_new_uvp.png
```

Resume transient training from a checkpoint:

```bash
python3 scripts/train_unsteady.py \
  --device mps \
  --adam-iters 10000 \
  --load-checkpoint PINN_unsteady/uvNN_torch.pt \
  --resume \
  --save-every 200 \
  --save-best
```

Plot a steady loss curve from a checkpoint or loss pickle:

```bash
python3 scripts/plot_steady_loss.py \
  --input PINN_steady/result/steady_new.pt \
  --output PINN_steady/result/steady_loss_curve.png
```

Device selection is supported through `--device {auto,cpu,cuda,mps}`.

Checkpoint behavior:

- Both steady and transient flows save model state, optimizer state, scheduler state, and loss history in `.pt` checkpoints.
- `--resume` continues from the stored iteration.
- `--save-every` writes periodic checkpoints.
- `--save-best` writes the current best model by total loss.

## Results

- Latest steady artifacts live in `PINN_steady/result/`.
- Transient frames and animations live under `PINN_unsteady/output/`.

Representative outputs:

![](PINN_steady/result/steady_new_uvp.png)

![](PINN_steady/result/steady_new_iter_loss_full.png)
