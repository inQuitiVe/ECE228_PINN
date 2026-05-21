# Steady Results

This folder stores artifacts from the steady PyTorch PINN run.

## Contents

- `checkpoints/steady_new.pt`
  - Latest checkpoint with model, optimizer, scheduler, history, and resume metadata.
- `checkpoints/steady_new_best.pt`
  - Best checkpoint by total loss.
- `logs/steady_new_loss.pkl`
  - Serialized loss history for plotting and analysis.
- `figures/steady_new_uvp.png`
  - Velocity and pressure field comparison.
- `figures/steady_new_iter_loss_full.png`
  - Full `iteration` vs `total loss` curve from checkpoint history.

## Resume

Run from repo root:

```bash
python3 scripts/train_steady.py \
  --device mps \
  --adam-iters 30000 \
  --load-checkpoint results/steady/checkpoints/steady_new.pt \
  --resume \
  --save-every 200 \
  --save-best \
  --checkpoint results/steady/checkpoints/steady_new.pt \
  --best-checkpoint results/steady/checkpoints/steady_new_best.pt \
  --loss-history results/steady/logs/steady_new_loss.pkl \
  --output-figure results/steady/figures/steady_new_uvp.png
```

`--adam-iters` is the target final iteration. Increase it to continue beyond an already completed run.
