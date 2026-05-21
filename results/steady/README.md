# Steady Results

This folder stores artifacts from the steady PyTorch PINN run.

## Contents

- `checkpoints/latest.pt`
  - Latest checkpoint with model, optimizer, scheduler, history, and resume metadata.
- `checkpoints/best.pt`
  - Best checkpoint by total loss.
- `logs/loss_history.pkl`
  - Serialized loss history for plotting and analysis.
- `figures/field_comparison.png`
  - Velocity and pressure field comparison.
- `figures/loss_curve.png`
  - Full `iteration` vs `total loss` curve from checkpoint history.

## Resume

Run from repo root:

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

`--adam-iters` is the target final iteration. Increase it to continue beyond an already completed run.
