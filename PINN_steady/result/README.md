# Steady Training Results

This folder stores artifacts from the steady PINN PyTorch run.

## Files

- `steady_new.pt`
  - Latest checkpoint (model + optimizer/scheduler state + history).
  - Use this to continue training from the latest saved step.
- `steady_new_best.pt`
  - Best checkpoint by total loss seen so far.
  - Use this for evaluation/inference when you want the lowest-loss model.
- `steady_new_loss.pkl`
  - Serialized loss history list.
  - Use this to plot full training curves (`loss`, `loss_f`, `loss_wall`, `loss_inlet`, `loss_outlet`).
- `steady_new_uvp.png`
  - Post-process visualization (predicted fields).

## Resume Training

Run from repo root:

```bash
python3 -u PINN_steady/SteadyFlowCylinder_mixed.py \
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

Notes:
- `--resume` continues from the checkpoint iteration.
- `--adam-iters` is the target final iteration for this run.
- If `--adam-iters` is already reached, increase it (for example `40000`) to continue.
