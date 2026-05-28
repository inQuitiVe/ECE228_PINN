# Results Layout

Experiment artifacts are grouped by project phase instead of by artifact type.

- `phase1_reproduction/vanilla_pytorch/` — original PyTorch Re=10 reproduction baseline.
- `phase1_reproduction/strict_reproduce_scipy/` — best SciPy L-BFGS-B Re=10 reproduction baseline.
- `phase3_re100_reference/` — Re=100 reference-generation logs.
- `phase4a_loss_balancing/` — fixed-beta, GradNorm, strict GradNorm, and smoke-test runs.
- `reference/` — copied reference papers, analysis reports, and original-paper output figures.

Run directories use this layout when applicable:

- `checkpoints/`
- `logs/`
- `figures/`
- `benchmarks/`

New ad hoc runs that do not belong to a named phase should use:

- `results/experiments/<exp_name>/checkpoints/`
- `results/experiments/<exp_name>/logs/`
- `results/experiments/<exp_name>/figures/`
- `results/experiments/<exp_name>/benchmarks/`
