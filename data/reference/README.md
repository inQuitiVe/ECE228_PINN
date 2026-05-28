# Reference Datasets

Tracked CFD reference data and generated visualizations used for PINN training and evaluation.

## Re=10 Unsteady Reference

- Data: `unsteady_reference.mat`
- Generator: `gen_unsteady_reference.py`
- Figures: `figures/unsteady/` and `figures/field_animation.gif`

This is the default benchmark target for Re=10 training and evaluation.

Typical benchmark usage:

```bash
bash scripts/bench_unsteady.sh --reference data/reference/unsteady_reference.mat
```

Stored variables:

- `x`, `y`: fluid point coordinates
- `t`: snapshot times
- `u`, `v`, `p`: reference velocity and pressure fields

Main settings:

- Domain: `x in [0, 1.1]`, `y in [0, 0.41]`
- Cylinder: center `(0.2, 0.2)`, radius `0.05`
- Dynamic viscosity: `mu = 0.005`
- Reynolds number: `Re = 10`
- Time range: `t in [0, 0.5]`
- Snapshots: 51

## Re=100 Unsteady Reference

- Data: `unsteady_reference_re100_t2.mat`
- Generator: `gen_unsteady_reference_re100.py`
- Figure generator: `gen_figures_re100_t2.py`
- Figures: `figures/re100_t2/` and `figures/re100_t2_animation.gif`

This dataset supports the Phase 3 Re=100 stress test.

Typical benchmark usage:

```bash
bash scripts/bench_re100.sh --reference data/reference/unsteady_reference_re100_t2.mat
```

Main settings:

- Domain: `x in [0, 1.1]`, `y in [0, 0.41]`
- Cylinder: center `(0.2, 0.2)`, radius `0.05`
- Dynamic viscosity: `mu = 0.0005`
- Reynolds number: `Re = 100`
- Time range: `t in [0, 2.0]`
- Snapshots: 51

## Notes

- This directory only documents files currently tracked by Git.
- Additional local CFD or Fluent files under ignored directories are not part of this reference package.
