# Benchmark Matrix: Original Rao PINN Pickle Models

Last updated: 2026-05-26

This file records the evaluation of the original repository cloned under:

```text
ref/PINN-laminar-flow
```

Reference repository commit:

```text
d34fc03 Update README.md
```

The original models are stored as TensorFlow-era pickle files containing raw neural-network weights and biases. For this benchmark, they were evaluated without TensorFlow by reconstructing the network forward pass in PyTorch and using autograd for the stream-function derivatives:

```text
u = d psi / d y
v = - d psi / d x
p = direct network output channel
```

Pressure is reported after spatial demeaning because pressure has an arbitrary additive constant.

## Metric Definitions

Per-field relative L2:

```text
L2_u = ||u_pred - u_ref||_2 / ||u_ref||_2
L2_v = ||v_pred - v_ref||_2 / ||v_ref||_2
L2_p = ||demean(p_pred) - demean(p_ref)||_2 / ||demean(p_ref)||_2
```

Combined velocity L2:

```text
L2_velocity = ||[u_pred, v_pred] - [u_ref, v_ref]||_2 / ||[u_ref, v_ref]||_2
```

For unsteady evaluation, the primary metric matches `bench_unsteady.py`: mean per-snapshot relative L2 over developed-flow snapshots with `t >= 0.1s`. Global Frobenius L2 over all time is also listed as a reference.

## Summary

| Case | Model | Reference | Primary window | L2_u | L2_v | L2_p, demeaned | Combined / global velocity note |
|---|---|---|---:|---:|---:|---:|---|
| Steady | `ref/PINN-laminar-flow/PINN_steady/uvNN.pickle` | `ref/PINN-laminar-flow/FluentReferenceMu002/FluentSol.mat` | full field | 1.86% | 4.85% | 1.57% | combined velocity L2 = 1.98% |
| Unsteady | `ref/PINN-laminar-flow/PINN_unsteady/uvNN.pickle` | `data/reference/unsteady_reference.mat` | `t >= 0.1s` | 4.75% | 18.47% | 4.79% | global Frobenius: u 4.19%, v 14.38%, p 4.39% |

## Paper-Reported Numbers

The Rao et al. paper does not report separate `L2_u`, `L2_v`, or `L2_p`.

For the steady case, Table 1 reports only a combined velocity-field relative L2 error. The best mixed-variable PINN result is:

```text
Mixed-variable PINN, width=40, depth=8:
velocity relative L2 = 1.8 x 10^-2 = 1.8%
```

Our direct evaluation of the original steady `uvNN.pickle` gives:

```text
combined velocity L2 = 1.98%
```

This is close to the paper's reported 1.8%.

For the unsteady case, the paper reports no quantitative L2 values. It only provides:

- Fig. 7: qualitative `u`, `v`, `p` snapshots at `t = 0.3, 0.4, 0.5s`
- Fig. 8: qualitative pressure time histories at probes P1, P2, P3

Therefore the unsteady `L2_u/L2_v/L2_p` values below are our own benchmark against `data/reference/unsteady_reference.mat`.

## Steady Details

Model:

```text
ref/PINN-laminar-flow/PINN_steady/uvNN.pickle
```

Reference:

```text
ref/PINN-laminar-flow/FluentReferenceMu002/FluentSol.mat
```

Reference field size:

```text
Ns = 19340
```

Detailed metrics:

| Quantity | Value |
|---|---:|
| L2_u | 1.8637% |
| L2_v | 4.8528% |
| L2_p, demeaned | 1.5743% |
| L2_velocity, combined `[u, v]` | 1.9786% |

Field RMS sanity check:

| Field | pred RMS | ref RMS |
|---|---:|---:|
| u | 0.738689 | 0.737555 |
| v | 0.110903 | 0.110610 |
| p, demeaned | 0.781244 | 0.787048 |

Max absolute value sanity check:

| Field | pred max abs | ref max abs |
|---|---:|---:|
| u | 1.302488 | 1.300000 |
| v | 0.552438 | 0.552000 |
| p, demeaned | 2.901253 | 2.821196 |

## Unsteady Details

Model:

```text
ref/PINN-laminar-flow/PINN_unsteady/uvNN.pickle
```

Reference:

```text
data/reference/unsteady_reference.mat
```

Reference field size:

```text
Ns = 17724
Nt = 51
t range = [0.0, 0.5]
```

Primary metric over developed flow:

```text
t >= 0.1s, 41 snapshots
```

| Metric | Value |
|---|---:|
| Mean L2_u | 4.7542% |
| Mean L2_v | 18.4731% |
| Mean L2_p, demeaned | 4.7927% |

Global Frobenius metrics over all time:

| Metric | Value |
|---|---:|
| Global L2_u | 4.1876% |
| Global L2_v | 14.3797% |
| Global L2_p, demeaned | 4.3884% |

Selected snapshot metrics:

| t | L2_u | L2_v | L2_p, demeaned | pred RMS u | pred RMS v | pred RMS p, demeaned | ref RMS u | ref RMS v | ref RMS p, demeaned |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 8.77% | 37.16% | 2.23% | 0.06759 | 0.009628 | 0.4466 | 0.06688 | 0.008609 | 0.4560 |
| 0.20 | 4.84% | 21.95% | 2.62% | 0.2445 | 0.03404 | 0.7736 | 0.2443 | 0.03277 | 0.7930 |
| 0.30 | 4.14% | 14.99% | 4.02% | 0.4651 | 0.06428 | 0.8358 | 0.4666 | 0.06447 | 0.8697 |
| 0.40 | 4.12% | 13.20% | 6.10% | 0.6447 | 0.08913 | 0.6263 | 0.6508 | 0.09160 | 0.6639 |
| 0.50 | 4.08% | 13.44% | 16.97% | 0.7258 | 0.1012 | 0.2911 | 0.7281 | 0.1028 | 0.2763 |

Early-time notes:

| t | L2_u | L2_v | L2_p, demeaned | Note |
|---:|---:|---:|---:|---|
| 0.00 | NaN | NaN | NaN | reference field norm is near zero |
| 0.01 | 201.82% | NaN | 7.86% | denominator artifact from near-rest initial condition |
| 0.05 | 19.37% | 71.21% | 1.27% | transition period before developed-flow window |

## Interpretation

The original unsteady pickle does not collapse to the near-zero-flow solution. Against our unsteady reference dataset, it reaches:

```text
L2_u ~= 4.8%
L2_v ~= 18.5%
L2_p ~= 4.8%
```

This is substantially better than the failing PyTorch run discussed separately, where the model produced approximately:

```text
L2_u ~= 48%
L2_v ~= 100%
```

That contrast supports the conclusion that the benchmark/reference data are not the cause of the large PyTorch unsteady errors. The original pickle can learn the transient velocity field on the same dataset, while the current PyTorch training run converged to a near-trivial flow solution.
