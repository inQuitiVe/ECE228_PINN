# Unsteady L2 Matrix: Original Rao Pickle vs Local Reference

Model: `ref/PINN-laminar-flow/PINN_unsteady/uvNN.pickle`
Reference: `data/reference/unsteady_reference.mat`

Pressure is spatially demeaned per snapshot before computing L2. Early near-rest snapshots can produce NaN or inflated relative errors because the reference norm is close to zero.

## Segment Means

| Time segment | Snapshots | Mean L2_u | Mean L2_v | Mean L2_p, demeaned | Mean combined velocity L2 |
|---|---:|---:|---:|---:|---:|
| 0.00 <= t < 0.05 | 5 | 82.77% | 180.87% | 3.39% | 95.15% |
| 0.05 <= t < 0.10 | 5 | 13.74% | 52.83% | 1.75% | 15.14% |
| 0.10 <= t < 0.20 | 10 | 6.46% | 29.01% | 2.42% | 7.44% |
| 0.20 <= t < 0.30 | 10 | 4.41% | 18.36% | 3.12% | 5.02% |
| 0.30 <= t < 0.40 | 10 | 4.12% | 13.88% | 5.03% | 4.51% |
| 0.40 <= t <= 0.50 | 11 | 4.09% | 13.18% | 8.25% | 4.45% |
| developed: t >= 0.10 | 41 | 4.75% | 18.47% | 4.79% | 5.33% |
| all valid snapshots | 51 | 11.89% | 31.92% | 4.38% | 13.50% |

## Global Frobenius L2

| Metric | Value |
|---|---:|
| Global L2_u | 4.19% |
| Global L2_v | 14.38% |
| Global L2_p, demeaned per snapshot | 4.39% |
| Global combined velocity L2 | 4.60% |

## Selected Per-Time Matrix

| t | L2_u | L2_v | L2_p, demeaned | combined velocity L2 |
|---:|---:|---:|---:|---:|
| 0.00 | NaN | NaN | NaN | NaN |
| 0.01 | 201.82% | NaN | 7.86% | 237.83% |
| 0.05 | 19.37% | 71.21% | 1.27% | 21.10% |
| 0.10 | 8.77% | 37.16% | 2.23% | 9.90% |
| 0.15 | 6.01% | 27.64% | 2.44% | 6.97% |
| 0.20 | 4.84% | 21.95% | 2.62% | 5.62% |
| 0.25 | 4.32% | 17.78% | 3.12% | 4.91% |
| 0.30 | 4.14% | 14.99% | 4.02% | 4.58% |
| 0.35 | 4.11% | 13.63% | 5.17% | 4.49% |
| 0.40 | 4.12% | 13.20% | 6.10% | 4.47% |
| 0.45 | 4.09% | 13.09% | 6.43% | 4.44% |
| 0.50 | 4.08% | 13.44% | 16.97% | 4.46% |

Full 51-snapshot matrix is in `ref/unsteady_l2_matrix.csv`.
