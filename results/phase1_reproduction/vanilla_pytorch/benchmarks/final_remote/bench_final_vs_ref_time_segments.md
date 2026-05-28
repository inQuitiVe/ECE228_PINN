# bench_final vs ref Unsteady L2 by Time Segment

Bench source: `results/bench_final/snapshot_metrics.csv`
Ref source: `ref/unsteady_l2_matrix.csv` from original Rao `PINN_unsteady/uvNN.pickle`.

All values below are percentages. Delta is `bench_final - ref` in percentage points. Pressure uses the demeaned L2 convention used by the benchmark.

## Segment Mean Comparison

| Time segment | N | bench u | ref u | delta u | bench v | ref v | delta v | bench p | ref p | delta p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 <= t < 0.05 | 5 | 85.10% | 82.77% | +2.33 pp | 194.16% | 180.87% | +13.29 pp | 6.04% | 3.39% | +2.65 pp |
| 0.05 <= t < 0.10 | 5 | 16.86% | 13.74% | +3.13 pp | 66.86% | 52.83% | +14.02 pp | 3.08% | 1.75% | +1.33 pp |
| 0.10 <= t < 0.20 | 10 | 9.09% | 6.46% | +2.63 pp | 36.40% | 29.01% | +7.39 pp | 2.47% | 2.42% | +0.05 pp |
| 0.20 <= t < 0.30 | 10 | 5.75% | 4.41% | +1.34 pp | 22.94% | 18.36% | +4.58 pp | 6.64% | 3.12% | +3.52 pp |
| 0.30 <= t < 0.40 | 10 | 4.73% | 4.12% | +0.62 pp | 16.53% | 13.88% | +2.65 pp | 8.77% | 5.03% | +3.74 pp |
| 0.40 <= t <= 0.50 | 11 | 4.33% | 4.09% | +0.24 pp | 13.89% | 13.18% | +0.72 pp | 22.16% | 8.25% | +13.90 pp |
| developed: t >= 0.10 | 41 | 5.94% | 4.75% | +1.18 pp | 22.23% | 18.47% | +3.76 pp | 10.31% | 4.79% | +5.51 pp |
| all valid snapshots | 51 | 13.36% | 11.89% | +1.47 pp | 37.31% | 31.92% | +5.39 pp | 9.24% | 4.38% | +4.87 pp |

## Segment Ratios

| Time segment | bench/ref u | bench/ref v | bench/ref p |
|---|---:|---:|---:|
| 0.00 <= t < 0.05 | 1.03x | 1.07x | 1.78x |
| 0.05 <= t < 0.10 | 1.23x | 1.27x | 1.76x |
| 0.10 <= t < 0.20 | 1.41x | 1.25x | 1.02x |
| 0.20 <= t < 0.30 | 1.30x | 1.25x | 2.13x |
| 0.30 <= t < 0.40 | 1.15x | 1.19x | 1.74x |
| 0.40 <= t <= 0.50 | 1.06x | 1.05x | 2.68x |
| developed: t >= 0.10 | 1.25x | 1.20x | 2.15x |
| all valid snapshots | 1.12x | 1.17x | 2.11x |

## Selected Time Points

| t | bench u | ref u | delta u | bench v | ref v | delta v | bench p | ref p | delta p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| 0.01 | 206.11% | 201.82% | +4.29 pp | NaN | NaN | NaN | 10.06% | 7.86% | +2.20 pp |
| 0.05 | 22.29% | 19.37% | +2.92 pp | 87.84% | 71.21% | +16.63 pp | 3.54% | 1.27% | +2.27 pp |
| 0.10 | 11.87% | 8.77% | +3.10 pp | 47.50% | 37.16% | +10.34 pp | 2.42% | 2.23% | +0.19 pp |
| 0.15 | 8.61% | 6.01% | +2.60 pp | 34.43% | 27.64% | +6.79 pp | 2.10% | 2.44% | -0.34 pp |
| 0.20 | 6.77% | 4.84% | +1.93 pp | 27.27% | 21.95% | +5.31 pp | 4.23% | 2.62% | +1.61 pp |
| 0.25 | 5.56% | 4.32% | +1.24 pp | 22.30% | 17.78% | +4.52 pp | 7.01% | 3.12% | +3.89 pp |
| 0.30 | 4.93% | 4.14% | +0.79 pp | 18.61% | 14.99% | +3.62 pp | 9.04% | 4.02% | +5.02 pp |
| 0.35 | 4.71% | 4.11% | +0.60 pp | 16.16% | 13.63% | +2.53 pp | 9.17% | 5.17% | +4.00 pp |
| 0.40 | 4.50% | 4.12% | +0.39 pp | 14.70% | 13.20% | +1.51 pp | 6.33% | 6.10% | +0.23 pp |
| 0.45 | 4.15% | 4.09% | +0.07 pp | 13.74% | 13.09% | +0.65 pp | 14.13% | 6.43% | +7.70 pp |
| 0.50 | 4.72% | 4.08% | +0.64 pp | 13.57% | 13.44% | +0.13 pp | 63.86% | 16.97% | +46.89 pp |

Full per-snapshot comparison is in `results/bench_final/bench_final_vs_ref_l2_matrix.csv`.
