# Checkpoint scheduling results

All three saved RL seeds; each baseline is evaluated once on the same worlds.

Rank by mean per-job excess within each split. Lower is better. Completed P=T; unfinished jobs stop at 8W and receive P=10W. E=100(P/W-1).

These test suites were previously evaluated. This is a measured benchmark comparison, not an unseen confirmation. RL seed pooling is not an ensemble.

## test

| Policy | Excess % | Complete % | Timeouts / runs | P95 excess % | CVaR95 excess % | Penalized s |
| --- | --- | --- | --- | --- | --- | --- |
| PPO (grid + fractions) | 29.05 | 100.00 | 0 / 4860 | 124.64 | 184.15 | 8,109.42 |
| Double DQN (trained grid) | 30.95 | 100.00 | 0 / 4860 | 132.22 | 197.78 | 8,225.96 |
| Daly | 31.14 | 100.00 | 0 / 1620 | 126.98 | 190.82 | 8,223.48 |
| Young | 31.45 | 100.00 | 0 / 1620 | 132.77 | 205.66 | 8,221.76 |
| Fixed 1,200 s | 51.54 | 99.94 | 1 / 1620 | 186.16 | 291.49 | 9,555.31 |

### Cost and checkpoint diagnostics

| Policy | Attempts/job | Writing s/job | Lost work s/job | Recovery s/job | Efficiency % | Usable % | Protected % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO (grid + fractions) | 7.93 | 593.50 | 1,059.19 | 137.31 | 77.93 | 82.83 | 50.45 |
| Double DQN (trained grid) | 10.03 | 748.50 | 1,018.99 | 139.05 | 76.82 | 85.30 | 49.32 |
| Daly | 7.93 | 802.58 | 962.76 | 138.71 | 76.85 | 82.54 | 48.86 |
| Young | 7.47 | 686.06 | 1,077.56 | 138.71 | 76.86 | 77.49 | 44.83 |
| Fixed 1,200 s | 4.98 | 1,846.40 | 1,226.09 | 160.20 | 66.16 | 67.02 | 23.10 |

### Seed stability and paired comparisons

| Policy | Seed SD pp | Seed min excess % | Seed max excess % |
| --- | --- | --- | --- |
| PPO (grid + fractions) | 0.45 | 28.61 | 29.52 |
| Double DQN (trained grid) | 0.37 | 30.71 | 31.38 |
| Daly | N/A | 31.14 | 31.14 |
| Young | N/A | 31.45 | 31.45 |
| Fixed 1,200 s | N/A | 51.54 | 51.54 |

| Policy | Reference | Difference pp | 95% interval pp | World wins % |
| --- | --- | --- | --- | --- |
| PPO (grid + fractions) | Double DQN (trained grid) | -1.90 | -2.68 to -1.13 | 55.62 |
| PPO (grid + fractions) | Young | -2.39 | -3.33 to -1.50 | 51.91 |
| PPO (grid + fractions) | Daly | -2.09 | -2.97 to -1.17 | 54.14 |
| PPO (grid + fractions) | Fixed 1,200 s | -22.49 | -25.39 to -19.96 | 68.70 |
| Double DQN (trained grid) | Young | -0.49 | -1.41 to 0.43 | 40.12 |
| Double DQN (trained grid) | Daly | -0.19 | -1.07 to 0.68 | 42.22 |
| Double DQN (trained grid) | Fixed 1,200 s | -20.59 | -23.18 to -18.23 | 62.53 |

## ood_test

| Policy | Excess % | Complete % | Timeouts / runs | P95 excess % | CVaR95 excess % | Penalized s |
| --- | --- | --- | --- | --- | --- | --- |
| PPO (grid + fractions) | 46.47 | 100.00 | 0 / 2430 | 204.15 | 288.09 | 13,829.49 |
| Young | 49.86 | 99.88 | 1 / 810 | 210.46 | 330.90 | 14,148.68 |
| Daly | 50.55 | 100.00 | 0 / 810 | 211.57 | 309.70 | 14,136.81 |
| Double DQN (trained grid) | 52.70 | 99.92 | 2 / 2430 | 235.98 | 343.12 | 14,490.49 |
| Fixed 1,200 s | 81.58 | 99.88 | 1 / 810 | 283.23 | 388.51 | 17,759.65 |

### Cost and checkpoint diagnostics

| Policy | Attempts/job | Writing s/job | Lost work s/job | Recovery s/job | Efficiency % | Usable % | Protected % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO (grid + fractions) | 9.17 | 1,478.23 | 2,587.98 | 285.12 | 68.54 | 85.32 | 47.09 |
| Young | 8.53 | 1,449.23 | 2,909.37 | 294.69 | 67.07 | 82.85 | 42.65 |
| Daly | 9.43 | 1,842.75 | 2,520.32 | 295.58 | 67.05 | 84.88 | 45.69 |
| Double DQN (trained grid) | 11.58 | 1,812.11 | 2,881.66 | 302.02 | 65.48 | 81.96 | 42.26 |
| Fixed 1,200 s | 8.06 | 5,442.52 | 2,495.90 | 337.92 | 53.38 | 77.85 | 27.68 |

### Seed stability and paired comparisons

| Policy | Seed SD pp | Seed min excess % | Seed max excess % |
| --- | --- | --- | --- |
| PPO (grid + fractions) | 0.19 | 46.27 | 46.64 |
| Young | N/A | 49.86 | 49.86 |
| Daly | N/A | 50.55 | 50.55 |
| Double DQN (trained grid) | 1.95 | 50.52 | 54.28 |
| Fixed 1,200 s | N/A | 81.58 | 81.58 |

| Policy | Reference | Difference pp | 95% interval pp | World wins % |
| --- | --- | --- | --- | --- |
| PPO (grid + fractions) | Double DQN (trained grid) | -6.22 | -8.44 to -4.18 | 62.59 |
| PPO (grid + fractions) | Young | -3.38 | -6.24 to -0.83 | 55.06 |
| PPO (grid + fractions) | Daly | -4.07 | -6.59 to -1.89 | 61.23 |
| PPO (grid + fractions) | Fixed 1,200 s | -35.11 | -40.09 to -30.46 | 74.32 |
| Double DQN (trained grid) | Young | 2.84 | 0.21 to 5.47 | 35.68 |
| Double DQN (trained grid) | Daly | 2.15 | -0.56 to 5.00 | 41.48 |
| Double DQN (trained grid) | Fixed 1,200 s | -28.89 | -33.39 to -24.28 | 64.07 |

Intervals resample worlds after averaging RL seeds within each world; they are pointwise and conditional on the saved models, with no multiple-comparison adjustment.

Full metrics, individual seeds, every process/work/risk/cost slice and all 81 cells are included in the adjacent JSON/CSV files. Counts and protection are diagnostics.
