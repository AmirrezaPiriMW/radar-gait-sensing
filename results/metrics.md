# Results

MAE with 95 % subject-level cluster-bootstrap CI in brackets. Uncertainty columns use the variance recalibrated on validation data.

## Speed (m/s)

| Condition | Physics baseline | CNN mono | CNN multi |
|---|---|---|---|
| Nominal | 0.114 [0.101, 0.131] | 0.046 [0.038, 0.055] | 0.045 [0.037, 0.056] |
| Oblique heading (45–70°) | 0.448 [0.369, 0.537] | 0.122 [0.099, 0.149] | 0.120 [0.105, 0.133] |
| Low SNR (−22 to −14 dB) | 0.085 [0.065, 0.105] | 0.284 [0.224, 0.342] | 0.289 [0.232, 0.345] |

| Condition | Model | 90 % interval coverage | calib. error | Spearman(abs err, σ) | mean σ | MAE, 80 % most confident |
|---|---|---|---|---|---|---|
| Nominal | mono | 95.6 % | 0.074 | 0.29 | 0.069 | 0.042 |
| Nominal | multi | 95.0 % | 0.069 | 0.17 | 0.069 | 0.042 |
| Oblique heading (45–70°) | mono | 71.7 % | 0.138 | 0.52 | 0.083 | 0.106 |
| Oblique heading (45–70°) | multi | 64.2 % | 0.229 | 0.39 | 0.089 | 0.112 |
| Low SNR (−22 to −14 dB) | mono | 40.0 % | 0.316 | 0.37 | 0.123 | 0.242 |
| Low SNR (−22 to −14 dB) | multi | 42.5 % | 0.325 | 0.35 | 0.116 | 0.263 |

Paired differences in MAE (positive = second method better):

| Condition | baseline − mono | mono − multi | baseline − multi |
|---|---|---|---|
| Nominal | +0.068 [+0.050, +0.091] | +0.000 [-0.004, +0.005] | +0.068 [+0.050, +0.089] |
| Oblique heading (45–70°) | +0.326 [+0.248, +0.415] | +0.002 [-0.024, +0.031] | +0.328 [+0.256, +0.412] |
| Low SNR (−22 to −14 dB) | -0.200 [-0.256, -0.130] | -0.004 [-0.015, +0.005] | -0.204 [-0.261, -0.138] |

## Cadence (steps/min)

| Condition | Physics baseline | CNN mono | CNN multi |
|---|---|---|---|
| Nominal | 0.98 [0.87, 1.10] | 1.13 [0.99, 1.26] | 1.17 [0.99, 1.34] |
| Oblique heading (45–70°) | 11.85 [7.07, 17.20] | 4.30 [3.02, 5.69] | 2.63 [2.28, 3.05] |
| Low SNR (−22 to −14 dB) | 11.26 [9.37, 13.01] | 9.76 [7.83, 11.95] | 9.72 [7.76, 11.81] |

| Condition | Model | 90 % interval coverage | calib. error | Spearman(abs err, σ) | mean σ | MAE, 80 % most confident |
|---|---|---|---|---|---|---|
| Nominal | mono | 96.7 % | 0.084 | 0.37 | 1.71 | 0.91 |
| Nominal | multi | 97.2 % | 0.115 | 0.32 | 1.93 | 0.97 |
| Oblique heading (45–70°) | mono | 62.5 % | 0.135 | 0.67 | 2.25 | 3.37 |
| Oblique heading (45–70°) | multi | 85.8 % | 0.023 | 0.50 | 2.55 | 2.28 |
| Low SNR (−22 to −14 dB) | mono | 45.0 % | 0.252 | 0.68 | 3.90 | 7.72 |
| Low SNR (−22 to −14 dB) | multi | 43.3 % | 0.290 | 0.57 | 3.94 | 8.23 |

Paired differences in MAE (positive = second method better):

| Condition | baseline − mono | mono − multi | baseline − multi |
|---|---|---|---|
| Nominal | -0.15 [-0.36, +0.02] | -0.04 [-0.15, +0.08] | -0.18 [-0.38, +0.02] |
| Oblique heading (45–70°) | +7.56 [+3.82, +11.85] | +1.66 [+0.73, +2.68] | +9.22 [+4.73, +14.27] |
| Low SNR (−22 to −14 dB) | +1.50 [-1.08, +4.33] | +0.04 [-0.28, +0.30] | +1.54 [-1.06, +4.50] |

## Pooled selective prediction (speed, all conditions)

| Model | MAE all | MAE keep 80 % | MAE keep 60 % | Spearman |
|---|---|---|---|---|
| mono | 0.136 | 0.088 | 0.053 | 0.65 |
| multi | 0.136 | 0.091 | 0.059 | 0.64 |

## Uncertainty-gated fallback (mono CNN → baseline when σ > 0.092 m/s)

| Condition | CNN | baseline | gated | fallback share |
|---|---|---|---|---|
| Nominal | 0.046 | 0.114 | 0.058 | 0.03 |
| Oblique heading (45–70°) | 0.122 | 0.448 | 0.335 | 0.33 |
| Low SNR (−22 to −14 dB) | 0.284 | 0.085 | 0.107 | 0.82 |

Speed from true cadence + height only (no radar), MAE: Nominal 0.103, Oblique heading (45–70°) 0.111, Low SNR (−22 to −14 dB) 0.105; speed–cadence correlation r = 0.91.
Training walks with line-of-sight angle > 40°: 13 %.

## Correlation of CNN speed errors with cadence+height-prior errors

| Condition | mono r [95 % CI] | multi r [95 % CI] |
|---|---|---|
| Nominal | 0.06 [-0.18, 0.48] | 0.16 [-0.03, 0.43] |
| Oblique heading (45–70°) | 0.23 [-0.06, 0.54] | 0.16 [-0.09, 0.55] |
| Low SNR (−22 to −14 dB) | 0.40 [0.13, 0.67] | 0.39 [0.11, 0.68] |
