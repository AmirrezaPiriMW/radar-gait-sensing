# Decoupling experiment

Speed–cadence correlation in the training data: coupled r = 0.91, decoupled r = 0.57.

Speed from true cadence + height only (no radar), MAE (m/s): coupled Nominal 0.103, Oblique heading 0.111, Low SNR 0.105; decoupled Nominal 0.203, Oblique heading 0.189, Low SNR 0.201

MAE with 95 % subject-level cluster-bootstrap CI. C = trained on coupled data, D = trained on decoupled data.

## Speed on coupled test data

| Condition | Physics baseline | Mono, trained coupled | Multi, trained coupled | Mono, trained decoupled | Multi, trained decoupled |
|---|---|---|---|---|---|
| Nominal | 0.114 [0.101, 0.131] | 0.046 [0.038, 0.055] | 0.045 [0.037, 0.056] | 0.049 [0.045, 0.052] | 0.041 [0.036, 0.046] |
| Oblique heading | 0.448 [0.369, 0.537] | 0.122 [0.099, 0.149] | 0.120 [0.105, 0.133] | 0.166 [0.141, 0.192] | 0.122 [0.107, 0.135] |
| Low SNR | 0.085 [0.065, 0.105] | 0.284 [0.224, 0.342] | 0.289 [0.232, 0.345] | 0.256 [0.210, 0.307] | 0.233 [0.188, 0.280] |

Paired differences (positive = second method better):

| Condition | mono C − mono D | multi C − multi D | mono D − multi D | baseline − multi D |
|---|---|---|---|---|
| Nominal | -0.003 [-0.010, +0.004] | +0.004 [-0.004, +0.013] | +0.007 [+0.006, +0.009] | +0.072 [+0.058, +0.090] |
| Oblique heading | -0.044 [-0.052, -0.034] | -0.002 [-0.024, +0.017] | +0.044 [+0.022, +0.062] | +0.326 [+0.240, +0.423] |
| Low SNR | +0.028 [+0.011, +0.042] | +0.055 [+0.038, +0.073] | +0.023 [+0.017, +0.029] | -0.149 [-0.195, -0.093] |

## Speed on decoupled test data

| Condition | Physics baseline | Mono, trained coupled | Multi, trained coupled | Mono, trained decoupled | Multi, trained decoupled |
|---|---|---|---|---|---|
| Nominal | 0.102 [0.086, 0.123] | 0.140 [0.131, 0.150] | 0.118 [0.113, 0.123] | 0.050 [0.041, 0.058] | 0.045 [0.039, 0.050] |
| Oblique heading | 0.382 [0.340, 0.427] | 0.154 [0.137, 0.176] | 0.177 [0.159, 0.198] | 0.135 [0.116, 0.160] | 0.105 [0.086, 0.127] |
| Low SNR | 0.082 [0.066, 0.101] | 0.286 [0.251, 0.325] | 0.275 [0.240, 0.311] | 0.216 [0.194, 0.241] | 0.205 [0.185, 0.225] |

Paired differences (positive = second method better):

| Condition | mono C − mono D | multi C − multi D | mono D − multi D | baseline − multi D |
|---|---|---|---|---|
| Nominal | +0.090 [+0.080, +0.106] | +0.073 [+0.064, +0.082] | +0.005 [-0.001, +0.012] | +0.057 [+0.040, +0.076] |
| Oblique heading | +0.019 [+0.006, +0.033] | +0.072 [+0.068, +0.078] | +0.030 [+0.016, +0.039] | +0.277 [+0.220, +0.331] |
| Low SNR | +0.071 [+0.057, +0.084] | +0.069 [+0.044, +0.093] | +0.010 [-0.003, +0.023] | -0.123 [-0.137, -0.105] |

## Cadence on coupled test data

| Condition | Physics baseline | Mono, trained coupled | Multi, trained coupled | Mono, trained decoupled | Multi, trained decoupled |
|---|---|---|---|---|---|
| Nominal | 0.98 [0.87, 1.10] | 1.13 [0.99, 1.26] | 1.17 [0.99, 1.34] | 1.26 [1.01, 1.54] | 1.19 [1.00, 1.40] |
| Oblique heading | 11.85 [7.07, 17.20] | 4.30 [3.02, 5.69] | 2.63 [2.28, 3.05] | 3.26 [2.59, 4.02] | 2.65 [2.14, 3.17] |
| Low SNR | 11.26 [9.37, 13.01] | 9.76 [7.83, 11.95] | 9.72 [7.76, 11.81] | 9.74 [7.67, 12.37] | 10.58 [8.91, 12.82] |

Paired differences (positive = second method better):

| Condition | mono C − mono D | multi C − multi D | mono D − multi D | baseline − multi D |
|---|---|---|---|---|
| Nominal | -0.13 [-0.31, +0.04] | -0.03 [-0.24, +0.19] | +0.07 [-0.02, +0.15] | -0.21 [-0.53, +0.08] |
| Oblique heading | +1.04 [+0.41, +2.00] | -0.01 [-0.37, +0.33] | +0.61 [+0.30, +0.95] | +9.21 [+4.73, +14.36] |
| Low SNR | +0.02 [-0.48, +0.39] | -0.86 [-1.49, -0.30] | -0.84 [-1.25, -0.43] | +0.68 [-1.75, +3.18] |

## Cadence on decoupled test data

| Condition | Physics baseline | Mono, trained coupled | Multi, trained coupled | Mono, trained decoupled | Multi, trained decoupled |
|---|---|---|---|---|---|
| Nominal | 1.02 [0.84, 1.29] | 2.31 [2.07, 2.59] | 2.76 [2.45, 3.08] | 1.43 [1.20, 1.65] | 1.46 [1.23, 1.67] |
| Oblique heading | 10.27 [6.81, 13.13] | 6.66 [5.77, 7.67] | 5.11 [4.19, 6.25] | 4.46 [3.55, 5.27] | 3.75 [2.79, 4.61] |
| Low SNR | 14.98 [11.18, 19.05] | 12.19 [9.78, 14.57] | 13.51 [11.13, 15.68] | 10.67 [8.36, 12.97] | 11.73 [9.35, 13.88] |

Paired differences (positive = second method better):

| Condition | mono C − mono D | multi C − multi D | mono D − multi D | baseline − multi D |
|---|---|---|---|---|
| Nominal | +0.88 [+0.58, +1.18] | +1.30 [+0.92, +1.68] | -0.02 [-0.09, +0.05] | -0.44 [-0.72, -0.16] |
| Oblique heading | +2.21 [+1.22, +3.17] | +1.37 [+0.65, +2.01] | +0.71 [+0.28, +1.06] | +6.53 [+3.39, +9.22] |
| Low SNR | +1.52 [+1.20, +1.84] | +1.78 [+1.08, +2.37] | -1.06 [-1.79, -0.38] | +3.25 [+1.23, +5.41] |

## Correlation of speed errors with cadence+height-prior errors

| Test data | Condition | mono C | multi C | mono D | multi D |
|---|---|---|---|---|---|
| coupled | Nominal | 0.06 [-0.18, 0.48] | 0.16 [-0.03, 0.43] | -0.00 [-0.22, 0.40] | -0.03 [-0.19, 0.25] |
| coupled | Oblique heading | 0.23 [-0.06, 0.54] | 0.16 [-0.09, 0.55] | 0.18 [-0.06, 0.51] | 0.08 [-0.17, 0.49] |
| coupled | Low SNR | 0.40 [0.13, 0.67] | 0.39 [0.11, 0.68] | 0.40 [0.12, 0.70] | 0.36 [0.09, 0.69] |
| decoupled | Nominal | 0.64 [0.60, 0.68] | 0.63 [0.58, 0.69] | 0.28 [0.14, 0.42] | 0.27 [0.17, 0.37] |
| decoupled | Oblique heading | 0.73 [0.67, 0.79] | 0.77 [0.68, 0.84] | 0.48 [0.43, 0.53] | 0.58 [0.50, 0.64] |
| decoupled | Low SNR | 0.85 [0.79, 0.90] | 0.80 [0.75, 0.86] | 0.81 [0.74, 0.87] | 0.75 [0.69, 0.81] |

## Mean raw predicted speed sigma (m/s)

| Test data | Condition | mono C | multi C | mono D | multi D |
|---|---|---|---|---|---|
| coupled | Nominal | 0.092 | 0.088 | 0.085 | 0.079 |
| coupled | Oblique heading | 0.111 | 0.113 | 0.105 | 0.104 |
| coupled | Low SNR | 0.164 | 0.147 | 0.159 | 0.149 |
| decoupled | Nominal | 0.098 | 0.094 | 0.085 | 0.078 |
| decoupled | Oblique heading | 0.111 | 0.114 | 0.099 | 0.101 |
| decoupled | Low SNR | 0.167 | 0.152 | 0.161 | 0.150 |

## 90 % interval coverage (each family recalibrated on its own validation data)

| Test data | Condition | target | mono C | multi C | mono D | multi D |
|---|---|---|---|---|---|---|
| coupled | Nominal | speed | 95.6 % | 95.0 % | 93.3 % | 95.6 % |
| coupled | Nominal | cadence | 96.7 % | 97.2 % | 95.0 % | 95.6 % |
| coupled | Oblique heading | speed | 71.7 % | 64.2 % | 55.8 % | 62.5 % |
| coupled | Oblique heading | cadence | 62.5 % | 85.8 % | 70.8 % | 89.2 % |
| coupled | Low SNR | speed | 40.0 % | 42.5 % | 45.0 % | 45.0 % |
| coupled | Low SNR | cadence | 45.0 % | 43.3 % | 46.7 % | 43.3 % |
| decoupled | Nominal | speed | 53.9 % | 62.2 % | 93.3 % | 93.3 % |
| decoupled | Nominal | cadence | 80.6 % | 79.4 % | 92.2 % | 93.3 % |
| decoupled | Oblique heading | speed | 55.0 % | 44.2 % | 64.2 % | 68.3 % |
| decoupled | Oblique heading | cadence | 58.3 % | 66.7 % | 70.0 % | 80.0 % |
| decoupled | Low SNR | speed | 40.0 % | 38.3 % | 48.3 % | 45.0 % |
| decoupled | Low SNR | cadence | 41.7 % | 35.0 % | 45.0 % | 40.8 % |
