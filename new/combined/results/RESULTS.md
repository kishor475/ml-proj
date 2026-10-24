# Results

All numbers are weighted-F1 (%). `*` = the 95 % paired-bootstrap interval excludes 0. One training run per arm: differences below ~1-2 points are within noise.

## A. Full-length runs (3 epochs, test set)

Comparison uses each arm's FINAL epoch against the base's FINAL epoch (no test-set epoch selection); 'best' is the best epoch on the test set (optimistic, shown for reference). Paper = CoE reported score.

### emorynlp  (paper 44.29)

| arm | epochs | best | final | final vs base [95% CI] | |
|---|---|---|---|---|---|
| base | 3 | 40.00 | 40.00 | - | |
| chain_all | 3 | 37.90 | 36.44 | -3.53 [-6.10, -0.95] | worse* |
| chain_lf_all | 3 | 36.56 | 36.56 | -3.44 [-6.03, -0.86] | worse* |
| chain_prev | 3 | 38.30 | 38.30 | -1.69 [-3.97, +0.61] | n.s. |
| chain_self | 3 | 38.07 | 38.07 | -1.85 [-4.28, +0.50] | n.s. |
| erc_continued_to_8_epochs | 5 | 39.27 | 39.27 | -0.81 [-2.71, +1.06] | n.s. |

### meld  (paper 70.11)

| arm | epochs | best | final | final vs base [95% CI] | |
|---|---|---|---|---|---|
| base | 3 | 67.76 | 67.35 | - | |
| chain_all | 3 | 66.23 | 66.23 | -1.15 [-2.54, +0.36] | n.s. |
| chain_prev | 3 | 68.56 | 66.56 | -0.78 [-2.24, +0.65] | n.s. |
| chain_self | 3 | 65.86 | 65.38 | -1.98 [-3.37, -0.56] | worse* |

### iemocap  (paper 70.46)

| arm | epochs | best | final | final vs base [95% CI] | |
|---|---|---|---|---|---|
| base | 3 | 69.97 | 69.97 | - | |
| chain_all | 3 | 66.19 | 66.19 | -3.81 [-5.88, -1.70] | worse* |
| chain_lf_all | 3 | 69.24 | 66.00 | -3.95 [-5.92, -2.00] | worse* |
| chain_prev | 3 | 69.08 | 69.08 | -0.88 [-2.64, +0.92] | n.s. |
| chain_self | 3 | 68.62 | 68.62 | -1.27 [-3.04, +0.51] | n.s. |

## B. Short protocol (1 epoch, 30 % of the training data, held-out DEV rows)

Every arm starts from the same stage-1 adapter. `nocue` = the same fine-tune without any added clue; `shuffled*` = the same clue text on random turns (control). EmoryNLP/MELD dev = their dev sets; IEMOCAP dev = every 5th training dialogue.

### emorynlp

| arm | dev wF1 | vs nocue [95% CI] | |
|---|---|---|---|
| lf-all | 41.95 | +2.57 [+0.81, +4.48] | better* |
| chain-all | 41.85 | +2.44 [+0.45, +4.45] | better* |
| lf-prev | 41.11 | +1.69 [+0.04, +3.24] | better* |
| chain-prev | 40.19 | +0.73 [-0.95, +2.52] | n.s. |
| relmap | 39.73 | +0.30 [-0.89, +1.52] | n.s. |
| nocue | 39.44 | - |  |
| desc-ctx-nop | 39.42 | -0.01 [-1.21, +1.31] | n.s. |
| desc-ctx-nov | 39.41 | -0.05 [-1.42, +1.26] | n.s. |
| relmap-cat | 39.38 | -0.07 [-1.17, +1.07] | n.s. |
| desc | 39.34 | -0.07 [-1.24, +1.09] | n.s. |
| shuffled-relmap | 39.24 | -0.19 [-1.26, +0.83] | n.s. |
| desc-ctx | 39.21 | -0.25 [-1.56, +1.02] | n.s. |
| desc-ctx-spk | 38.93 | -0.48 [-1.88, +0.88] | n.s. |
| desc-ctx-3c | 38.91 | -0.53 [-2.00, +0.75] | n.s. |
| shuffled-desc | 38.68 | -0.75 [-2.00, +0.35] | n.s. |
| shuffled | 38.53 | -0.94 [-2.28, +0.41] | n.s. |
| shuffled-desc-ctx | 38.53 | -0.95 [-2.22, +0.33] | n.s. |
| desc-ctx-nor | 38.48 | -0.93 [-2.32, +0.42] | n.s. |
| default | 38.42 | -1.03 [-2.40, +0.44] | n.s. |
| both-ctx | 37.69 | -1.74 [-3.30, -0.34] | worse* |
| relmap-desc | 37.61 | -1.86 [-3.22, -0.52] | worse* |

### meld

| arm | dev wF1 | vs nocue [95% CI] | |
|---|---|---|---|
| lf-all | 69.20 | +0.26 [-1.64, +2.15] | n.s. |
| nocue | 68.97 | - |  |
| relmap | 68.91 | -0.07 [-1.09, +0.88] | n.s. |
| shuffled-relmap | 68.89 | -0.06 [-1.06, +0.90] | n.s. |
| relmap-desc | 68.80 | -0.16 [-1.22, +0.87] | n.s. |
| relmap-cat | 68.74 | -0.24 [-1.16, +0.64] | n.s. |
| lf-prev | 68.67 | -0.33 [-2.05, +1.42] | n.s. |
| desc-ctx-nop | 68.58 | -0.39 [-1.69, +0.96] | n.s. |
| chain-all | 68.56 | -0.34 [-2.50, +1.60] | n.s. |
| shuffled-desc-ctx | 68.51 | -0.47 [-1.89, +0.88] | n.s. |
| desc-ctx-spk | 68.50 | -0.48 [-1.85, +0.91] | n.s. |
| default | 68.39 | -0.59 [-1.66, +0.50] | n.s. |
| desc-ctx-3c | 68.38 | -0.55 [-1.92, +0.83] | n.s. |
| both-ctx | 68.37 | -0.58 [-1.89, +0.69] | n.s. |
| desc-ctx-nov | 68.30 | -0.67 [-1.91, +0.50] | n.s. |
| desc | 68.28 | -0.70 [-1.93, +0.47] | n.s. |
| shuffled-desc | 68.20 | -0.75 [-2.08, +0.55] | n.s. |
| desc-ctx | 68.17 | -0.82 [-2.09, +0.49] | n.s. |
| desc-ctx-nor | 68.14 | -0.84 [-2.14, +0.42] | n.s. |
| shuffled | 68.02 | -0.95 [-2.04, +0.14] | n.s. |
| chain-prev | 65.81 | -3.19 [-5.34, -1.03] | worse* |

### iemocap

| arm | dev wF1 | vs nocue [95% CI] | |
|---|---|---|---|
| chain-all | 64.67 | +4.46 [+2.23, +6.75] | better* |
| chain-all-known | 64.26 | +4.00 [+1.63, +6.51] | better* |
| lf-all | 64.11 | +3.94 [+1.74, +6.07] | better* |
| lf-prev | 62.09 | +1.83 [-0.01, +3.66] | n.s. |
| desc-nop | 60.86 | +0.64 [-0.98, +2.31] | n.s. |
| nocue | 60.24 | - |  |
| desc-3c | 60.07 | -0.21 [-1.96, +1.52] | n.s. |
| 3c-p25-75-global-8feat | 59.98 | -0.30 [-2.10, +1.39] | n.s. |
| shuffled | 59.74 | -0.50 [-1.88, +1.04] | n.s. |
| desc-nov | 59.58 | -0.68 [-2.36, +0.86] | n.s. |
| desc | 59.31 | -0.95 [-2.68, +0.84] | n.s. |
| both | 59.04 | -1.25 [-3.37, +0.84] | n.s. |
| desc-nor | 58.93 | -1.26 [-3.19, +0.73] | n.s. |
| chain-prev | 58.54 | -1.66 [-3.85, +0.61] | n.s. |
| desc-spk | 58.24 | -2.04 [-4.01, +0.02] | n.s. |
| 5c-p10-30-70-90-speaker-pct-8feat | 58.02 | -2.29 [-4.25, -0.33] | worse* |
| default | 57.62 | -2.60 [-4.78, -0.57] | worse* |
| shuffled-desc | 56.25 | -3.93 [-5.67, -2.23] | worse* |

