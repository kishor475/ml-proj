# Results

All numbers are weighted-F1 (%). `*` = the 95 % paired-bootstrap interval excludes 0. One training run per arm: differences below ~1-2 points are within noise.

## A. Full-length runs (3 epochs, test set)

Comparison uses each arm's FINAL epoch against the base's FINAL epoch (no test-set epoch selection); 'best' is the best epoch on the test set (optimistic, shown for reference). Paper = CoE reported score.

### emorynlp  (paper 44.29)

| arm | epochs | best | final | final vs base [95% CI] | |
|---|---|---|---|---|---|
| base | 3 | 42.00 | 42.00 | - | |
| chain_all | 3 | 39.90 | 38.44 | -3.53 [-6.10, -0.95] | worse* |
| chain_lf_all | 3 | 38.56 | 38.56 | -3.44 [-6.03, -0.86] | worse* |
| chain_prev | 3 | 40.30 | 40.30 | -1.69 [-3.97, +0.61] | n.s. |
| chain_self | 3 | 40.07 | 40.07 | -1.85 [-4.28, +0.50] | n.s. |
| erc_continued_to_8_epochs | 5 | 41.27 | 41.27 | -0.81 [-2.71, +1.06] | n.s. |
| erc_lora_rank16 | 3 | 40.55 | 40.55 | -1.42 [-3.67, +0.60] | n.s. |
| erc_mistral_v0.1 | 6 | 42.16 | 42.16 | +0.18 [-2.26, +2.69] | n.s. |

### meld  (paper 70.11)

| arm | epochs | best | final | final vs base [95% CI] | |
|---|---|---|---|---|---|
| base | 3 | 69.26 | 68.85 | - | |
| chain_all | 3 | 67.73 | 67.73 | -1.15 [-2.54, +0.36] | n.s. |
| chain_prev | 3 | 70.06 | 68.06 | -0.78 [-2.24, +0.65] | n.s. |
| chain_self | 3 | 67.36 | 66.88 | -1.98 [-3.37, -0.56] | worse* |

### iemocap  (paper 70.46)

| arm | epochs | best | final | final vs base [95% CI] | |
|---|---|---|---|---|---|
| base | 3 | 69.97 | 69.97 | - | |
| chain_all | 3 | 66.19 | 66.19 | -3.81 [-5.88, -1.70] | worse* |
| chain_lf_all | 3 | 69.24 | 66.00 | -3.95 [-5.92, -2.00] | worse* |
| chain_prev | 3 | 69.08 | 69.08 | -0.88 [-2.64, +0.92] | n.s. |
| chain_self | 3 | 68.62 | 68.62 | -1.27 [-3.04, +0.51] | n.s. |

## A2. Mistral-7B-v0.1 (EmoryNLP, 3 epochs, test set)

Same pipeline with Mistral-7B-v0.1 as the base model. Every arm is compared with the v0.1 base after its first 3 epochs (epochs 0-2). The v0.1 base was also continued to 6 epochs (a second warm-restarted learning-rate cycle, so its epoch 3 dips).

| arm | epoch 0 | epoch 1 | epoch 2 | vs v0.1 base @ epoch 2 [95% CI] | |
|---|---|---|---|---|---|
| acoustic_clue | 38.67 | 41.65 | 43.42 | +1.62 [-0.45, +3.79] | n.s. |
| chain_all | 36.16 | 36.70 | 38.27 | -3.46 [-5.88, -1.01] | worse* |
| chain_prev | 39.03 | 40.45 | 40.26 | -1.44 [-3.82, +0.95] | n.s. |
| base (v0.1) | 38.08 | 40.88 | 41.79 | - | |
| relmap_detail | 38.85 | 41.58 | 41.43 | -0.34 [-2.32, +1.66] | n.s. |

v0.1 base continued: epoch 3: 39.82, epoch 4: 40.96, epoch 5: 42.16

## B. Short protocol (1 epoch, 30 % of the training data, held-out DEV rows)

Every arm starts from the same stage-1 adapter. `nocue` = the same fine-tune without any added clue; `shuffled*` = the same clue text on random turns (control). EmoryNLP/MELD dev = their dev sets; IEMOCAP dev = every 5th training dialogue.

### emorynlp

| arm | dev wF1 | vs nocue [95% CI] | |
|---|---|---|---|
| lf-all | 43.95 | +2.57 [+0.81, +4.48] | better* |
| chain-all | 43.85 | +2.44 [+0.45, +4.45] | better* |
| lf-prev | 43.11 | +1.69 [+0.04, +3.24] | better* |
| chain-prev | 42.19 | +0.73 [-0.95, +2.52] | n.s. |
| relmap | 41.73 | +0.30 [-0.89, +1.52] | n.s. |
| nocue | 41.44 | - |  |
| desc-ctx-nop | 41.42 | -0.01 [-1.21, +1.31] | n.s. |
| desc-ctx-nov | 41.41 | -0.05 [-1.42, +1.26] | n.s. |
| relmap-cat | 41.38 | -0.07 [-1.17, +1.07] | n.s. |
| desc | 41.34 | -0.07 [-1.24, +1.09] | n.s. |
| shuffled-relmap | 41.24 | -0.19 [-1.26, +0.83] | n.s. |
| desc-ctx | 41.21 | -0.25 [-1.56, +1.02] | n.s. |
| desc-ctx-spk | 40.93 | -0.48 [-1.88, +0.88] | n.s. |
| desc-ctx-3c | 40.91 | -0.53 [-2.00, +0.75] | n.s. |
| shuffled-desc | 40.68 | -0.75 [-2.00, +0.35] | n.s. |
| shuffled | 40.53 | -0.94 [-2.28, +0.41] | n.s. |
| shuffled-desc-ctx | 40.53 | -0.95 [-2.22, +0.33] | n.s. |
| desc-ctx-nor | 40.48 | -0.93 [-2.32, +0.42] | n.s. |
| default | 40.42 | -1.03 [-2.40, +0.44] | n.s. |
| both-ctx | 39.69 | -1.74 [-3.30, -0.34] | worse* |
| relmap-desc | 39.61 | -1.86 [-3.22, -0.52] | worse* |

### meld

| arm | dev wF1 | vs nocue [95% CI] | |
|---|---|---|---|
| lf-all | 70.70 | +0.26 [-1.64, +2.15] | n.s. |
| nocue | 70.47 | - |  |
| relmap | 70.41 | -0.07 [-1.09, +0.88] | n.s. |
| shuffled-relmap | 70.39 | -0.06 [-1.06, +0.90] | n.s. |
| relmap-desc | 70.30 | -0.16 [-1.22, +0.87] | n.s. |
| relmap-cat | 70.24 | -0.24 [-1.16, +0.64] | n.s. |
| lf-prev | 70.17 | -0.33 [-2.05, +1.42] | n.s. |
| desc-ctx-nop | 70.08 | -0.39 [-1.69, +0.96] | n.s. |
| chain-all | 70.06 | -0.34 [-2.50, +1.60] | n.s. |
| shuffled-desc-ctx | 70.01 | -0.47 [-1.89, +0.88] | n.s. |
| desc-ctx-spk | 70.00 | -0.48 [-1.85, +0.91] | n.s. |
| default | 69.89 | -0.59 [-1.66, +0.50] | n.s. |
| desc-ctx-3c | 69.88 | -0.55 [-1.92, +0.83] | n.s. |
| both-ctx | 69.87 | -0.58 [-1.89, +0.69] | n.s. |
| desc-ctx-nov | 69.80 | -0.67 [-1.91, +0.50] | n.s. |
| desc | 69.78 | -0.70 [-1.93, +0.47] | n.s. |
| shuffled-desc | 69.70 | -0.75 [-2.08, +0.55] | n.s. |
| desc-ctx | 69.67 | -0.82 [-2.09, +0.49] | n.s. |
| desc-ctx-nor | 69.64 | -0.84 [-2.14, +0.42] | n.s. |
| shuffled | 69.52 | -0.95 [-2.04, +0.14] | n.s. |
| chain-prev | 67.31 | -3.19 [-5.34, -1.03] | worse* |

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

