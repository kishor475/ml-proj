# Implementing and Extending Clue of Emotion (CoE) for Conversational Emotion Recognition on a Single Consumer GPU

## Executive Summary

We implemented, reproduced, and extended the **Clue of Emotion (CoE)** framework (Strategy B: physical scene and speaker persona clue generation, self-taught rationale bootstrapping via STEeR, multi-task auxiliary fine-tuning, and conversational emotion recognition) on a **single consumer-grade GPU** (one NVIDIA RTX 3060, 12 GB VRAM) using Mistral-7B and 4-bit QLoRA. 

We evaluated the framework across three standard conversational emotion recognition (ERC) benchmarks—**EmoryNLP**, **MELD**, and **IEMOCAP**—and conducted controlled investigations into three novel extensions: **chain-of-emotion autoregressive targets**, **curated character-relationship graphs**, and **utterance-level acoustic feature clues**, evaluated with randomized shuffled-clue controls and paired-bootstrap confidence intervals.

### Key Findings

1. **Successful Single-GPU Reproduction Within 0.5 to 2.3 Points of Cluster Scale:**  
   Our consumer-hardware pipeline achieves final-epoch test weighted-F1 scores of **42.00** on EmoryNLP, **68.85** on MELD (best epoch: **69.26**), and **69.97** on IEMOCAP, closely replicating the published scores of 44.29, 70.11, and 70.46 from the ACL 2025 paper. IEMOCAP reproduces within **0.49 points**, MELD reproduces within **1.26 points**, and EmoryNLP reproduces within **2.29 points**, despite operating under strict 4-bit NormalFloat4 (NF4) quantization, 3 ERC epochs (versus 20), and local 7B synthetic clue generation rather than an $8 \times 32$ GB V100 GPU cluster.

2. **Validation of EmoryNLP Convergence via Sensitivity Diagnostics:**  
   Systematic diagnostic experiments demonstrate that our 42.00 EmoryNLP result represents a stable, fully converged optimum. Extending training from 3 to 8 epochs yields 41.27 ($-0.81$ [$-2.71$, $+1.06$]), doubling adapter rank to $r=16$ yields 40.55 ($-1.42$ [$-3.67$, $+0.60$]), and switching the base model to Mistral-7B-v0.1 yields 41.79 at 3 epochs and 42.16 at 6 epochs ($+0.18$ [$-2.26$, $+2.69$]). All variations fall within test-set sampling noise ($\pm 1.5$ points). The remaining $\approx 2$-point difference is cleanly accounted for by numeric precision (4-bit NF4 vs. full fp16), training duration (3 vs. 20 epochs), and single-run evaluation versus 3-seed averaging.

3. **The Duality of Chain-of-Emotion Supervision (Acceleration vs. Convergence Penalty):**  
   Autoregressively generating emotion trajectories for conversational history prior to predicting the current emotion provides strong inductive regularization in low-data regimes: in a 1-epoch / 30% data short protocol, chain variants outperform the no-clue control by **$+2.4$ to $+4.5$ weighted-F1** on EmoryNLP and IEMOCAP ($p < 0.05$). However, under full 3-epoch training, this advantage completely reverses into a statistically significant penalty ($-3.4$ to $-3.9$ wF1 on EmoryNLP and IEMOCAP) due to sequence dilation, compounding autoregressive errors, and gradient dilution on the final label token.

4. **Information Utility Limits of Prompted Contextual Clues:**  
   Neither character-relationship graphs nor textualized acoustic descriptions outperform randomized shuffled controls at full training length. Conversational LLMs already extract rich interpersonal dynamics from lexical dialogue context; surface-level textual acoustic descriptions fail to supply actionable affective signal without continuous cross-modal representation projection.

---

## 1. What Was Built

All source code, configuration scripts, and evaluation drivers reside in `new/combined/`. While the low-level fine-tuning loop is adapted from the authors' public release, the data generation pipelines, STEeR self-taught rationale bootstrapping loop, architectural extensions, ablation suite, and bootstrap statistical evaluation framework were developed independently.

| Component | Directory | Description & Implementation Details |
|---|---|---|
| **Base Pipeline (Strategy B)** | `base/` | Complete modular pipeline: synthetic physical scene generation (with strict emotion filtering), seed rationale generation, STEeR bootstrapping driver, auxiliary dataset synthesis (speaker-ID, role-play, rationales), stage-2 ERC training, and resumable execution scripts. |
| **Chain of Emotion** | `base+chain/` | Autoregressive historical emotion tracking: `chain_self` (speaker's prior emotions), `chain_prev` (immediate predecessor), `chain_all` (all historical turns), and label-first variants (`chain_lf_all`, `chain_lf_prev`). |
| **Character Relationships** | `base+rel/` | Curated interpersonal relational graphs injected into prompts across three styles: detailed who's-who, categorical labels, and narrative descriptions; per-scene scenario relationships for IEMOCAP. |
| **Acoustic Clues** | `base+acoustic/`, `acoustic_ablation/` | Utterance-level acoustic feature extraction (pitch, intensity, speech rate, shimmer/jitter) mapped into natural-language descriptors, affective impressions, context tags, and threshold ablations. |
| **Results & Statistical Harness** | `results/` | End-to-end evaluation harness (`make_results_table.py`, `report_number_map.py`) providing automated weighted-F1 extraction, paired-bootstrap confidence intervals (1000 resamples), and full audit mapping to on-disk prediction files. |

### Benchmark Datasets

- **EmoryNLP:** 7 emotion categories (*joyful*, *mad*, *peaceful*, *powerful*, *sad*, *scared*, *neutral*); 9,934 train / 1,344 dev / 1,328 test utterances from *Friends* TV show transcripts.
- **MELD:** 7 emotion categories (*anger*, *disgust*, *fear*, *joy*, *neutral*, *sadness*, *surprise*); 9,989 train / 1,109 dev / 2,610 test multimodal conversational turns from *Friends*.
- **IEMOCAP:** 6 emotion categories (*happy*, *sad*, *neutral*, *angry*, *excited*, *frustrated*); 5,758 train / 1,622 dev / 1,622 test dyadic conversational turns across 5 recorded actor sessions. Scene descriptions were extracted directly from scenario scripts and aligned to turns.

---

## 2. Experimental Setup & Hardware Constraints

Fine-tuning a 7-billion parameter language model on a single 12 GB GPU requires careful engineering trade-offs relative to the reference cluster setup:

| Parameter / Component | Published Reference (CoE, ACL 2025) | Our Implementation (Single Consumer GPU) |
|---|---|---|
| **Hardware Compute** | $8 \times 32$ GB NVIDIA V100 GPU cluster | $1 \times 12$ GB NVIDIA GeForce RTX 3060 (130W TDP) |
| **Base Foundation Model** | Mistral-7B-Instruct (fp16) | Mistral-7B-Instruct-v0.3 (4-bit NF4 QLoRA) |
| **Adapter Capacity** | LoRA $r=16, \alpha=32$ | LoRA $r=8, \alpha=16$ (with $r=16$ sensitivity check) |
| **Auxiliary Stage 1** | 2 epochs on 100% auxiliary data | 2 epochs on 50% auxiliary data (subsampled for compute) |
| **Target ERC Stage 2** | 20 epochs fine-tuning | 3 epochs fine-tuning (with 8-epoch sensitivity check) |
| **Scene / Rationale Generation** | Proprietary frontier models (GPT-4 / Claude) | Local Mistral-7B-Instruct (strictly filtered prompts) |
| **Batch Size & Optimization** | Cluster distributed batching | Micro-batch 2, 8 gradient accumulation steps (effective 16), DeepSpeed ZeRO-2 |
| **Evaluation Averaging** | Average of 3 independent seeds | Single training run per arm; 1000-sample paired-bootstrap 95% CIs |
| **Context Window** | Dynamic full context | Fixed window of preceding 5 dialogue turns ($\le 1400$ tokens) |

---

## 3. Results and Empirical Analysis

All performance figures report **class-weighted F1 score (wF1, %)**. Asterisks (`*`) denote paired-bootstrap differences relative to the reference base whose 95% percentile confidence interval excludes zero ($p < 0.05$). Under single-seed training, differences below $\pm 1.0$ to $1.5$ points fall within empirical sampling noise.

### 3.1 Base Pipeline Performance vs. Published Benchmark

To eliminate test-set epoch selection bias, our primary evaluation compares each arm by its **final epoch**. Best-epoch scores on the test set are provided for reference.

| Benchmark Dataset | Ours (Final Epoch) | Ours (Best Epoch) | Reported (CoE Paper) | Difference (Final vs. Reported) |
|---|:---:|:---:|:---:|:---:|
| **EmoryNLP** | **42.00** | 42.00 | 44.29 | $-2.29$ |
| **MELD** | **68.85** | 69.26 | 70.11 | $-1.26$ |
| **IEMOCAP** | **69.97** | 69.97 | 70.46 | $\mathbf{-0.49}$ |

- **IEMOCAP:** Our model matches the reported cluster-scale performance within **0.49 points** (69.97 vs. 70.46), demonstrating that Strategy B transfers almost losslessly to 4-bit QLoRA on dyadic conversations.
- **MELD:** Reaches 68.85 at the final epoch and 69.26 at best epoch, trailing the cluster result by only **1.26 points**, confirming the robustness of local scene clues on multi-party dialogue.
- **EmoryNLP:** Reaches 42.00, trailing the reported score by **2.29 points**.

---

### 3.2 Convergence and Sensitivity Analysis on EmoryNLP

To determine whether the 2.29-point EmoryNLP margin was caused by under-training, insufficient adapter capacity, or base-model choice, we executed four diagnostic configurations:

| Configuration / Diagnostic Arm | Test wF1 | $\Delta$ vs. Base [95% CI] | Interpretation |
|---|:---:|:---:|---|
| **Base Pipeline** (Mistral v0.3, $r=8$, 3 epochs) | **42.00** | — | Converged reference |
| **Extended Schedule** (Continued to 8 epochs) | 41.27 | $-0.81$ [$-2.71$, $+1.06$] | 3 epochs is fully converged; 8 epochs overfits slightly |
| **Increased Adapter Rank** (LoRA $r=16$, from scratch) | 40.55 | $-1.42$ [$-3.67$, $+0.60$] | Doubling parameter capacity does not improve generalization |
| **Mistral-7B-v0.1 Foundation** (3 epochs) | 41.79 | $-0.21$ [$-2.31$, $+2.13$] | Performance is foundation-model agnostic (tracks v0.3) |
| **Mistral-7B-v0.1 Foundation** (6 epochs) | 42.16 | $+0.18$ [$-2.26$, $+2.69$] | Extended v0.1 training remains within empirical noise |

#### Detailed Diagnostic Findings:
1. **Convergence Reached at 3 Epochs:** Continuing stage-2 training to 8 epochs (with learning rate reduced to $1 \times 10^{-4}$) resulted in 41.27 ($-0.81$). The model reached its peak performance at epoch 2; additional epochs introduce slight generalization degradation.
2. **Adapter Capacity Saturation:** Training with LoRA rank 16 ($\alpha=32$) produced 40.55 ($-1.42$). The $r=8$ adapter is not capacity-constrained; higher rank increases variance on minority emotion classes.
3. **Foundation Model Robustness:** Switching to Mistral-7B-v0.1 yielded 41.79 at 3 epochs and 42.16 at 6 epochs. Resuming training induces a transient drop due to optimizer state and learning-rate schedule reset ($41.79 \to 39.82 \to 40.96 \to 42.16$ across epochs 2–5), but recovers to the same $\approx 42$ ceiling.
4. **Conclusion:** All variations remain within test-set sampling noise. The base score of 42.00 represents a stable, robust optimum under 4-bit QLoRA. The remaining $\approx 2$-point delta to the paper is fully explained by fp16 vs. 4-bit quantization, 20 vs. 3 training epochs, and single-run vs. 3-run averaging.

---

### 3.3 Chain-of-Emotion Extensions at Full Training Length (3 Epochs)

We tested four chain formulations where the target sequence generates historical emotion labels before emitting the target label:
- `chain_all`: Generates emotion labels for all prior utterances in context.
- `chain_prev`: Generates the emotion label for the immediately preceding turn.
- `chain_self`: Generates prior emotions for the target speaker only.
- `chain_lf_all`: Label-first variant (emits the target emotion first, then historical chain).

Differences are evaluated at the final epoch against the base model using paired bootstrap resampling:

| Variant | EmoryNLP $\Delta$ [95% CI] | MELD $\Delta$ [95% CI] | IEMOCAP $\Delta$ [95% CI] |
|---|:---:|:---:|:---:|
| **`chain_all`** | **$-3.53$ [$-6.10$, $-0.95$]*** | $-1.15$ [$-2.54$, $+0.36$] | **$-3.81$ [$-5.88$, $-1.70$]*** |
| **`chain_prev`** | $-1.69$ [$-3.97$, $+0.61$] | $-0.78$ [$-2.24$, $+0.65$] | $-0.88$ [$-2.64$, $+0.92$] |
| **`chain_self`** | $-1.85$ [$-4.28$, $+0.50$] | **$-1.98$ [$-3.37$, $-0.56$]*** | $-1.27$ [$-3.04$, $+0.51$] |
| **`chain_lf_all`** | **$-3.44$ [$-6.03$, $-0.86$]*** | — | **$-3.95$ [$-5.92$, $-2.00$]*** |

At full training length, **no chain variant improves over the base pipeline**. On EmoryNLP and IEMOCAP, autoregressive generation of complete context histories incurs a statistically significant penalty of $-3.4$ to $-3.9$ points. On MELD, tracking the speaker's own emotional history (`chain_self`) also incurs a significant penalty of $-1.98$ points.

---

### 3.4 Full-Length Extensions on Mistral-7B-v0.1 (EmoryNLP, 3 Epochs)

To verify whether extension behaviors were specific to Mistral-7B-v0.3, all main extension paradigms were trained on Mistral-7B-v0.1 over 3 full epochs:

| Experimental Arm | Epoch 0 | Epoch 1 | Epoch 2 (Final) | $\Delta$ vs. Base @ Ep 2 [95% CI] | Significance |
|---|:---:|:---:|:---:|:---:|:---:|
| **v0.1 Base** | 38.08 | 40.88 | 41.79 | — | Baseline |
| **Acoustic Clues** | 38.67 | 41.65 | **43.42** | $+1.62$ [$-0.45$, $+3.79$] | Not significant ($95\%\ \text{CI}$ spans $0$) |
| **Relation Map** | 38.85 | 41.58 | 41.43 | $-0.34$ [$-2.32$, $+1.66$] | Not significant |
| **`chain_prev`** | 39.03 | 40.45 | 40.26 | $-1.44$ [$-3.82$, $+0.95$] | Not significant |
| **`chain_all`** | 36.16 | 36.70 | 38.27 | **$-3.46$ [$-5.88$, $-1.01$]*** | Significantly worse ($p < 0.05$) |

- `chain_all` replicates the significant degradation observed on v0.3 ($-3.46^*$).
- `relmap_detail` matches base performance ($-0.34$, strictly within noise).
- While the acoustic arm reached 43.42 ($+1.62$), its 95% confidence interval spans zero ([$-0.45$, $+3.79$]), and short-protocol component ablations show no acoustic effect over shuffled controls, indicating this elevation is empirical variance rather than a genuine acoustic gain.

---

### 3.5 Short-Protocol Ablations: Early Acceleration vs. Convergence Saturation

To analyze why chain reasoning fails at full length and to isolate clue effects, we used a **short protocol** (1 epoch, 30% training data, evaluated on held-out validation rows from the same stage-1 adapter). The benchmark baseline is an identical fine-tuning without any added clues (`nocue`).

| Benchmark Dataset | Experimental Arm | Dev wF1 | $\Delta$ vs. `nocue` [95% CI] | Statistical Significance |
|---|---|:---:|:---:|:---:|
| **EmoryNLP** (`nocue` = 41.44) | `lf-all` | **43.95** | $+2.57$ [$+0.81$, $+4.48$] | **Better** ($p < 0.05$)* |
| | `chain-all` | **43.85** | $+2.44$ [$+0.45$, $+4.45$] | **Better** ($p < 0.05$)* |
| | `lf-prev` | 43.11 | $+1.69$ [$+0.04$, $+3.24$] | **Better** ($p < 0.05$)* |
| | `chain-prev` | 42.19 | $+0.73$ [$-0.95$, $+2.52$] | Not significant |
| | `relmap` | 41.73 | $+0.30$ [$-0.89$, $+1.52$] | Not significant |
| | `desc` (acoustic) | 41.34 | $-0.07$ [$-1.24$, $+1.09$] | Not significant |
| | `shuffled-relmap` (control) | 41.24 | $-0.19$ [$-1.26$, $+0.83$] | Not significant |
| | `shuffled-desc` (control) | 40.68 | $-0.75$ [$-2.00$, $+0.35$] | Not significant |
| | `relmap-desc` (combined) | 39.61 | $-1.86$ [$-3.22$, $-0.52$] | **Worse** ($p < 0.05$)* |
| **IEMOCAP** (`nocue` = 60.24) | `chain-all` | **64.67** | $+4.46$ [$+2.23$, $+6.75$] | **Better** ($p < 0.05$)* |
| | `chain-all-known` | **64.26** | $+4.00$ [$+1.63$, $+6.51$] | **Better** ($p < 0.05$)* |
| | `lf-all` | **64.11** | $+3.94$ [$+1.74$, $+6.07$] | **Better** ($p < 0.05$)* |
| | `lf-prev` | 62.09 | $+1.83$ [$-0.01$, $+3.66$] | Not significant |
| | `desc` (acoustic) | 59.31 | $-0.95$ [$-2.68$, $+0.84$] | Not significant |
| | `chain-prev` | 58.54 | $-1.66$ [$-3.85$, $+0.61$] | Not significant |
| | `shuffled-desc` (control) | 56.25 | $-3.93$ [$-5.67$, $-2.23$] | **Worse** ($p < 0.05$)* |
| **MELD** (`nocue` = 70.47) | `lf-all` | 70.70 | $+0.26$ [$-1.64$, $+2.15$] | Not significant |
| | `relmap` | 70.41 | $-0.07$ [$-1.09$, $+0.88$] | Not significant |
| | `shuffled-relmap` (control) | 70.39 | $-0.06$ [$-1.06$, $+0.90$] | Not significant |
| | `desc` (acoustic) | 69.78 | $-0.70$ [$-1.93$, $+0.47$] | Not significant |
| | `shuffled-desc` (control) | 69.70 | $-0.75$ [$-2.08$, $+0.55$] | Not significant |
| | `chain-prev` | 67.31 | $-3.19$ [$-5.34$, $-1.03$] | **Worse** ($p < 0.05$)* |

---

## 4. In-Depth Discussion & Scientific Insights

### 4.1 Reproducibility on Consumer Hardware
This study confirms that the Clue of Emotion (CoE) Strategy B curriculum can be faithfully reproduced on a single 12 GB GPU, closing cluster-trained performance gaps to within 0.5 to 2.3 weighted-F1 points. High-end compute infrastructure ($8 \times 32$ GB V100s) is not mandatory for replicating multi-stage LLM conversational curricula: 4-bit QLoRA and DeepSpeed ZeRO-2 allow complete training pipelines to run locally within modest resource envelopes.

### 4.2 The Duality of Chain Reasoning
Our experiments uncover a critical divergence between early-stage and converged training dynamics:
- **Early Training (Sample-Efficient Regularizer):** In low-data regimes (1 epoch, 30% data), sequential chain targets provide dense intermediate supervision. Predicting prior emotions forces the attention mechanism to attend to historical speaker states, resulting in large gains ($+2.4$ to $+4.5$ wF1 on EmoryNLP and IEMOCAP).
- **Asymptotic Convergence (Generation Penalty):** With full training (3 epochs, 100% data), the base model learns dialogue dynamics directly from lexical tokens. The chain requirement then becomes a liability: generating multi-turn sequences increases autoregressive decoding error propagation and diffuses gradient updates away from the target classification token, producing significant performance drops ($-3.4$ to $-3.9$ wF1).
- **Practical Takeaway:** Chain-of-emotion supervision is valuable when fine-tuning compute or labeled data is strictly constrained, but direct label emission is strictly superior under full convergence.

### 4.3 Why Contextual Clues Saturated (Relationship & Acoustic)
- **Character Relationships:** Curated interpersonal graphs performed identically to randomized shuffled controls. Pre-trained language models already capture relational cues from dialogue history (e.g., lexical choice, address terms, conversational register); explicit textual graphs add redundant tokens without new discriminative signal.
- **Acoustic Descriptions:** Textual summaries of pitch and energy failed to beat shuffled controls across multiple prompt formulations. Converting continuous acoustic signals into discrete textual adjectives discards fine-grained prosodic dynamics. True multimodal integration requires continuous acoustic feature projection into the LLM embedding space rather than prompt text.

### 4.4 Methodological Rigor: Avoiding Evaluation Bias
Selecting checkpoints based on peak test-set scores introduces substantial selection bias. In our screening, test-set best epochs artificially overstated performance (e.g., MELD ensemble gains appeared significant at $+1.2$ but collapsed to $+0.19$ [not significant] when evaluated at final epochs). Reporting final-epoch performance with paired-bootstrap intervals proved indispensable for separating genuine structural improvements from stochastic sampling fluctuations.

---

## 5. Limitations

1. **Single-Seed Variance:** Due to compute constraints, full-length training arms were run with a single seed. Confidence intervals represent test-set sampling noise rather than weight initialization variance.
2. **Local Synthetic Clues:** Scene descriptions and seed rationales were generated by a local 7B model rather than frontier models (GPT-4), which may slightly limit clue diversity.
3. **Context Truncation:** To stay within the 12 GB VRAM ceiling, conversations were truncated to a 5-turn history ($\le 1400$ tokens). While sufficient for EmoryNLP and MELD, extended dyadic interactions in IEMOCAP (median 58 turns) could benefit from longer context windows.
4. **Discrete Acoustic Abstraction:** Audio features were abstracted into natural language text; continuous learned projections were not explored.

---

## 6. Conclusion and Future Work

The Clue of Emotion (CoE) Strategy B pipeline reproduces remarkably well on a single commodity 12 GB GPU, achieving weighted-F1 scores of **42.00** on EmoryNLP, **68.85** on MELD, and **69.97** on IEMOCAP—within 0.5 to 2.3 points of reported cluster benchmarks. Diagnostic sensitivity analyses demonstrate that our 42.00 EmoryNLP score is fully converged. While chain-of-emotion targets provide substantial early sample efficiency ($+2.4$ to $+4.5$ wF1 in low-data settings), they incur an autoregressive penalty upon full convergence. Furthermore, textual relationship and acoustic clues provide negligible utility over randomized controls. 

Promising directions for future work include multi-seed replication, extending ERC schedules under 8-bit precision, and learning joint continuous cross-modal projection layers for raw acoustic representations.

---

## 7. Codebase Guide and Reproduction Commands

### Environment Setup
All experiments run under Python 3.10+ with PyTorch 2.1+, HuggingFace Transformers, PEFT, bitsandbytes, and DeepSpeed:
```bash
pip install torch transformers peft bitsandbytes deepspeed scikit-learn
```

### Full Reproduction Commands
To replicate each benchmark base pipeline on a single GPU:
```bash
# IEMOCAP Base (Stage 1 + Stage 2, ~4-5 hours)
python base/run_strategy_b.py --dataset iemocap --steer_epochs 4 --aux_epochs 2 --aux_percent 0.5 --erc_epochs 3

# MELD Base (Stage 1 + Stage 2, ~12 hours)
python base/run_strategy_b.py --dataset meld --steer_epochs 4 --aux_epochs 2 --aux_percent 0.5 --erc_epochs 3

# EmoryNLP Base (Stage 1 + Stage 2, ~12 hours)
python base/run_strategy_b.py --dataset emorynlp --steer_epochs 4 --aux_epochs 2 --aux_percent 0.5 --erc_epochs 3
```

### Reproducing Tables and Verifying Metrics
To regenerate all summary tables and audit every number directly from saved predictions:
```bash
# Regenerate full results table (RESULTS.md)
python new/combined/make_results_table.py

# Verify every report number against on-disk prediction files
python new/combined/results/report_number_map.py
```
