# CoE (Clue of Emotion) — combined pipeline

Re-implementation of *CoE: A Clue of Emotion Framework for Emotion Recognition in Conversations* (ACL 2025) for a single
12 GB GPU (RTX 3060, 4-bit QLoRA, Mistral-7B-Instruct-v0.3). Targets the paper's **Strategy B (Mixed Auxiliary)**:
STEeR rationales -> joint auxiliary training (speaker-ID + role-play + STEeR) -> separate ERC stage.
Datasets: EmoryNLP, MELD, IEMOCAP.

**Status: pipeline implemented and smoke-tested stage by stage on tiny slices. No full training run has been done yet,
so there are no F1 results to report.**

## Run
```
cd base
python generate_scenes.py --dataset emorynlp      # synthetic scenes (also: meld). IEMOCAP has no scene clue.
python run_strategy_b.py --dataset emorynlp --steer_epochs 4 --aux_epochs 2 --erc_epochs 5
#   --tag _noscene / _nopersona  for the clue ablations;  --dataset meld | iemocap
```
`run_strategy_b.py` is resumable: build data -> seed rationales -> STEeR loop -> rebuild aux data with the accepted
rationales -> stage 1 (aux) -> stage 2 (ERC from the stage-1 adapter, evaluated on the test set).

## Layout (`base/`)
- `coe_clues.py` per-dataset config (labels, personas, main speakers), dialogue reconstruction, scene cache
- `build_coe_data.py` builds `Truebaseline/data/{dataset}_coe{tag}_{aux,erc,mixed,steer}` in the paper's Appendix-E prompt layout
- `generate_scenes.py` / `generate_seed_rationales.py` local-Mistral scene and seed-rationale generation
- `run_local_steer.py`, `run_local_combined.py`, `run_strategy_b.py` launchers; `Truebaseline/src/` DeepSpeed trainers
- `source_data/` ERC-only rows of the standard splits; `Truebaseline/data/*_scenes.json`, `*_coe_steer/trainbegin.json` are the generated scenes / seeds
- Built datasets and `outputs/` are not committed; `build_coe_data.py` regenerates the datasets.

## Deviations from the paper (hardware-driven; results are not directly comparable to Table 1)
- Mistral-7B-Instruct-v0.3, 4-bit QLoRA, LoRA r=8, batch 2 x accumulation 8 (paper: Mistral-7B-v0.1, fp16, r=16).
- **Scenes are synthetic** (local Mistral from the first 3 utterances, setting only); mostly generic ("Apartment"). Paper used frontier models.
- **STEeR seeds** are written by local Mistral shown the gold label (STaR-style rationalisation), not GPT-4o. Some contain invented details.
- IEMOCAP: no persona/scene (as in the paper); the speaker-ID task is skipped (F/M alternation carries no persona signal).
- Appendix-B filtering (drop first utterance, six main speakers) is applied to the auxiliary tasks only; ERC train/test use the standard splits.
- Model selection uses the test set (as the official code does). Single run, no 3-seed averaging.
