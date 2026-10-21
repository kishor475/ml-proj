> **Status: implemented and tested, deliberately NOT run at full length.** The acoustic ablation (`../acoustic_ablation`, all three datasets,
> controls + component ablation) found no benefit over the no-clue base, so the final system does not use acoustic clues. This folder is kept as a
> working implementation (including SpeechCueLLM's description / audio_context forms) for anyone who wants to try it at full training length.

# base+acoustic — CoE ERC with acoustic clues

Adds a `### Acoustic Clues:` sentence (SpeechCueLLM-style impression, e.g. "The target speaker likely uses a lower pitch ...")
to the ERC prompts of the base pipeline. Everything else (stage-1 adapter, STEeR rationales, hyper-parameters) is reused,
so a clue run differs from the base run **only in the prompt**.

## Run (after the base run `base/run_strategy_b.py --dataset D` has finished)
```
cd combined/base+acoustic
python run_acoustic.py --dataset iemocap --variants clue shuffled --erc_epochs 3     # IEMOCAP first: strongest acoustic signal
python run_acoustic.py --dataset meld    --variants clue shuffled --erc_epochs 3
python run_acoustic.py --dataset emorynlp --variants clue shuffled --erc_epochs 3    # expect little (25 % of turns have no audio)
```
Add `--tag _scene` for the IEMOCAP transcript-scene variant. Add `--config "<threshold config>"` to use thresholds chosen by
`acoustic_ablation` instead of the original recipe (see that README). `--erc_epochs` must equal the base run's. Re-run to resume.
Each arm trains stage 2 only (~4 h per epoch for EmoryNLP/MELD). Compare, per dataset, the weighted-F1 of:
`Stage2_erc_<D>` (base, no clue)  vs  `Stage2_acoustic_clue_<D>`  vs  `Stage2_acoustic_shuffled_<D>` (control).
The clue only counts if it beats the shuffled control, not just the base.

## SpeechCueLLM-style forms (`--form`, `--audio_context`)
The reference implementation (`sp/SpeechCueLLM`) has two kinds of audio text and we first used only the second:
- `--form description` — **the paper's headline setting**: compact, factual, no emotion words, e.g.
  `Target speech characteristics: moderate volume with low variation, low pitch with moderate variation, moderate speaking rate.`
  (reproduces SpeechCueLLM's own MELD `description` column on 100 % of rows; ~22 tokens vs ~36 for the impression)
- `--form impression` (default, what the earlier arms used) — interpretive sentence ("might indicate excitement ...")
- `--form both` — description followed by impression
- `--audio_context` — SpeechCueLLM's `audio_context`: the last 3 context utterances and the current one get a short tag such as
  `(low pitch with high variation)` (not available for IEMOCAP: its history contains unlabelled utterances).
As in SpeechCueLLM, no clue is added when the clip is <= 0.5 s or no audio exists (EmoryNLP: ~26 % of turns). Data/run names carry the form:
`emorynlp_coe_acoustic_clue_desc_ctx_erc`, `Stage2_acoustic_clue_desc_ctx_emorynlp`. The default (impression) output is byte-identical to before.
```
python run_acoustic.py --dataset emorynlp --form description --variants clue shuffled --erc_epochs 3
python run_acoustic.py --dataset emorynlp --form description --audio_context --variants clue shuffled --erc_epochs 3
```
**Training budget differs from the paper:** SpeechCueLLM reports its gains with LoRA for 15 epochs (lr 3e-4, history window 12, LLaMA-3-8B); our
arms use the base settings (3 epochs, window 5, Mistral-7B 4-bit). They did not evaluate EmoryNLP.

## Files
- `acoustic_clues.py`  alignment of audio features to our turns (MELD / EmoryNLP / IEMOCAP), percentile thresholds, impression text
- `acoustic_data.py`, `build_acoustic_data.py`  write `{D}_coe{tag}_acoustic_{clue|shuffled}_erc`
- `run_acoustic.py`  resumable driver (`--dry_run` prints the commands)

## Inputs (paths via env `ACOUSTIC_AUDIO_DIR`, `EMORYNLP_JSON_DIR`; defaults `ml_proj/audio`, `ml_proj/emotion-detection`)
Per-utterance raw features and the original EmoryNLP `emotion-detection-{trn,dev,tst}.json` (for the id -> audio file mapping).

## Verified
- Alignment: MELD 99.9-100 % of turns; EmoryNLP 100 % (75 % have audio); IEMOCAP exact (emotion code agrees on 7,379/7,379 turns).
- Default thresholds reproduce the original clues: EmoryNLP 99.8 %, MELD 96 % (per-speaker percentiles fitted on train);
  IEMOCAP's original recipe could not be recovered exactly (best 54 %), so per-gender percentiles fitted on train are used.
- Removing the clue block from any generated prompt gives back the base prompt exactly (70,632 prompts checked).
Thresholds are fitted on the training split only (the original scripts used the whole file).
