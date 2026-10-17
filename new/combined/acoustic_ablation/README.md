# acoustic_ablation — which thresholds make good acoustic clues?

Two tiers. Only tier 1 has been run.

## Tier 1 — CPU screen (minutes) — `screen_thresholds.py`   [run; results in `results/`]
48 configs per dataset (8 percentile sets x {global, per-speaker percentiles, per-speaker z} x {5, 8 features}); a small
classifier (one-hot categories -> emotion) is trained on train and scored on a dev split. Test is never used.
IEMOCAP has no dev (its dev file equals test), so every 5th training dialogue is held out.
Finding: all sensible configs sit within noise (+-3 wF1) of the continuous-feature ceiling, so thresholds barely matter.
IEMOCAP carries strong acoustic signal (+28 wF1 over majority), MELD (+6) and EmoryNLP (+3.5) little.

## Tier 2 — LLM fine-tune ablation (GPU, **not run**) — `run_ablation.py`
```
python run_ablation.py --dataset iemocap --top_k 2 --percent 0.3          # arms: nocue, shuffled, default, top-2 screen configs
python run_ablation.py --dataset iemocap --configs "5c p10-30-70-90 | speaker_pct | 8feat"   # explicit configs
python summarize_ablation.py --dataset iemocap                            # ranks arms + paired bootstrap vs the controls
```
All arms start from the same stage-1 adapter, train on the training part of the ERC data for the same budget and are scored
on the held-out dev rows. `nocue` isolates plain extra training, `shuffled` isolates extra text. Adopt a config only if it
beats both controls with a 95 % interval above zero. ~1.2 h per arm for EmoryNLP/MELD at `--percent 0.3` (+ dev evaluation),
less for IEMOCAP. Needs the finished base stage 1 (`Stage1_aux_<D>`). `--dry_run` prints the commands.

Expect the LLM noise to be about +-1 wF1, so only large differences are resolvable; tier 1 suggests none are likely.

## Component ablation of the SpeechCueLLM-style clue (the justification for each design choice)
Full design (reference) = `description` text + `audio_context` tags + 5 features + 5-class percentile cutoffs + the dataset's default normalisation.
Each other arm changes ONE component; controls are no clue and the reference shuffled onto random turns:

| Arm | Change relative to the full design |
|---|---|
| `desc_ctx` / `shuffled_desc_ctx` | reference / its shuffled control |
| `desc` / `shuffled_desc` | - audio_context tags / its control |
| `desc_ctx_nov`, `desc_ctx_nop`, `desc_ctx_nor` | - volume features, - pitch features (this also removes the pitch-based tags), - speech-rate feature |
| `desc_ctx_3c` | 5-class -> 3-class percentile cutoffs (25/75) |
| `desc_ctx_spk` | global -> per-speaker percentiles |
| `both_ctx` | + the interpretive impression sentence |
| `default` (already run) | impression form instead of description |

IEMOCAP has no audio_context, so its preset uses `desc` as the reference (`desc_nov`, `desc_nop`, `desc_nor`, `desc_3c`, `desc_spk`, `both`).
```
python run_ablation.py --dataset emorynlp --configs --speechcue_ablation --percent 0.3     # nocue/shuffled/default are reused
python summarize_ablation.py --dataset emorynlp
```
The summary prints (1) every arm vs `nocue` and `shuffled`, (2) a **component table**: each arm minus the full design with a paired-bootstrap 95 %
interval and a reading ("change hurts -> component helps" / "no measurable difference"), (3) whether the full design beats both its controls.
A component is justified only where removing it hurts with the interval below zero. Cost (estimates from the measured arms): ~1.5 h per arm on
EmoryNLP/MELD (10 new arms ~15 h), ~30 min per arm on IEMOCAP (8 arms ~4 h). With ~1,100-1,300 dev rows the intervals are about +-2-3 wF1,
so only large effects will be resolvable; report "no measurable difference" as such.

## Testing the SpeechCueLLM-style forms (cheap check before a full-length run)
Each form is compared with `nocue` AND its own shuffled control (`shuffled_desc`, `shuffled_desc_ctx`, `shuffled_both`):
```
python run_ablation.py --dataset emorynlp --configs --forms desc shuffled_desc desc_ctx shuffled_desc_ctx --percent 0.3
python summarize_ablation.py --dataset emorynlp       # prints "FORM 'desc': ... -> ADOPT: python run_acoustic.py ..." or "not supported"
```
Already-finished arms (nocue, shuffled, default) are reused. ~1.5 h per new arm on EmoryNLP (4 arms ~6 h; estimate). The ablation is a
1-epoch / 30 % fine-tune, much shorter than SpeechCueLLM's 15-epoch LoRA, so a "not supported" here is weaker evidence than a full-length arm.

## Using a winning threshold config (steps)
1. **Run the ablation** (GPU; needs the base `Stage1_aux_<D>` adapter; IEMOCAP ~2.5-3 h for 5 arms, estimated):
   `cd acoustic_ablation && python run_ablation.py --dataset iemocap --top_k 2 --percent 0.3`
   (add `--tag _scene` to ablate on the scene variant; the thresholds themselves do not depend on the tag).
2. **Read the verdict:** `python summarize_ablation.py --dataset iemocap`. A config is adopted only if it beats BOTH `nocue` and
   `shuffled` with the 95 % interval above zero. The script prints either `VERDICT: ... To use it:` with the exact commands, or
   "keep the default recipe".
3. **Make the real clues with it** (full-length arms, comparable to the base run):
   `cd ../base+acoustic && python run_acoustic.py --dataset iemocap --config "<config name or slug>" --variants clue shuffled --erc_epochs 3`
   `cd ../base+chain && python run_chain.py --dataset iemocap --variants all --acoustic --acoustic_config "<config>" --erc_epochs 3`
   `--config` takes the name from `results/*_screen.csv` ("5c p10-30-70-90 | speaker_pct | 8feat") or the slug the summary prints.
   Data/run names carry the slug, so they never overwrite the default-recipe arms: `iemocap_coe_acoustic_clue_<slug>_erc`,
   `Stage2_acoustic_clue_<slug>_iemocap`.
4. **Compare all arms side by side:** `cd ../base+chain && python analyze_chain.py --dataset iemocap`
   (base, default-recipe clue, config clue, shuffled controls, chain arms). Confirm the config also wins at full length: the ablation
   is a short fine-tune, so a config can win there and not at 3 epochs.
Notes: `8feat` configs add one sentence about duration / pitch range / voice quality to the clue text (the original template
only covers pitch, volume and speech rate); `5feat` reproduces the original text exactly. The ablation uses the IEMOCAP dev
hold-out (every 5th training dialogue), never the test set. MELD/EmoryNLP carry little acoustic signal; a config found on IEMOCAP is
not validated for them.

## Files
`configs.py` (parse/name/shortlist configs) | `build_ablation_data.py` | `run_ablation.py` | `summarize_ablation.py` | `screen_thresholds.py`
Shared code lives in `../base+acoustic/`.
