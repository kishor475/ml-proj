# combined — CoE pipeline, acoustic clues, chain of emotion

Everything here targets one 12 GB GPU (RTX 3060, 4-bit QLoRA, Mistral-7B-Instruct-v0.3) and three datasets:
`emorynlp`, `meld`, `iemocap`. Every experiment arm is **one command**, resumable, and comparable to the base run.

| Folder | What it is |
|---|---|
| `base/` | CoE Strategy B (paper re-implementation): persona + scene clues, STEeR rationales, auxiliary stage, ERC stage |
| `base+acoustic/` | the base ERC stage + an `### Acoustic Clues:` sentence per turn (+ a shuffled-clue control) |
| `base+chain/` | the base ERC stage + a chain of emotions (`Emotion Flow: ... / Current Emotion: ...`), 3 chain definitions |
| `base+rel/` | the base ERC stage + a `### Character Relationships:` who's-who block (named side characters; per-scene relationship for IEMOCAP) |
| `acoustic_ablation/` | which threshold set makes good acoustic clues (CPU screen: done; LLM ablation: written, not run) |

Per-folder READMEs explain the details. This file is the **run book**.

---------------------------------------------------------------------------------------------------------------------

## 0. Conventions (read once)

```
cd ~/ml_proj/combined
PY=~/ml_proj/.venv/bin/python        # always this interpreter
mkdir -p logs
```
- **Run one training job at a time** (single GPU). Two at once collide on the GPU, the output folders and the log. Check first:
  `pgrep -af "run_strategy_b|run_acoustic|run_chain|run_ablation"` should print nothing.
- **Everything is resumable.** If a run stops (crash, shutdown), run the *same command again*: finished stages are skipped and an
  interrupted training stage continues from its latest saved epoch adapter (optimizer state / LR schedule restart).
- **Arms are only comparable if the settings match the base run.** `--erc_epochs` and `--tag` must equal the base run's.
- **Order matters:** `base` first for a dataset; every other arm starts from that dataset's base **stage-1 adapter** and is compared
  to its base **stage-2** result.
- On another machine add `--model_path /path/to/Mistral-7B-Instruct-v0.3` to any `run_*.py` command.
- `--dry_run` on any `run_*.py` prints the commands without running anything.
- Outputs: `base/Truebaseline/outputs/<run name>/...` (per-epoch adapters `lora_epoch_N`, `best_checkpoint`, `final_checkpoint`,
  and `preds_for_eval_<epoch>_<F1>.text` — the F1 is in the file name). Test set F1 is printed per epoch in the log.

Monitor a run:
```
tr '\r' '\n' < logs/<name>.log | grep -E "STEeR EPOCH|Accepted|Loss|Weighted-F1|Traceback|Error" | tail -15
ls base/Truebaseline/outputs/                      # which stages exist
nvidia-smi
```

## 1. One-time preparation (already done on the main machine)

| Step | Command | Needed for |
|---|---|---|
| Scenes for the Friends datasets (synthetic, local Mistral) | `cd base && $PY generate_scenes.py --dataset emorynlp` and `--dataset meld` | base (EmoryNLP, MELD) |
| IEMOCAP per-turn scene ids from the original transcripts | `cd base && $PY align_iemocap_scenes.py --transcripts ~/ml_project` | IEMOCAP `--tag _scene` only |
| Seed rationales, datasets | automatic: `run_strategy_b.py` builds anything missing | base |
| Audio features + EmoryNLP JSONs | folders `~/ml_proj/audio` and `~/ml_proj/emotion-detection` (override with env `ACOUSTIC_AUDIO_DIR`, `EMORYNLP_JSON_DIR`) | acoustic arms |

## 2. The experiment matrix

Rows = datasets, columns = arms. Each cell is a command in section 3.

| Dataset | A. base | B. acoustic | C. chain (self / all / prev) | D. chain + acoustic | E. extras |
|---|---|---|---|---|---|
| IEMOCAP | A1 | B1 (strongest acoustic signal) | C1 | D1 | A-scene, B-scene, D-scene |
| MELD | A2 | B2 (small signal) | C2 | D2 | |
| EmoryNLP | A3 | B3 (little signal, 25 % of turns lack audio) | C3 | D3 | A-noscene (scene ablation) |

Arms inside a cell: B = `clue` + `shuffled` (control); C = `self`, `all`, `prev`; D = chain variant(s) on the acoustic prompts.
**A clue/chain result only counts relative to its control/base** (see section 4).

## 3. Commands

All commands run from `~/ml_proj/combined/<folder>`. `--steer_epochs 4 --aux_epochs 2 --aux_percent 0.5 --erc_epochs 3` are the
settings used for the main runs.

### A. Base (Strategy B)  — `base/`
```
cd ~/ml_proj/combined/base
$PY run_strategy_b.py --dataset iemocap  --steer_epochs 4 --aux_epochs 2 --aux_percent 0.5 --erc_epochs 3   # A1
$PY run_strategy_b.py --dataset meld     --steer_epochs 4 --aux_epochs 2 --aux_percent 0.5 --erc_epochs 3   # A2
$PY run_strategy_b.py --dataset emorynlp --steer_epochs 4 --aux_epochs 2 --aux_percent 0.5 --erc_epochs 3   # A3
```
Variants (separate output folders, same flags):
```
$PY run_strategy_b.py --dataset iemocap  --tag _scene   ...same flags...   # A-scene: IEMOCAP + transcript-derived scene (NOT paper-comparable: the scene predicts the emotion)
$PY run_strategy_b.py --dataset emorynlp --tag _noscene ...same flags...   # A-noscene: scene clue ablation (also _nopersona)
```
Stages run in order: seed rationales -> STEeR loop -> aux data rebuild -> stage 1 (auxiliary) -> stage 2 (ERC, scored on test each epoch).

### B. Acoustic clues  — `base+acoustic/`   (needs A for the same dataset/tag)
```
cd ~/ml_proj/combined/base+acoustic
$PY run_acoustic.py --dataset iemocap  --variants clue shuffled --erc_epochs 3     # B1
$PY run_acoustic.py --dataset meld     --variants clue shuffled --erc_epochs 3     # B2
$PY run_acoustic.py --dataset emorynlp --variants clue shuffled --erc_epochs 3     # B3
$PY run_acoustic.py --dataset iemocap  --tag _scene --variants clue shuffled --erc_epochs 3   # B-scene
```
Run names: `Stage2_acoustic_clue_<D>`, `Stage2_acoustic_shuffled_<D>`. Compare with `Stage2_erc_<D>` (base).

SpeechCueLLM-style variants (the reference repo in `sp/`; its headline form is the compact `description`, not our `impression`; optional `--audio_context`):
```
$PY run_acoustic.py --dataset emorynlp --form description --variants clue shuffled --erc_epochs 3
$PY run_acoustic.py --dataset emorynlp --form description --audio_context --variants clue shuffled --erc_epochs 3
```
**Component ablation to justify the design choices** (full design vs removing/changing one component at a time, with controls; ~15 h on EmoryNLP/MELD, ~4 h on IEMOCAP, estimates):
```
cd ~/ml_proj/combined/acoustic_ablation
$PY run_ablation.py --dataset emorynlp --configs --speechcue_ablation --percent 0.3
$PY summarize_ablation.py --dataset emorynlp          # arms vs controls + COMPONENT ABLATION table
```
Cheap check first (EmoryNLP, ~6 h, estimate): `cd ../acoustic_ablation && $PY run_ablation.py --dataset emorynlp --configs --forms desc shuffled_desc desc_ctx shuffled_desc_ctx --percent 0.3`, then `$PY summarize_ablation.py --dataset emorynlp`.
Run names: `Stage2_acoustic_clue_desc[_ctx]_<D>` / `Stage2_acoustic_shuffled_desc[_ctx]_<D>`.

### C. Chain of emotion  — `base+chain/`   (needs A)
```
cd ~/ml_proj/combined/base+chain
$PY run_chain.py --dataset iemocap  --variants self all prev --erc_epochs 3        # C1
$PY run_chain.py --dataset meld     --variants self all prev --erc_epochs 3        # C2
$PY run_chain.py --dataset emorynlp --variants self all prev --erc_epochs 3        # C3
```
Run names: `Stage2_chain_<variant>_<D>`. `self` = current speaker's earlier emotions, `all` = every history turn, `prev` = previous turn only.
To try one variant only: `--variants all`.

### D. Chain + acoustic (base+chain+acoustic)  — `base+chain/`   (needs A; builds the acoustic data itself)
```
cd ~/ml_proj/combined/base+chain
$PY run_chain.py --dataset iemocap  --variants all --acoustic --erc_epochs 3       # D1  (use the chain variant that won in C)
$PY run_chain.py --dataset meld     --variants all --acoustic --erc_epochs 3       # D2
$PY run_chain.py --dataset emorynlp --variants all --acoustic --erc_epochs 3       # D3
$PY run_chain.py --dataset iemocap  --tag _scene --variants all --acoustic --erc_epochs 3     # D-scene
```
Run names: `Stage2_chain_<variant>_acoustic_<D>`. Pick the variant after looking at C (or run all three: `--variants self all prev`).

### E. Acoustic threshold ablation  — `acoustic_ablation/`
CPU screen (minutes, **already run**, results in `acoustic_ablation/results/*_screen.csv`):
```
cd ~/ml_proj/combined/acoustic_ablation
$PY screen_thresholds.py --dataset iemocap          # also: meld, emorynlp
```
LLM ablation (GPU, **written but not run**; needs A for the dataset; ~1.2 h per arm for EmoryNLP/MELD at `--percent 0.3`, less for IEMOCAP):
```
$PY run_ablation.py --dataset iemocap --top_k 2 --percent 0.3        # arms: nocue, shuffled, default, top-2 screen configs
$PY summarize_ablation.py --dataset iemocap                          # ranks arms; paired bootstrap vs the nocue and shuffled controls
```
Explicit configs: `--configs "5c p10-30-70-90 | speaker_pct | 8feat" "3c p25-75 | global | 8feat"`.
The screen found that all sensible threshold sets are within noise of each other, so this ablation is optional.

**If the ablation finds a better threshold config** (the summary prints the exact commands when one beats both controls):
```
cd ~/ml_proj/combined/acoustic_ablation
$PY summarize_ablation.py --dataset iemocap          # -> "VERDICT: '<slug>' beats nocue and shuffled ..." or "keep the default recipe"
cd ../base+acoustic
$PY run_acoustic.py --dataset iemocap --config "5c p10-30-70-90 | speaker_pct | 8feat" --variants clue shuffled --erc_epochs 3
cd ../base+chain                                      # optional: chain + the chosen clues
$PY run_chain.py --dataset iemocap --variants all --acoustic --acoustic_config "5c p10-30-70-90 | speaker_pct | 8feat" --erc_epochs 3
$PY analyze_chain.py --dataset iemocap               # default-recipe arms and config arms side by side
```
`--config` accepts a name from `results/*_screen.csv` or the slug the summary prints; outputs carry the slug and never overwrite
the default-recipe arms. See `acoustic_ablation/README.md` for the full steps and caveats.

### F. Character relationships  — `base+rel/`   (needs A)
```
cd ~/ml_proj/combined/base+rel
$PY run_rel.py --dataset emorynlp --erc_epochs 3                    # F3  (28 % of turns get a block)
$PY run_rel.py --dataset meld --erc_epochs 3                        # F2  (17 % of turns get a block)
$PY run_rel.py --dataset iemocap --erc_epochs 3                     # F1  (per-scene relationship, 100 %; needs iemocap_turn_scenes.json)
$PY run_rel.py --dataset iemocap --tag _scene --erc_epochs 3        # F-scene
$PY run_rel.py --dataset meld --acoustic --erc_epochs 3             # relationships + acoustic clues
```
Run name: `Stage2_rel_<D>` (`Stage2_rel_acoustic_<D>`). All clue families stack with the chain through `run_chain.py`:
```
cd ~/ml_proj/combined/base+chain
$PY run_chain.py --dataset emorynlp --variants all --rel --erc_epochs 3              # chain + relationships
$PY run_chain.py --dataset emorynlp --variants all --rel --acoustic --erc_epochs 3   # chain + relationships + acoustic (all three)
```
Run names: `Stage2_chain_<variant>_rel_<D>`, `Stage2_chain_<variant>_acoustic_rel_<D>`.

## 4. Reading the results

```
cd ~/ml_proj/combined/base+chain
$PY analyze_chain.py --dataset iemocap          # also meld, emorynlp; add --tag _scene for the scene variant
```
It finds every finished arm (base, acoustic, acoustic-shuffled, chain-*, chain-*+acoustic) and prints, per arm: weighted-F1 and
accuracy of the final label, F1 on emotion-**shift** vs non-shift turns, the accuracy of the generated chain (chain arms), and a
paired-bootstrap difference to the base arm with a 95 % interval.

How to judge:
- **acoustic** counts only if it beats **acoustic-shuffled** (otherwise the gain is just extra text/extra training), not just base.
- **chain** counts if its interval vs base is above zero; look at the shift-turn column for *where* it helps.
- The bootstrap covers test-set sampling noise, **not training-seed variance** (one seed per arm). Treat differences under ~1 F1 as unproven.
- Selection of the best epoch/arm uses the test set (as the official code does); the acoustic *ablation* uses a held-out dev set.

## 5. Suggested order and cost (single 3060; "measured" from the EmoryNLP run, the rest estimated)

| Step | Approx. GPU time |
|---|---|
| A base, EmoryNLP | ~23 h (STEeR 4.7 h + stage 1 5.8 h + stage 2 ~3 x 4 h) — measured |
| A base, MELD | ~23 h (+ ~20 min test eval per epoch) |
| A base, IEMOCAP | ~8-10 h |
| one stage-2 arm (B / C / D), EmoryNLP or MELD | ~12-13 h (chain `all`: +1-2 h for longer generation at eval) |
| one stage-2 arm, IEMOCAP | ~4-5 h |

Cheapest informative path: **A1 -> B1 + C1 (all three chain variants) -> D1 -> analyze**, i.e. ~35 h on IEMOCAP, then repeat on MELD/EmoryNLP
if time allows. The full matrix is ~250-300 GPU-hours.

Run several arms unattended (one dataset end to end; a failed step does not stop the next, so check each log):
```
cd ~/ml_proj/combined && mkdir -p logs; PY=~/ml_proj/.venv/bin/python
nohup sh -c "
cd base            && $PY run_strategy_b.py --dataset iemocap --steer_epochs 4 --aux_epochs 2 --aux_percent 0.5 --erc_epochs 3 > ../logs/iemocap_base.log 2>&1
cd ../base+acoustic && $PY run_acoustic.py --dataset iemocap --variants clue shuffled --erc_epochs 3 > ../logs/iemocap_acoustic.log 2>&1
cd ../base+chain    && $PY run_chain.py --dataset iemocap --variants self all prev --erc_epochs 3 > ../logs/iemocap_chain.log 2>&1
cd ../base+chain    && $PY run_chain.py --dataset iemocap --variants all --acoustic --erc_epochs 3 > ../logs/iemocap_chain_acoustic.log 2>&1
" > /dev/null 2>&1 &
```

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| Machine shut down / run crashed | re-run the same command (resumes) |
| "stage-1 adapter missing" | the base run for that dataset/tag has not finished; run section A first |
| Two runs started by mistake | `ps -eo pid,etime,args \| grep -E "trainer\|run_"`, kill the newer run's driver, launcher and trainer PIDs |
| Out of GPU memory | another job is on the GPU (`nvidia-smi`); never run two |
| `ninja` / FusedAdam error | always launch with `$PY` (the runners add the venv's `bin` to PATH) |
| Stage 2 score looks low after epoch 0 | early epochs are usually lower (EmoryNLP base: 36.7 wF1 after epoch 0); judge by the best / last epoch |
| Chain run scores collapse | check `max_new_tokens` reached the trainer (`run_chain.py` sets 32/64/128 per variant); a too-short budget cuts the answer |

## 7. Deviations from the paper (hardware-driven; results are **not** comparable to the paper's Table 1)
- Mistral-7B-Instruct-v0.3, 4-bit QLoRA, LoRA r=8, batch 2 x accumulation 8, reduced auxiliary data (`--aux_percent 0.5`).
- Scenes (EmoryNLP, MELD) are synthetic and mostly generic; STEeR seed rationales come from the local model shown the gold label (no GPT-4o).
- IEMOCAP: no persona/scene (as in the paper); no speaker-ID task; the `_scene` variant is an extension, not the paper's setting.
- Single run per arm; model selection on the test set (as in the official code).


## 8. Running on a second machine (split the workload)
Every run is self-contained: it needs the code + data (`new/combined`), the Mistral model, the base **stage-1 adapter** of each dataset it trains
(and the base **stage-2 result** only for the final comparison), and the same Python environment. Keep the same paths on both machines
(`/home/user2/ml_proj/new/combined`, `/home/user2/Mistral-7B-Instruct-v0.3`, `/home/user2/ml_proj/.venv`) so no path needs changing.
Current split: **EmoryNLP + IEMOCAP on the main machine, MELD on the second one.** Never run the same dataset/arm on both machines.

### 8a. Send everything (run on the main machine; HOST = the other desktop)
```
HOST=user2@100.78.88.86
DS="meld"                                  # dataset(s) the other machine will run, space separated (e.g. "meld" or "meld emorynlp")
ssh-copy-id $HOST                          # once, if it asks for a password every time
ssh $HOST 'mkdir -p ~/ml_proj/new ~/ml_proj/wheels ~/.cache'

# 1. code + built data, without the large outputs (~1.7 GB)
rsync -avh --progress --exclude 'outputs/' --exclude '__pycache__' ~/ml_proj/new/combined/ $HOST:~/ml_proj/new/combined/

# 2. only the outputs those datasets need: the stage-1 adapter (to train from) and the stage-2 base result (to compare with) (~350 MB per dataset)
for d in $DS; do rsync -avh --progress --relative ~/ml_proj/./new/combined/base/Truebaseline/outputs/Stage1_aux_$d ~/ml_proj/./new/combined/base/Truebaseline/outputs/Stage2_erc_$d $HOST:~/ml_proj/; done

# 3. the model, skipping the duplicate consolidated file (14.5 GB instead of 28 GB)
rsync -avh --progress --exclude consolidated.safetensors ~/Mistral-7B-Instruct-v0.3/ $HOST:~/Mistral-7B-Instruct-v0.3/

# 4. offline deepspeed wheel + compiled FusedAdam cache (the cache only works on the same GPU architecture / torch / python)
rsync -avh ~/ml_proj/wheels/ $HOST:~/ml_proj/wheels/
rsync -avh ~/.cache/torch_extensions/ $HOST:~/.cache/torch_extensions/
```
Each rsync can be re-run safely; it only sends what is missing.

### 8b. Set up the environment (run on the other machine)
**The venv must use Python 3.12** (the pinned `torch==2.6.0+cu124` has no wheels for Python 3.13/3.14; pip then reports "No matching distribution" and
skips packages "that require a different python version"). Check first: `python3 --version`. If it is not 3.12, get 3.12 (option A needs sudo, option B does not):
```
# option A (Ubuntu with python3.12 in apt):   sudo apt install python3.12 python3.12-venv
#                                              python3.12 -m venv --clear ~/ml_proj/.venv
# option B (no sudo): uv installs a standalone Python 3.12
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv python install 3.12
$(uv python find 3.12) -m venv --clear ~/ml_proj/.venv          # --clear replaces a venv created with the wrong Python
```
then continue below (the `python3 -m venv` line is replaced by one of the above).

```
ssh user2@100.78.88.86
nvidia-smi                                                   # needs >= 12 GB VRAM and a driver that supports CUDA 12.4
python3 --version                                            # 3.12 to match; install python3.12-venv if the next line fails
python3 -m venv ~/ml_proj/.venv && source ~/ml_proj/.venv/bin/activate
pip install --upgrade pip
pip install -r ~/ml_proj/new/combined/requirements_freeze.txt --extra-index-url https://download.pytorch.org/whl/cu124
#   if deepspeed fails:  pip install ~/ml_proj/wheels/deepspeed-0.19.7-py3-none-any.whl   then re-run the line above
python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
~/ml_proj/.venv/bin/python --version                          # must say 3.12.x
PATH=~/ml_proj/.venv/bin:$PATH python -c "from deepspeed.ops.adam import FusedAdam;import torch;p=torch.nn.Parameter(torch.zeros(4,device='cuda'));FusedAdam([p]);print('FusedAdam OK')"
ln -s ~/ml_proj/new/combined ~/ml_proj/combined              # REQUIRED: the commands in this README start with  cd ~/ml_proj/combined/...
cd ~/ml_proj/combined/base+chain && ~/ml_proj/.venv/bin/python run_chain.py --dataset meld --variants all --erc_epochs 3 --dry_run
```
The dry run must print a `run_local_combined.py ...` command; "stage-1 adapter missing" means step 2 of 8a did not complete.
If FusedAdam fails to compile ("nvcc" / "CUDA_HOME" errors), the copied `torch_extensions` cache did not match that GPU: install the CUDA 12.4 toolkit there.

### 8c. Launch
Main machine (EmoryNLP + IEMOCAP chain; ~17 h IEMOCAP + ~40 h EmoryNLP, estimates):
```
cd ~/ml_proj/combined/base+chain; PY=~/ml_proj/.venv/bin/python; mkdir -p ../logs
nohup sh -c "
$PY run_chain.py --dataset iemocap  --variants all self prev --erc_epochs 3 > ../logs/chain_iemocap.log 2>&1
$PY run_chain.py --dataset emorynlp --variants all self prev --erc_epochs 3 > ../logs/chain_emorynlp.log 2>&1
" > ../logs/chain_chain.out 2>&1 &
```
Other machine (MELD chain; ~45 h if its GPU is as fast as the 3060, estimate):
```
cd ~/ml_proj/combined/base+chain; PY=~/ml_proj/.venv/bin/python; mkdir -p ../logs
nohup sh -c "$PY run_chain.py --dataset meld --variants all self prev --erc_epochs 3 > ../logs/chain_meld.log 2>&1" > ../logs/chain_chain.out 2>&1 &
```
Check each started: `sleep 20; ls -la ../logs | grep chain; cat ../logs/chain_chain.out` (should be empty); GPU memory appears after ~4 minutes.

### 8d. Bring the results back (run on the main machine; only the small score/prediction files are needed)
```
rsync -avh --include='*/' --include='preds_for_eval_*' --exclude='*' $HOST:~/ml_proj/new/combined/base/Truebaseline/outputs/ ~/ml_proj/new/combined/base/Truebaseline/outputs/
cd ~/ml_proj/combined/base+chain && $PY analyze_chain.py --dataset meld
```
