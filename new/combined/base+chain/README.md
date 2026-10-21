# base+chain — Chain of Emotion for LLM-based ERC

**Idea.** Instead of mapping (history, utterance) straight to a label, the model first writes the emotion trajectory of the
dialogue history and then the label of the current utterance:
```
Emotion Flow: Chandler Bing: Joyful -> Joey Tribbiani: Neutral -> Chandler Bing: Scared
Current Emotion: Joyful
```
The input never contains gold emotions; the chain is supervised with the gold labels of the history turns at training time and
is self-generated at test time (so nothing leaks). The final answer stays on the `Current Emotion:` line, which the existing
evaluator parses.

## Research questions
1. Does generating the emotion trajectory first improve ERC over direct prediction (same adapter, same budget)?
2. Which chain definition works: `self` (the current speaker's previous emotions; the earlier design), `all` (every history
   turn, speaker-tagged), or `prev` (only the previous turn)?
3. Is the gain concentrated on **emotion-shift turns** (gold emotion differs from the previous turn)? On base EmoryNLP (epoch 0)
   68 % of test turns are shifts and the base model scores 32.0 wF1 on them vs 46.7 on non-shifts.
4. How accurate are the generated chains, and does a wrong chain hurt the final label (error propagation)?
5. Are chain and acoustic clues complementary (`--acoustic`, see base+acoustic)?

## Run (after the base run for the dataset has finished)
```
cd combined/base+chain
python run_chain.py --dataset iemocap --variants self all prev --erc_epochs 3     # IEMOCAP first (smallest: ~5 h per arm)
python run_chain.py --dataset iemocap --variants all --acoustic --erc_epochs 3    # base+chain+acoustic
python analyze_chain.py --dataset iemocap                                          # wF1, shift/non-shift wF1, chain accuracy, paired bootstrap vs base
```
Each arm starts from the same stage-1 adapter as the base stage 2 and uses the same ERC hyper-parameters, so the arms differ
only in the prompt instruction and target. Re-run a command to resume. Roughly 12 h per arm for EmoryNLP/MELD at 3 epochs.
Evaluation takes longer than for the base (the model generates the chain): the generation budget is set per variant
(prev 32 / self 64 / all 128 tokens).

## Files
`chain_data.py` (gold-history alignment, targets, instructions) | `build_chain_data.py` | `run_chain.py` | `analyze_chain.py`

## Caveats
- IEMOCAP: ~23 % of history lines were never labelled in the dataset; they appear as `?` (`all`) / are skipped (`self`) / `unknown` (`prev`).
- Statistics: the paired bootstrap covers test-set sampling noise only. Training-seed variance is not measured (one seed per arm);
  repeat the headline comparison with a second seed before claiming small gains.
