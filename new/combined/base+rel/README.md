# base+rel — character-relationship clue

Adds a `### Character Relationships:` block (before the dialogue history) to the ERC prompts of the base pipeline. The block is a
**who's-who map** for the characters in the current dialogue window (the speakers of the history lines + the current speaker):
```
### Character Relationships:
Carol Willick is Ross Geller's ex-wife. Susan Bunch is Carol Willick's partner. Ross Geller and Monica Geller are siblings.
```
- **EmoryNLP / MELD:** recognised side characters are introduced *by name* with their relation to the cast; the Ross-Monica sibling tie
  is added when both are present. Turns whose window has nothing known get the unchanged base prompt (no block). Coverage on train:
  EmoryNLP 28 %, MELD 17 % of turns (of those, ~70 % mention a side character, ~30 % only the sibling tie), so any effect on the overall
  score is diluted accordingly.
- **IEMOCAP:** one relationship sentence per scene ("The speakers are strangers: a customer and a customer-service agent."), taken from the
  transcript-derived scene of each turn (needs `base/Truebaseline/data/iemocap_turn_scenes.json`, the same prerequisite as the `_scene`
  variant). 100 % coverage. The wording is emotion-free (the old experiment said "a frustrated customer", which leaks the label), but
  the relationship still correlates with the elicited emotion (as the scene does), so treat IEMOCAP results as an extension, not paper-comparable.

## Improvements over the earlier base+chain+acous+rel clue
The old clue listed descriptions without names ("known associates (Ross's ex-wife., Carol's partner.)"), printed a constant "a group of
close friends" for every main-cast-only window (no information), and used emotion words for IEMOCAP. This one keeps the name->relation
mapping, omits the block when nothing is known, and is emotion-free.

## Run (after the base run for the dataset has finished)
```
cd combined/base+rel
python run_rel.py --dataset emorynlp --erc_epochs 3
python run_rel.py --dataset meld --erc_epochs 3
python run_rel.py --dataset iemocap --erc_epochs 3                 # relationship from the scene alignment (no scene text)
python run_rel.py --dataset iemocap --tag _scene --erc_epochs 3    # on top of the IEMOCAP scene variant
python run_rel.py --dataset meld --acoustic --erc_epochs 3         # relationships + acoustic clues
# stacked with the chain (and optionally the acoustic clue): base+chain/run_chain.py ... --rel [--acoustic]
```
Each arm starts from the base stage-1 adapter with the base ERC hyper-parameters (`--erc_epochs` must equal the base run's), so it differs
only in the prompt. Compare `Stage2_rel_<D>` with `Stage2_erc_<D>` using `base+chain/analyze_chain.py --dataset <D>`. Resumable.
~12-13 h per arm for EmoryNLP/MELD, ~4-5 h for IEMOCAP (estimates from the other stage-2 arms).

## Files
`relations.py` (knowledge base + renderers) | `build_rel_data.py` | `run_rel.py`

## Caveats
- **Review `FRIENDS_KB` in `relations.py`.** It is the curated list from the earlier experiment (with names added) plus a few entries I added
  where the show facts are certain; a wrong fact becomes a wrong training prompt. Ambiguous first names (Paul, Julie, Mark, Bonnie) are left out.
  Only time-invariant facts are used (no season-dependent romances).
- There is no control arm for this clue yet (e.g. relationships assigned to random turns). If a gain appears, check it against such a control
  before attributing it to the relationship information.
