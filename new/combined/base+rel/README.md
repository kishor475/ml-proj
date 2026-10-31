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


## Building the relationship map yourself (planned replacement for the hand-written `FRIENDS_KB`)
1. `python make_character_inventory.py` writes `character_inventory_emorynlp.csv` and `character_inventory_meld.csv`: every speaker name exactly as it appears in the
   data, with turns, dialogues, the main characters they appear with and an example line. Describe the ~45 non-main characters with >= 15 turns first
   (they cover most of the benefit; the long tail of one-line characters can be skipped).
2. Write the relations as rows of one CSV (template: `relations_map_template.csv`), using the names from the inventory:
   ```
   character_a,relation,character_b,note
   Chandler Bing,friend,Rachel Green,
   Jack Geller,parent_of,Ross Geller,
   Carol Willick,ex_spouse,Ross Geller,
   ```
   Relation words (keep to this list so the clue text stays consistent): symmetric `friend, sibling, spouse, ex_spouse, partner, ex_partner, roommate,
   coworker, neighbor, acquaintance, enemy`; directed (a -> b) `parent_of, child_of, boss_of, employee_of, landlord_of`; or `other` with the
   wording in `note`. For MELD, use the same names as the data (first names, e.g. `Carol`) or add alias rows.
3. Rendering (to be implemented once the file exists): for each turn the characters in the window are the history speakers plus the current speaker;
   every pair with a row becomes one sentence ("Jack Geller is the parent of Ross Geller."), pairs involving the current speaker first, at most 6 sentences;
   turns with no matching pair get no block (as now). The result replaces `FRIENDS_KB`, and the same short-protocol ablation as the other clues
   (`acoustic_ablation/run_ablation.py`, arms: no clue / relationships / relationships shuffled onto random turns) decides whether it helps.
Only time-invariant relations should be entered (no season-dependent romances), since one map is used for every episode.


## The hand-built relationship map (EmoryNLP, MELD) — `relmap.py`, `build_relmap_data.py`
Source files (this folder, spelled as on disk): `emory_chars.json` / `emory_char_realtions.json` / `emory_relation_tuples.json` and `meld_chars.json` /
`meld_char_relations.json` / `meld_realtion_tuples.json`. For every kept side character they give the relation to the six core friends (category, short `detail`, a
`description`, and `all_core_relations`). Checked: every name matches a speaker name in the data (164/164, 210/210); the tuple files agree with `all_core_relations`
on every pair they contain but lack some (2 pairs EmoryNLP, 32 MELD), so `all_core_relations` is the source of truth. `core_group` rows (the main cast about each other, with
late-series facts like "Husband to Monica") are skipped by default (anachronistic in early seasons); `--core_group` includes them.
For each turn the window is the current speaker + the history speakers; a side character in the window gets a line for its relations to the core friends in the scene
(or its primary relation if none of them is present), at most 4 lines, current speaker first:
```
### Character Relationships:
Alan - relation to Monica Geller: Boyfriend.
```
Styles: `detail` (default; ~18 tokens), `category` (a fixed phrase per category), `desc` (+ the description sentence, ~47 tokens). Coverage of turns with a block:
EmoryNLP 39-40 %, MELD 27-28 % (the older built-in KB: 28 % / 17 %).
```
python build_relmap_data.py --dataset emorynlp --style detail          # -> {dataset}_coe_relmap_detail_erc  (also: meld; --style category | desc)
python run_rel.py --dataset emorynlp --map user --style detail --erc_epochs 3     # full-length arm, run name Stage2_relmap_detail_<D>
```
**Short protocol with controls** (1 epoch, 30 % of the data, held-out dev; ~1.5 h per arm; arms `relmap`, `shuffled_relmap` (control: same blocks on random turns),
`relmap_cat`, `relmap_desc`; `nocue` is already run):
```
cd ../acoustic_ablation
python run_ablation.py --dataset emorynlp --configs --rel_ablation --percent 0.3
python run_ablation.py --dataset meld     --configs --rel_ablation --percent 0.3
python summarize_ablation.py --dataset emorynlp      # prints "RELMAP 'relmap': vs nocue ..., vs shuffled_relmap ... -> SUPPORTED / not supported"
```
