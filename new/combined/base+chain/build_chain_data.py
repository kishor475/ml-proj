#!/usr/bin/env python3
"""Build chain-of-emotion ERC datasets from an existing ERC dataset (same rows, new instruction + chain target).

  python build_chain_data.py --dataset iemocap --variant self
  python build_chain_data.py --dataset emorynlp --variant all --acoustic      # chain on top of the acoustic-clue prompts
Output: Truebaseline/data/{dataset}_coe{tag}_chain_{variant}[_acoustic]_erc/{train,dev,test}.json
Dev/test targets carry the gold chain too (the evaluator only scores the text after 'Current Emotion:'; the gold chain
is used by analyze_chain.py to measure how accurate the generated chain is).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
import chain_data as cd  # noqa: E402
from coe_clues import DATA_DIR, load_dialogues  # noqa: E402


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def src_name(dataset, tag, acoustic, rel=False):
    """acoustic: False | True (original clue recipe) | config slug (clues made with an ablation-chosen thresholds).
    rel: also use the prompts with the 'Character Relationships' block (base+rel; stacked after the acoustic clue)."""
    name = f"{dataset}_coe{tag}"
    if acoustic:
        name += "_acoustic_clue" + (f"_{acoustic}" if isinstance(acoustic, str) else "")
    return name + ("_rel" if rel else "") + "_erc"


def out_name(dataset, tag, variant, acoustic, rel=False):
    ac_part = "" if not acoustic else ("_acoustic" + (f"_{acoustic}" if isinstance(acoustic, str) else ""))
    return f"{dataset}_coe{tag}_chain_{variant}{ac_part}{'_rel' if rel else ''}_erc"


def build(dataset, variant, tag="", acoustic=False, rel=False):
    src = os.path.join(DATA_DIR, src_name(dataset, tag, acoustic, rel))
    if not os.path.exists(os.path.join(src, "train.json")):
        sys.exit(f"missing {src}" + ("; run base+acoustic/build_acoustic_data.py first" if acoustic else "; run base/build_coe_data.py first") + ("; and base+rel/build_rel_data.py" if rel else ""))
    out = os.path.join(DATA_DIR, out_name(dataset, tag, variant, acoustic, rel))
    os.makedirs(out, exist_ok=True)
    print(f"{dataset} / {variant}{' + acoustic' if acoustic else ''}{' + rel' if rel else ''} -> {os.path.basename(out)}")
    for split in ("train", "dev", "test"):
        base = read_rows(os.path.join(src, f"{split}.json"))
        turns = [(d, i) for d in load_dialogues(split, dataset) for i in range(len(d["turns"]))]
        assert len(base) == len(turns), f"{split}: {len(base)} rows vs {len(turns)} turns"
        rows, unknown = [], 0
        for b, (d, i) in zip(base, turns):
            t = d["turns"][i]
            assert b["target"] == t["emotion"], "row/turn mismatch"
            hist = cd.gold_history(d, i)
            unknown += sum(e is None for _, e in hist)
            rows.append({"input": cd.chain_prompt(b["input"], variant),
                         "target": cd.chain_target(variant, t["speaker"], hist, t["emotion"])})
        with open(os.path.join(out, f"{split}.json"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {len(rows):6d} rows -> {split}.json   (history lines without a gold emotion: {unknown})")
    return os.path.basename(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--variant", required=True, choices=cd.VARIANTS)
    ap.add_argument("--tag", default="", help="base tag ('' or _scene for IEMOCAP)")
    ap.add_argument("--acoustic", action="store_true", help="build on the acoustic-clue ERC data")
    ap.add_argument("--rel", action="store_true", help="build on the prompts with the character-relationship block (base+rel)")
    ap.add_argument("--acoustic_config", default=None, help="build on acoustic data made with this ablation-chosen config (implies --acoustic)")
    a = ap.parse_args()
    slug = None
    if a.acoustic_config:
        sys.path.insert(0, os.path.join(HERE, "..", "base+acoustic"))
        import acoustic_clues as ac
        slug = ac.parse_config(a.acoustic_config, a.dataset)[1]
    build(a.dataset, a.variant, a.tag, slug or a.acoustic, a.rel)
