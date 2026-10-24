#!/usr/bin/env python3
"""ERC datasets with the hand-built relationship map ('### Character Relationships:' block) for EmoryNLP / MELD.

  python build_relmap_data.py --dataset emorynlp                 # style detail (default)
  python build_relmap_data.py --dataset meld --style category
Output: Truebaseline/data/{dataset}_coe_relmap_{style}_erc/{train,dev,test}.json   (turns with no known character get the unchanged base prompt)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
import build_rel_data as brd  # noqa: E402  (add_block: block placement + task-sentence edit)
import relmap  # noqa: E402
from coe_clues import DATA_DIR, load_dialogues, split_speaker  # noqa: E402


def window_speakers(d, i):
    """current speaker first, then the history speakers, newest first."""
    t = d["turns"][i]
    hist = [split_speaker(l)[0] for l in t["history"].split("\n") if l.strip()]
    return [t["speaker"]] + hist[::-1]


def out_name(dataset, style, tag=""):
    return f"{dataset}_coe{tag}_relmap_{style}_erc"


def block_for(rm, d, i, style):
    return "\n".join(rm.lines(window_speakers(d, i), style))


def build(dataset, style="detail", tag="", include_core_group=False):
    rm = relmap.RelMap(dataset, include_core_group)
    src = os.path.join(DATA_DIR, f"{dataset}_coe{tag}_erc")
    if not os.path.exists(os.path.join(src, "train.json")):
        sys.exit(f"missing {src}; run base/build_coe_data.py first")
    out = os.path.join(DATA_DIR, out_name(dataset, style, tag))
    os.makedirs(out, exist_ok=True)
    print(f"{dataset} / relmap-{style} -> {os.path.basename(out)}")
    for split in ("train", "dev", "test"):
        with open(os.path.join(src, f"{split}.json"), encoding="utf-8") as f:
            base = [json.loads(l) for l in f if l.strip()]
        turns = [(d, i) for d in load_dialogues(split, dataset) for i in range(len(d["turns"]))]
        assert len(base) == len(turns), f"{split}: {len(base)} rows vs {len(turns)} turns"
        rows, n_block = [], 0
        for b, (d, i) in zip(base, turns):
            assert b["target"] == d["turns"][i]["emotion"], "row/turn mismatch"
            text = block_for(rm, d, i, style)
            n_block += bool(text)
            rows.append({"input": brd.add_block(b["input"], text) if text else b["input"], "target": b["target"]})
        with open(os.path.join(out, f"{split}.json"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {len(rows):6d} rows -> {split}.json   (turns with a relationship block: {n_block} = {100*n_block/len(rows):.0f}%)")
    return os.path.basename(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld"])
    ap.add_argument("--style", default="detail", choices=relmap.STYLES)
    ap.add_argument("--tag", default="")
    ap.add_argument("--core_group", action="store_true", help="also include the main-cast rows (late-series facts, anachronistic early on)")
    a = ap.parse_args()
    build(a.dataset, a.style, a.tag, a.core_group)
