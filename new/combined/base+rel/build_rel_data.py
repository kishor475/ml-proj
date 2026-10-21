#!/usr/bin/env python3
"""Add a '### Character Relationships:' block to the ERC prompts (placed before the dialogue history).

  python build_rel_data.py --dataset emorynlp                 # on the base ERC data
  python build_rel_data.py --dataset meld --acoustic          # on top of the acoustic-clue prompts (rel + acoustic)
  python build_rel_data.py --dataset iemocap --tag _scene     # IEMOCAP: per-scene relationship (needs the transcript scene alignment)
Output: Truebaseline/data/{dataset}_coe{tag}_rel_erc  (or ..._acoustic_clue_rel_erc)
Turns with nothing known about the characters get the unchanged base prompt (no block).
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
import relations as rel  # noqa: E402
from coe_clues import DATA_DIR, load_dialogues, load_iemocap_turn_scenes, split_speaker  # noqa: E402

HEADER = "### Character Relationships:"


def src_name(dataset, tag, acoustic):
    return f"{dataset}_coe{tag}_acoustic_clue_erc" if acoustic else f"{dataset}_coe{tag}_erc"


def out_name(dataset, tag, acoustic):
    return f"{dataset}_coe{tag}_acoustic_clue_rel_erc" if acoustic else f"{dataset}_coe{tag}_rel_erc"


def add_block(prompt, text):
    """Insert the block before the dialogue history and mention it in the task sentence."""
    m = re.search(r"analyzing the (.*?) information\.", prompt)
    assert m and "\n### Dialogue History:\n" in prompt, "unexpected ERC prompt layout"
    items = re.split(r", | and ", m.group(1))
    items.insert(items.index("Dialogue History"), "Character Relationships")
    sentence = "analyzing the " + (", ".join(items[:-1]) + " and " + items[-1]) + " information."
    prompt = prompt[: m.start()] + sentence + prompt[m.end():]
    return prompt.replace("\n### Dialogue History:\n", f"\n{HEADER}\n{text}\n### Dialogue History:\n", 1)


def block_for(dataset, d, i, scene_ids):
    t = d["turns"][i]
    if dataset == "iemocap":
        return rel.iemocap_block(scene_ids[d["id"]][i])
    speakers = [split_speaker(l)[0] for l in t["history"].split("\n") if l.strip()] + [t["speaker"]]
    return rel.friends_block(speakers)


def build(dataset, tag="", acoustic=False):
    src = os.path.join(DATA_DIR, src_name(dataset, tag, acoustic))
    if not os.path.exists(os.path.join(src, "train.json")):
        sys.exit(f"missing {src}" + ("; run base+acoustic/build_acoustic_data.py first" if acoustic else "; run base/build_coe_data.py first"))
    scene_ids = load_iemocap_turn_scenes() if dataset == "iemocap" else None
    out = os.path.join(DATA_DIR, out_name(dataset, tag, acoustic))
    os.makedirs(out, exist_ok=True)
    print(f"{dataset} / relationships{' + acoustic' if acoustic else ''} -> {os.path.basename(out)}")
    for split in ("train", "dev", "test"):
        with open(os.path.join(src, f"{split}.json"), encoding="utf-8") as f:
            base = [json.loads(l) for l in f if l.strip()]
        turns = [(d, i) for d in load_dialogues(split, dataset) for i in range(len(d["turns"]))]
        assert len(base) == len(turns), f"{split}: {len(base)} rows vs {len(turns)} turns"
        rows, with_block = [], 0
        for b, (d, i) in zip(base, turns):
            assert b["target"] == d["turns"][i]["emotion"], "row/turn mismatch"
            text = block_for(dataset, d, i, scene_ids)
            with_block += bool(text)
            rows.append({"input": add_block(b["input"], text) if text else b["input"], "target": b["target"]})
        with open(os.path.join(out, f"{split}.json"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {len(rows):6d} rows -> {split}.json   (turns that received a relationship block: {with_block} = {100*with_block/len(rows):.0f}%)")
    return os.path.basename(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--tag", default="", help="base tag ('' or _scene for IEMOCAP)")
    ap.add_argument("--acoustic", action="store_true", help="build on the acoustic-clue ERC data")
    a = ap.parse_args()
    build(a.dataset, a.tag, a.acoustic)
