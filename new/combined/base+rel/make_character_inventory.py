#!/usr/bin/env python3
"""List every character (speaker name) in EmoryNLP / MELD with counts, co-occurring main characters and an example line,
so a relationship map can be written against the exact names used in the data.
  python make_character_inventory.py            -> character_inventory_emorynlp.csv, character_inventory_meld.csv
"""
import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
import relations as rel  # noqa: E402
from coe_clues import cfg, load_dialogues  # noqa: E402


def main():
    for ds in ("emorynlp", "meld"):
        main_names = set(cfg(ds)["main"])
        turns, dialogs, example, together = collections.Counter(), collections.defaultdict(set), {}, collections.defaultdict(collections.Counter)
        for split in ("train", "dev", "test"):
            for d in load_dialogues(split, ds):
                present = {t["speaker"] for t in d["turns"]}
                for t in d["turns"]:
                    turns[t["speaker"]] += 1
                    dialogs[t["speaker"]].add(d["id"])
                    example.setdefault(t["speaker"], t["text"][:90])
                for s in present:
                    for m in present & main_names:
                        if m != s:
                            together[s][m] += 1
        path = os.path.join(HERE, f"character_inventory_{ds}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "turns", "dialogues", "is_main_cast", "in_current_kb", "appears_with_main", "example_line"])
            for name, n in turns.most_common():
                r = rel.recognise(name)
                w.writerow([name, n, len(dialogs[name]), int(name in main_names), (r[1] if r else ""),
                            "; ".join(f"{m}({c})" for m, c in together[name].most_common(3)), example[name]])
        print(f"{ds}: {len(turns)} characters -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
