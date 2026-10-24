#!/usr/bin/env python3
"""Why does a chain arm score differently from the base? Diagnostics from the saved test predictions (best epoch of each arm).

For every chain arm of a dataset, on the same test turns as the base run:
  1. truncated   share of outputs without a 'Current Emotion:' line (generation cut off -> label lost)
  2. chain acc   share of generated chain items equal to the gold item (position-wise); 'full' = turns whose WHOLE chain is right
  3. error propagation: final-label accuracy of the chain arm vs the base arm, split by whether the generated chain was fully right
     (if the chain arm is fine when its chain is right but worse when it is wrong -> errors in the chain propagate to the label)
  4. copy behaviour (variants 'prev' and 'all'): how often the predicted current emotion equals the chain's last item,
     against how often the gold current emotion equals the gold previous emotion (true continuity rate)
  5. the final-label share of the target tokens (how much of the training loss is spent on the graded label)
  python chain_diagnostics.py --dataset meld
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
import analyze_chain as an  # noqa: E402
import chain_data as cd  # noqa: E402
import run_strategy_b as rsb  # noqa: E402
from coe_clues import cfg as dataset_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    labels = dataset_cfg(a.dataset)["labels"]; d = a.dataset + a.tag
    runs = {os.path.basename(p)[len("Stage2_"):-len("_" + d)]: p for p in sorted(glob.glob(os.path.join(rsb.OUT, f"Stage2_*_{d}")))}
    base = an.latest_preds(runs["erc"]); brows = an.read_preds(base[0])
    gold = np.array([an.final_label(r["target"], labels) for r in brows])
    bpred = np.array([an.final_label(r["output"], labels) for r in brows])
    print(f"{a.dataset}: {len(brows)} test turns; base acc {100*(bpred==gold).mean():.1f}%\n")
    print(f"{'arm':14s} trunc%  chain-acc  full-chain%  | label acc when chain FULLY right (chain / base)  | when chain wrong (chain / base) | copy: pred==last item (gold continuity)")
    for name, path in runs.items():
        if not name.startswith("chain_") or "acoustic" in name or "_rel" in name:
            continue
        r = an.latest_preds(path)
        if not r:
            continue
        rows = an.read_preds(r[0]); n = len(rows)
        if n != len(brows):
            print(f"{name}: {n} rows != {len(brows)} (incomplete run)"); continue
        pred = np.array([an.final_label(x["output"], labels) for x in rows])
        trunc = np.mean(["Current Emotion:" not in x["output"] for x in rows])
        gch = [an.items(cd.split_chain(x["target"])[0]) for x in rows]
        pch = [an.items(cd.split_chain(x["output"])[0]) for x in rows]
        ok = sum(sum(p == g for p, g in zip(pc, gc)) for pc, gc in zip(pch, gch)); tot = sum(len(g) for g in gch)
        full = np.array([pc == gc for pc, gc in zip(pch, gch)])
        acc = lambda m, y: 100 * (y[m] == gold[m]).mean() if m.any() else float("nan")
        copy = ""
        if name.startswith(("chain_prev", "chain_all")):
            last_p = [pc[-1].split(":")[-1].strip() if pc else None for pc in pch]
            last_g = [gc[-1].split(":")[-1].strip() if gc else None for gc in gch]
            cp = np.mean([(p is not None) and (pred[i] == p) for i, p in enumerate(last_p)])
            cg = np.mean([(g is not None) and (gold[i] == g) for i, g in enumerate(last_g)])
            copy = f"{100*cp:.0f}%  (gold {100*cg:.0f}%)"
        print(f"{name:14s} {100*trunc:5.1f}  {100*ok/tot:8.1f}  {100*full.mean():11.1f}   | {acc(full, pred):5.1f} / {acc(full, bpred):5.1f}  (n={int(full.sum()):4d})"
              f"            | {acc(~full, pred):5.1f} / {acc(~full, bpred):5.1f}  (n={int((~full).sum()):4d}) | {copy}")
    print("\ntarget tokens: the graded label is only the last part of the target, e.g. 'Emotion Flow: a -> b -> c -> d -> e\\nCurrent Emotion: x'.")


if __name__ == "__main__":
    main()
