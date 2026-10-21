#!/usr/bin/env python3
"""Analyse finished chain runs against the base run, on the test set.

Reports per arm: weighted-F1 / accuracy of the FINAL label; the same on emotion-SHIFT turns (the gold emotion differs from the
previous labelled turn of the dialogue) and on non-shift turns; for chain arms also the accuracy of the generated chain
against the gold chain (per position / per speaker-tagged item); and a paired bootstrap of the F1 difference to the base arm.

  python analyze_chain.py --dataset iemocap
Arms are every finished Stage2_<label>_<D> run under Truebaseline/outputs (base = Stage2_erc_<D>; acoustic, chain and
chain+acoustic arms, including ones made with an ablation-chosen threshold config).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
import chain_data as cd  # noqa: E402
import run_strategy_b as rsb  # noqa: E402
from coe_clues import cfg as dataset_cfg, load_dialogues  # noqa: E402


def latest_preds(run_dir):
    files = glob.glob(os.path.join(run_dir, "*", "preds_for_eval_*_*.text"))
    if not files:
        return None
    best = {}
    for f in files:  # best epoch by the score in the file name
        best[f] = float(re.search(r"preds_for_eval_\d+_([\d.]+)\.text", f).group(1))
    return max(best, key=best.get), max(best.values())


def read_preds(path):
    txt = open(path, encoding="utf-8").read()
    return json.loads(txt[txt.index("[\n"):])


def final_label(text, labels):
    tail = cd.split_chain(text)[1].lower()
    hits = [(tail.find(l.lower()), l.lower()) for l in labels if l.lower() in tail]
    return min(hits)[1] if hits else "<none>"


def wf1(y, p):
    return 100 * f1_score(y, p, average="weighted", zero_division=0)


def items(chain):
    """chain text -> list of emotion items ('speaker: emotion' items keep the speaker)."""
    body = re.sub(r"^(Emotion Flow|Previous Emotion):", "", chain.strip()).strip()
    return [x.strip().lower() for x in body.split("->")] if body else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    labels = dataset_cfg(a.dataset)["labels"]
    d = a.dataset + a.tag
    # every finished Stage2_<label>_<dataset><tag> run is an arm: erc = base; acoustic_clue[_<config>]; acoustic_shuffled[_<config>];
    # chain_<variant>[_acoustic[_<config>]]
    arms = {}
    for path in sorted(glob.glob(os.path.join(rsb.OUT, f"Stage2_*_{d}"))):
        run = os.path.basename(path)
        label = run[len("Stage2_"):-len("_" + d)]
        arms["base" if label == "erc" else label] = run
    found = {}
    for name, run in arms.items():
        r = latest_preds(os.path.join(rsb.OUT, run))
        if r:
            found[name] = (read_preds(r[0]), r[0])
    if not found:
        sys.exit("no finished arms found")

    # which test turns are emotion shifts? (the trainer evaluates a seeded permutation of test.json; reproduce it)
    turns = [(dl, i) for dl in load_dialogues("test", a.dataset) for i in range(len(dl["turns"]))]
    n = len(turns)
    perm = pd.DataFrame({"i": range(n)}).sample(n=n, random_state=42)["i"].to_numpy()
    shift = np.array([None if i == 0 else dl["turns"][i]["emotion"] != dl["turns"][i - 1]["emotion"] for dl, i in turns], dtype=object)[perm]
    gold = np.array([dl["turns"][i]["emotion"].lower() for dl, i in turns])[perm]

    P, C = {}, {}
    for k, (rows, _) in found.items():
        assert len(rows) == n, f"{k}: {len(rows)} predictions for {n} test turns (eval_percent < 1?)"
        P[k] = np.array([final_label(r["output"], labels) for r in rows])
        gt = [final_label(r["target"], labels) for r in rows]
        assert list(gold) == gt, f"{k}: predictions are not in the expected test order"
        C[k] = rows
    isshift = np.array([s is True for s in shift]); nonshift = np.array([s is False for s in shift])
    rng = np.random.RandomState(0); idx = [rng.randint(0, n, n) for _ in range(a.n_boot)]
    print(f"{a.dataset}: {n} test turns, {int(isshift.sum())} emotion-shift turns ({100*isshift.mean():.0f}%)\n")
    print(f"{'arm':44s} wF1    acc  | shift wF1  non-shift wF1 | chain acc | vs base: diff [95% CI]")
    for k in found:
        acc = 100 * accuracy_score(gold, P[k])
        line = f"{k:44s} {wf1(gold,P[k]):5.2f} {acc:5.1f}  | {wf1(gold[isshift],P[k][isshift]):8.2f}  {wf1(gold[nonshift],P[k][nonshift]):12.2f} |"
        if k.startswith("chain"):  # chain arms only
            ok = tot = 0
            for r in C[k]:
                g, p = items(cd.split_chain(r["target"])[0]), items(cd.split_chain(r["output"])[0])
                tot += len(g); ok += sum(x == y for x, y in zip(g, p))
            line += f" {100*ok/max(tot,1):8.1f}  |"
        else:
            line += "        - |"
        if k != "base" and "base" in P:
            dd = np.array([wf1(gold[i], P[k][i]) - wf1(gold[i], P["base"][i]) for i in idx])
            line += f" {dd.mean():+5.2f} [{np.percentile(dd,2.5):+5.2f},{np.percentile(dd,97.5):+5.2f}]"
        print(line)
    print("\nchain acc = share of generated chain items equal to the gold chain item at the same position (chain arms only).")


if __name__ == "__main__":
    main()
