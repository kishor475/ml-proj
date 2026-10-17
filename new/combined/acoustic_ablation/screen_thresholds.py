#!/usr/bin/env python3
"""Tier-1 screen for acoustic-clue thresholds (CPU only, minutes).

For every configuration (percentile cut points x normalisation x feature set) the training turns are categorised
(cut points fitted on TRAIN only), then a small classifier (one-hot categories -> emotion) is trained on train and
scored on a DEV split. This measures how much emotion information the discretised clues carry; it does not involve
the LLM. Shortlist the best configs for the GPU fine-tune check.

Dev split: EmoryNLP / MELD use their dev sets. IEMOCAP has no dev (its dev file equals test), so every 5th training
dialogue is held out. The test set is never used here.

Controls: majority class; 'has audio' flag only (guards against missing-audio leakage, relevant for EmoryNLP);
continuous raw features (ceiling for what discretisation can keep).
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "base+acoustic"))
import acoustic_clues as ac  # noqa: E402

SCHEMES = [("5c p10-25-75-90 (original)", (10, 25, 75, 90)), ("5c p5-20-80-95", (5, 20, 80, 95)),
           ("5c p20-40-60-80", (20, 40, 60, 80)), ("5c p10-30-70-90", (10, 30, 70, 90)),
           ("3c p25-75", (25, 75)), ("3c p33-67", (33, 67)), ("3c p20-80", (20, 80)), ("3c p10-90", (10, 90))]
NORMS = ["global", "speaker_pct", "speaker"]
FEATSETS = [("5feat", ac.FEATS5), ("8feat", ac.FEATS8)]


import acoustic_data as ad  # noqa: E402


def splits(dataset):
    return ad.holdout_split(dataset)


def wf1(y, p):
    return 100 * f1_score(y, p, average="weighted", zero_division=0)


def boot_ci(y, p, n=200, seed=0):
    rng = np.random.RandomState(seed); y = np.asarray(y); p = np.asarray(p); v = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y)); v.append(wf1(y[i], p[i]))
    return 1.96 * np.std(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["meld", "emorynlp", "iemocap"])
    ap.add_argument("--min_speaker_n", type=int, default=100)
    a = ap.parse_args()
    tr, dv = splits(a.dataset)
    ytr = [r["turn"]["emotion"] for r in tr]; ydv = [r["turn"]["emotion"] for r in dv]
    print(f"{a.dataset}: train {len(tr)} dev {len(dv)}; with audio train {sum(r['feats'] is not None for r in tr)} dev {sum(r['feats'] is not None for r in dv)}")
    out = []

    def add(name, p, extra=""):
        out.append(dict(config=name, dev_wF1=round(wf1(ydv, p), 2), dev_acc=round(100 * accuracy_score(ydv, p), 2),
                        ci95=round(boot_ci(ydv, p), 2), note=extra))

    maj = max(set(ytr), key=ytr.count); add("control: always majority", [maj] * len(ydv))
    dvz = DictVectorizer()
    flag = lambda rows: [{"has_audio": int(r["feats"] is not None)} for r in rows]
    add("control: has-audio flag only", LogisticRegression(max_iter=500).fit(dvz.fit_transform(flag(tr)), ytr).predict(dvz.transform(flag(dv))))
    # ceiling: continuous raw features (+ per-speaker z) with gradient boosting
    cont = lambda rows: np.array([[(r["feats"] or {}).get(f) or np.nan for f in ac.FEATS8] for r in rows], dtype=float)
    Xc, Xd = cont(tr), cont(dv)
    add("ceiling: continuous raw 8 feats (GB)", HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xc, ytr).predict(Xd))

    for (sname, perc) in SCHEMES:
        for norm in NORMS:
            for (fname, feats) in FEATSETS:
                cfg = ac.ClueConfig(perc, feats, norm, a.min_speaker_n)
                st = ac.fit(cfg, tr)
                enc = lambda rows: [{f"{k}={v}": 1 for k, v in ac.categorize(cfg, st, r).items()} for r in rows]
                v = DictVectorizer(); Xt = v.fit_transform(enc(tr)); Xv = v.transform(enc(dv))
                p = LogisticRegression(max_iter=500).fit(Xt, ytr).predict(Xv)
                add(f"{sname} | {norm} | {fname}", p)
    out_sorted = sorted(out, key=lambda r: -r["dev_wF1"])
    path = os.path.join(HERE, "results", f"{a.dataset}_screen.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out_sorted)
    print(f"{'config':58s} dev wF1   ±95%   acc")
    for r in out_sorted[:12] + [x for x in out_sorted if x["config"].startswith(("control", "ceiling"))]:
        print(f"{r['config']:58s} {r['dev_wF1']:6.2f}  {r['ci95']:5.2f}  {r['dev_acc']:5.1f}")
    print("->", path)


if __name__ == "__main__":
    main()
