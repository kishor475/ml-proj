#!/usr/bin/env python3
"""One consolidated results table for the whole project -> results/RESULTS.md (+ CSV).

A. Full-length runs (3 epochs, TEST set): for every arm the best-epoch and the FINAL-epoch weighted-F1, and the paired-bootstrap difference
   to the base on the FINAL epoch (no test-set epoch selection). Paper numbers (CoE, Table 1) for reference.
B. Short protocol (1 epoch, 30 % of the data, held-out DEV rows): every ablation arm vs the 'nocue' arm.
Only reads saved prediction files; nothing is trained.   python make_results_table.py
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
for sub in ("base+chain", "base", "acoustic_ablation"):
    sys.path.insert(0, os.path.join(HERE, sub))
import analyze_chain as an  # noqa: E402
import run_strategy_b as rsb  # noqa: E402
from coe_clues import cfg  # noqa: E402

PAPER = {"emorynlp": 44.29, "meld": 70.11, "iemocap": 70.46}
NB = 1000


def epochs(run_dir):
    out = {}
    for f in glob.glob(os.path.join(run_dir, "*", "preds_for_eval_*_*.text")):
        m = re.search(r"preds_for_eval_(\d+)_([\d.]+)\.text", f)
        out[int(m.group(1))] = (float(m.group(2)), f)
    return out


def preds(path, labels):
    rows = an.read_preds(path)
    return (np.array([an.final_label(r["output"], labels) for r in rows]), np.array([an.final_label(r["target"], labels) for r in rows]))


def boot(g, a, b, rng):
    n = len(g)
    d = np.array([an.wf1(g[i], a[i]) - an.wf1(g[i], b[i]) for i in (rng.randint(0, n, n) for _ in range(NB))])
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def flag(lo, hi):
    return "worse*" if hi < 0 else "better*" if lo > 0 else "n.s."


def main():
    md, rows_csv = ["# Results\n", "All numbers are weighted-F1 (%). `*` = the 95 % paired-bootstrap interval excludes 0. One training run per arm: "
                    "differences below ~1-2 points are within noise.\n"], []
    md.append("## A. Full-length runs (3 epochs, test set)\n")
    md.append("Comparison uses each arm's FINAL epoch against the base's FINAL epoch (no test-set epoch selection); 'best' is the best epoch on the test set "
              "(optimistic, shown for reference). Paper = CoE reported score.\n")
    for ds in ("emorynlp", "meld", "iemocap"):
        labels = cfg(ds)["labels"]; rng = np.random.RandomState(0)
        base = epochs(os.path.join(rsb.OUT, f"Stage2_erc_{ds}"))
        if not base:
            continue
        eb = max(base); gb_p, g = preds(base[eb][1], labels)
        md.append(f"### {ds}  (paper {PAPER[ds]})\n")
        md.append("| arm | epochs | best | final | final vs base [95% CI] | |\n|---|---|---|---|---|---|")
        md.append(f"| base | {len(base)} | {max(v[0] for v in base.values()):.2f} | {base[eb][0]:.2f} | - | |")
        runs = {os.path.basename(p)[len("Stage2_"):-len("_" + ds)]: p for p in sorted(glob.glob(os.path.join(rsb.OUT, f"Stage2_*_{ds}")))}
        runs.pop("erc", None)
        extra = {}
        ext = os.path.join(rsb.OUT, f"Stage2_erc_{ds}_ext8")
        if os.path.isdir(ext):
            extra["erc_continued_to_8_epochs"] = ext
        for name, path in list(runs.items()) + list(extra.items()):
            ep = epochs(path)
            if not ep:
                continue
            last = max(ep)
            p, gg = preds(ep[last][1], labels)
            if len(p) != len(g):
                md.append(f"| {name} | {len(ep)} | - | {ep[last][0]:.2f} | (incomplete run) | |"); continue
            m, lo, hi = boot(g, p, gb_p, rng)
            md.append(f"| {name} | {len(ep)} | {max(v[0] for v in ep.values()):.2f} | {ep[last][0]:.2f} | {m:+.2f} [{lo:+.2f}, {hi:+.2f}] | {flag(lo, hi)} |")
            rows_csv.append(["full", ds, name, len(ep), max(v[0] for v in ep.values()), ep[last][0], m, lo, hi])
        md.append("")
    md.append("## B. Short protocol (1 epoch, 30 % of the training data, held-out DEV rows)\n")
    md.append("Every arm starts from the same stage-1 adapter. `nocue` = the same fine-tune without any added clue; `shuffled*` = the same clue text on random turns (control). "
              "EmoryNLP/MELD dev = their dev sets; IEMOCAP dev = every 5th training dialogue.\n")
    for ds in ("emorynlp", "meld", "iemocap"):
        labels = cfg(ds)["labels"]; rng = np.random.RandomState(0)
        arms = {os.path.basename(p)[len(f"Abl_{ds}_"):]: p for p in sorted(glob.glob(os.path.join(rsb.OUT, f"Abl_{ds}_*")))}
        if "nocue" not in arms:
            continue
        ref = epochs(arms["nocue"]); pn, g = preds(ref[max(ref)][1], labels)
        md.append(f"### {ds}\n\n| arm | dev wF1 | vs nocue [95% CI] | |\n|---|---|---|---|")
        out = []
        for name, path in arms.items():
            ep = epochs(path)
            if not ep:
                continue
            p, gg = preds(ep[max(ep)][1], labels)
            if len(p) != len(g):
                continue
            m, lo, hi = (0, 0, 0) if name == "nocue" else boot(g, p, pn, rng)
            out.append((an.wf1(g, p), name, m, lo, hi))
        for w, name, m, lo, hi in sorted(out, reverse=True):
            md.append(f"| {name} | {w:.2f} | " + ("-" if name == "nocue" else f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]") + f" | {'' if name == 'nocue' else flag(lo, hi)} |")
            rows_csv.append(["short", ds, name, 1, w, w, m, lo, hi])
        md.append("")
    path = os.path.join(HERE, "results", "RESULTS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(HERE, "results", "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["protocol", "dataset", "arm", "epochs", "best_or_wF1", "final_or_wF1", "delta", "ci_low", "ci_high"]); w.writerows(rows_csv)
    print("\n".join(md)); print("->", path)


if __name__ == "__main__":
    main()
