#!/usr/bin/env python3
"""Majority-vote ensembles of the already-trained arms (no training, inference-level).

For a dataset, loads the saved test predictions of the base arm and the chain arms and evaluates
  base+prev+self   vote of three models        all four   vote of base+prev+self+all        prev+self+all   (no base)
with ties broken in favour of the base. Reports weighted-F1 and the paired-bootstrap difference to the base (95 % interval).
  --epoch last  use each arm's FINAL epoch (default; no test-set selection of epochs)
  --epoch best  use each arm's best epoch on the test set (what analyze_chain.py reports; optimistic, for comparison)
Cost of an ensemble at inference time: one forward/generation pass per member model.
  python ensemble_eval.py --dataset meld --epoch last
"""
import argparse
import collections
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
import analyze_chain as an  # noqa: E402
import run_strategy_b as rsb  # noqa: E402
from coe_clues import cfg as dataset_cfg  # noqa: E402


def pick(run_dir, which):
    files = glob.glob(os.path.join(run_dir, "*", "preds_for_eval_*_*.text"))
    if not files:
        return None
    ep = lambda f: int(re.search(r"preds_for_eval_(\d+)_", f).group(1))
    f = max(files, key=ep) if which == "last" else max(files, key=lambda f: float(re.search(r"_([\d.]+)\.text", f).group(1)))
    return f, ep(f), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--epoch", default="last", choices=["last", "best"])
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    labels = dataset_cfg(a.dataset)["labels"]; d = a.dataset + a.tag
    P, info, gold = {}, {}, None
    for key, run in (("base", "erc"), ("prev", "chain_prev"), ("self", "chain_self"), ("all", "chain_all")):
        r = pick(os.path.join(rsb.OUT, f"Stage2_{run}_{d}"), a.epoch)
        if not r:
            print(f"[skip] no predictions for {run}"); continue
        rows = an.read_preds(r[0]); info[key] = (r[1], r[2])
        P[key] = np.array([an.final_label(x["output"], labels) for x in rows])
        g = np.array([an.final_label(x["target"], labels) for x in rows])
        assert gold is None or (g == gold).all(), "arms were scored on different test turns"
        gold = g
    n = len(gold); rng = np.random.RandomState(0); idx = [rng.randint(0, n, n) for _ in range(a.n_boot)]
    f = lambda p, i=slice(None): an.wf1(gold[i], p[i])
    def vote(keys):
        out = []
        for i in range(n):
            c = collections.Counter(P[k][i] for k in keys); top = max(c.values())
            cand = [l for l, v in c.items() if v == top]
            out.append(P["base"][i] if P["base"][i] in cand else cand[0])
        return np.array(out)
    print(f"{a.dataset}: {n} test turns, epoch={a.epoch}; arms (epoch used, #epochs saved): " + ", ".join(f"{k}=ep{v[0]}/{v[1]}" for k, v in info.items()))
    if len({v[1] for v in info.values()}) > 1:
        print("  WARNING: arms have a different number of finished epochs (an arm is still training); re-run when it has finished")
    print(f"  {'method':32s} wF1     delta vs base [95% CI]")
    print(f"  {'base alone':32s} {f(P['base']):6.2f}")
    out = [("base alone", f(P["base"]), "", "", "")]
    for name, keys in (("vote base+prev+self", ["base", "prev", "self"]), ("vote base+prev+self+all", ["base", "prev", "self", "all"]),
                       ("vote prev+self+all (no base)", ["prev", "self", "all"])):
        if not all(k in P for k in keys):
            continue
        e = vote(keys); dd = np.array([f(e, i) - f(P["base"], i) for i in idx])
        print(f"  {name:32s} {f(e):6.2f}  {dd.mean():+5.2f} [{np.percentile(dd, 2.5):+5.2f}, {np.percentile(dd, 97.5):+5.2f}]")
        out.append((name, f(e), dd.mean(), np.percentile(dd, 2.5), np.percentile(dd, 97.5)))
    anyr = np.mean([any(P[k][i] == gold[i] for k in P) for i in range(n)]) * 100
    print(f"  (upper bound: at least one member is right on {anyr:.0f}% of turns; base alone {100*(P['base']==gold).mean():.0f}%)")
    path = os.path.join(HERE, "results", f"ensemble_{d}_{a.epoch}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "wF1", "delta_vs_base", "ci_low", "ci_high"]); w.writerows(out)
    print("  ->", path)


if __name__ == "__main__":
    main()
