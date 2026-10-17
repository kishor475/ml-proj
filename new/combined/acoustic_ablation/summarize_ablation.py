#!/usr/bin/env python3
"""Rank the ablation arms on the held-out dev rows, with paired bootstrap comparisons.

For each arm: weighted-F1, accuracy, and the paired-bootstrap difference to 'nocue' and to 'shuffled'
(95% interval of the difference; 'P(diff<=0)' is the share of resamples where the arm does not beat the control).
A config is only worth adopting if it beats BOTH controls with an interval above zero.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
sys.path.insert(0, os.path.join(HERE, "..", "base+acoustic"))
import acoustic_clues as ac  # noqa: E402
import build_ablation_data as bld  # noqa: E402
import configs  # noqa: E402
from coe_clues import cfg as dataset_cfg  # noqa: E402
import run_strategy_b as rsb  # noqa: E402


def load_preds(path, labels):
    txt = open(path, encoding="utf-8").read()
    rows = json.loads(txt[txt.index("[\n"):])
    low = [l.lower() for l in labels]
    def lab(s):
        s = s.lower(); best = (len(s) + 1, None)
        for l in low:
            k = s.find(l)
            if k != -1 and k < best[0]:
                best = (k, l)
        return best[1] or "<none>"
    return [lab(r["output"]) for r in rows], [r["target"].lower() for r in rows]


def wf1(y, p):
    return 100 * f1_score(y, p, average="weighted", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--reference", default=None, help="arm used as the full design for the component table (default: desc_ctx, else desc)")
    a = ap.parse_args()
    labels = dataset_cfg(a.dataset)["labels"]
    arms = {}
    for d in sorted(glob.glob(os.path.join(rsb.OUT, f"Abl_{a.dataset}{a.tag}_*"))):
        name = os.path.basename(d)[len(f"Abl_{a.dataset}{a.tag}_"):]
        files = glob.glob(os.path.join(d, "*", "preds_for_eval_*_*.text"))
        if files:
            arms[name] = load_preds(max(files, key=lambda f: int(re.search(r"preds_for_eval_(\d+)_", f).group(1))), labels)
    if not arms:
        sys.exit("no finished ablation arms found")
    ref_t = next(iter(arms.values()))[1]
    assert all(v[1] == ref_t for v in arms.values()), "arms were scored on different dev rows"
    y = np.array(ref_t); n = len(y); rng = np.random.RandomState(0)
    idx = [rng.randint(0, n, n) for _ in range(a.n_boot)]
    P = {k: np.array(v[0]) for k, v in arms.items()}
    base_f = {k: wf1(y, P[k]) for k in P}
    def diff(k, ctrl):
        if ctrl not in P or k == ctrl:
            return None
        d = np.array([wf1(y[i], P[k][i]) - wf1(y[i], P[ctrl][i]) for i in idx])
        return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), float((d <= 0).mean())
    print(f"{a.dataset}: {n} held-out dev rows\n{'arm':52s} wF1    acc | vs nocue: diff [95% CI] P(<=0) | vs shuffled: diff [95% CI] P(<=0)")
    for k in sorted(P, key=lambda k: -base_f[k]):
        acc = 100 * accuracy_score(y, P[k]); line = f"{k:52s} {base_f[k]:5.2f} {acc:5.1f} |"
        for ctrl in ("nocue", "shuffled"):
            d = diff(k, ctrl)
            line += "        -" + " " * 30 + "|" if d is None else f" {d[0]:+5.2f} [{d[1]:+5.2f},{d[2]:+5.2f}] {d[3]:.3f} |"
        print(line)

    # component ablation table: every arm vs the full design (change one component at a time)
    ref = configs.slug(a.reference) if a.reference else next((configs.slug(r) for r in ("desc_ctx", "desc") if configs.slug(r) in P), None)
    if ref in P:
        label = {"desc": "- audio_context tags", "desc-nov": "- volume features", "desc-ctx-nov": "- volume features",
                 "desc-nop": "- pitch features (and tags)", "desc-ctx-nop": "- pitch features (and tags)",
                 "desc-nor": "- speech-rate feature", "desc-ctx-nor": "- speech-rate feature",
                 "desc-3c": "5-class -> 3-class cutoffs", "desc-ctx-3c": "5-class -> 3-class cutoffs",
                 "desc-spk": "global -> per-speaker cutoffs", "desc-ctx-spk": "global -> per-speaker cutoffs",
                 "both": "+ impression sentence", "both-ctx": "+ impression sentence",
                 "default": "impression form instead of description", "nocue": "no clue at all",
                 "shuffled": "impression, shuffled onto random turns", "shuffled-desc": "description, shuffled (control)",
                 "shuffled-desc-ctx": "full design, shuffled (control)"}
        print(f"COMPONENT ABLATION   full design = '{ref}' (wF1 {base_f[ref]:.2f})   delta = arm - full, paired bootstrap")
        print(f"{'change relative to the full design':42s} {'arm':20s} {'wF1':>6s}  {'delta [95% CI]':24s} reading")
        for k in sorted((k for k in P if k != ref), key=lambda k: -base_f[k]):
            d = diff(k, ref)
            read = ("CHANGE HURTS -> component helps" if d[2] < 0 else "change helps -> component not needed" if d[1] > 0
                    else "no measurable difference")
            print(f"{label.get(k, k):42s} {k:20s} {base_f[k]:6.2f}  {d[0]:+5.2f} [{d[1]:+5.2f},{d[2]:+5.2f}]   {read}")
        print("(a component is justified only where removing it hurts with the 95% interval below zero; "
              "'no measurable difference' means the data cannot tell it from the full design)\n")

    # verdict: a threshold config is adopted only if it beats BOTH controls with the 95% interval above zero
    adopt = []
    for k in P:
        if ac._SLUG.match(k):
            d1, d2 = diff(k, "nocue"), diff(k, "shuffled")
            if d1 and d2 and d1[1] > 0 and d2[1] > 0:
                adopt.append(k)
    t = f" --tag {a.tag}" if a.tag else ""
    print()
    for fa, ctrl in bld.CONTROL_OF.items():          # SpeechCueLLM-style forms: each against nocue AND its own shuffled control
        fk, ck = configs.slug(fa), configs.slug(ctrl)  # arm folders are named with hyphens (desc-ctx)
        if fk in P and ck in P:
            d1, d2 = diff(fk, "nocue"), diff(fk, ck)
            ok = d1 and d2 and d1[1] > 0 and d2[1] > 0
            print(f"FORM '{fa}': vs nocue {d1[0]:+.2f} [{d1[1]:+.2f},{d1[2]:+.2f}], vs {ctrl} {d2[0]:+.2f} [{d2[1]:+.2f},{d2[2]:+.2f}] -> "
                  + ("ADOPT: python run_acoustic.py --dataset %s%s --form %s%s --variants clue shuffled --erc_epochs 3"
                     % (a.dataset, t, {"desc": "description", "desc_ctx": "description", "both": "both"}[fa], " --audio_context" if fa == "desc_ctx" else "")
                     if ok else "not supported (does not beat both controls)"))
    print()
    if adopt:
        best = max(adopt, key=base_f.get)
        print(f"VERDICT: '{best}' beats nocue and shuffled (95% interval above 0). To use it:")
        print(f'  cd ../base+acoustic && python run_acoustic.py --dataset {a.dataset}{t} --config "{best}" --variants clue shuffled --erc_epochs 3')
        print(f'  cd ../base+chain    && python run_chain.py --dataset {a.dataset}{t} --variants all --acoustic --acoustic_config "{best}" --erc_epochs 3')
        print("  (the base run and its stage-1 adapter must already exist; --erc_epochs must equal the base run's)")
    elif any(ac._SLUG.match(k) for k in P):
        print("VERDICT: no threshold config beats both controls with a 95% interval above zero -> keep the default recipe "
              "(run_acoustic.py without --config). Differences between configs are within noise.")


if __name__ == "__main__":
    main()
