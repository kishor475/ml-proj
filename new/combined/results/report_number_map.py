#!/usr/bin/env python3
"""Map every number in writeup/main.tex to the prediction file(s) it comes from and recompute it.

Writes results/REPORT_NUMBERS.md: one row per reported quantity with the report location, the reported value, the value recomputed
from the saved predictions, the source file(s), and MATCH / MISMATCH.

  python results/report_number_map.py          # scores only (fast)
  python results/report_number_map.py --ci     # also recompute the paired-bootstrap deltas and 95 % intervals (several minutes)

Weighted F1 is recomputed from the prediction rows with the same label extraction the report tables used (make_results_table.py).
Bootstrap intervals use a fixed seed per row, so they can differ from the report by a few hundredths; the tolerance is 0.3.
Absolute scores in the report are the trainer's printed value (first line of each prediction file); recomputing from the rows can differ by up to
0.04 (for example IEMOCAP base 69.97 printed, 69.93 recomputed). The cause (label parsing of the generated text) was not investigated; the tolerance is 0.045.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import make_results_table as mrt  # noqa: E402

OUT = mrt.rsb.OUT
CI_TOL, SCORE_TOL = 0.3, 0.045

# ---- everything the report states, with its source -----------------------------------------------------------------------------
# (report location, quantity, dataset, run folder, epoch (None = last), reported wF1, reference folder, reference epoch, reported (delta, lo, hi))
S2 = lambda arm, ds: f"Stage2_{arm}_{ds}"  # noqa: E731
CH = {
    "emorynlp": {"chain_all": (-3.53, -6.10, -0.95), "chain_prev": (-1.69, -3.97, 0.61), "chain_self": (-1.85, -4.28, 0.50), "chain_lf_all": (-3.44, -6.03, -0.86)},
    "meld": {"chain_all": (-1.15, -2.54, 0.36), "chain_prev": (-0.78, -2.24, 0.65), "chain_self": (-1.98, -3.37, -0.56)},
    "iemocap": {"chain_all": (-3.81, -5.88, -1.70), "chain_prev": (-0.88, -2.64, 0.92), "chain_self": (-1.27, -3.04, 0.51), "chain_lf_all": (-3.95, -5.92, -2.00)},
}
ROWS = []
# Table I: base vs reported (final epoch = epoch 2; best epoch listed separately)
for ds, final, best, best_ep in (("emorynlp", 40.00, 40.00, 2), ("meld", 67.35, 67.76, 0), ("iemocap", 69.97, 69.97, 2)):
    ROWS.append(("Table I", f"{ds} base, final epoch", ds, S2("erc", ds), 2, final, None, None, None))
    ROWS.append(("Table I", f"{ds} base, best epoch", ds, S2("erc", ds), best_ep, best, None, None, None))
# Table II: attempts to close the EmoryNLP gap (delta vs v0.3 base, final epoch)
B = S2("erc", "emorynlp")
ROWS += [
    ("Table II", "continued to 8 epochs (final = epoch 7)", "emorynlp", "Stage2_erc_emorynlp_ext8", 7, 39.27, B, 2, (-0.81, -2.71, 1.06)),
    ("Table II", "LoRA rank 16", "emorynlp", "Stage2_erc_emorynlp_r16", 2, 38.55, B, 2, (-1.42, -3.67, 0.60)),
    ("Table II", "Mistral-7B v0.1, 3 epochs", "emorynlp", "Stage2_erc_emorynlp_v01", 2, 39.79, B, 2, (-0.21, -2.31, 2.13)),
    ("Table II", "Mistral-7B v0.1, 6 epochs", "emorynlp", "Stage2_erc_emorynlp_v01", 5, 40.16, B, 2, (0.18, -2.26, 2.69)),
]
for e, v in ((3, 37.82), (4, 38.96)):
    ROWS.append(("Sec. V-B (warm restart)", f"v0.1 base, epoch {e}", "emorynlp", "Stage2_erc_emorynlp_v01", e, v, None, None, None))
# Table III: chain of emotion at full length (final epoch delta vs v0.3 base)
for ds, arms in CH.items():
    for arm, d in arms.items():
        ROWS.append(("Table III", f"{ds} {arm}", ds, f"Stage2_{arm}_{ds}", 2, None, S2("erc", ds), 2, d))
# IEMOCAP scene-relationship clue at full length (final epoch 69.72 = 69.97 - 0.25 raw; bootstrap delta vs the base)
ROWS.append(("Table V / Sec. VI", "iemocap scene relation, full length", "iemocap", "Stage2_rel_iemocap", 2, 69.72, S2("erc", "iemocap"), 2, (-0.23, -1.87, 1.38)))
# Table IV: v0.1 arms (epochs 0-2 and delta of epoch 2 vs the v0.1 base)
V = "Stage2_erc_emorynlp_v01"
for name, folder, eps, d in (
        ("v0.1 base", V, (36.08, 38.88, 39.79), None),
        ("acoustic", "Stage2_acoustic_clue_emorynlp_v01", (36.67, 39.65, 41.42), (1.62, -0.45, 3.79)),
        ("relation map", "Stage2_relmap_detail_emorynlp_v01", (36.85, 39.58, 39.43), (-0.34, -2.32, 1.66)),
        ("chain-prev", "Stage2_chain_prev_emorynlp_v01", (37.03, 38.45, 38.26), (-1.44, -3.82, 0.95)),
        ("chain-all", "Stage2_chain_all_emorynlp_v01", (34.16, 34.70, 36.27), (-3.46, -5.88, -1.01))):
    for e, v in enumerate(eps):
        last = e == 2
        ROWS.append(("Table IV", f"{name}, epoch {e}", "emorynlp", folder, e, v, V if (last and d) else None, 2 if (last and d) else None, d if last else None))
# Table V: short-protocol ablation (dev wF1; delta vs the base run at the short budget (output dir Abl_<ds>_nocue) of the same dataset; single epoch -> last = only epoch)
ABL = {
    "emorynlp": [("nocue", 39.44, None), ("lf-all", 41.95, (2.57, 0.81, 4.48)), ("chain-all", 41.85, (2.44, 0.45, 4.45)), ("lf-prev", 41.11, (1.69, 0.04, 3.24)),
                 ("chain-prev", 40.19, (0.73, -0.95, 2.52)), ("relmap", 39.73, (0.30, -0.89, 1.52)), ("desc", 39.34, (-0.07, -1.24, 1.09)),
                 ("shuffled-desc", 38.68, (-0.75, -2.00, 0.35)), ("shuffled-relmap", 39.24, (-0.19, -1.26, 0.83)), ("relmap-desc", 37.61, (-1.86, -3.22, -0.52))],
    "iemocap": [("nocue", 60.24, None), ("chain-all", 64.67, (4.46, 2.23, 6.75)), ("chain-all-known", 64.26, (4.00, 1.63, 6.51)), ("lf-all", 64.11, (3.94, 1.74, 6.07)),
                ("lf-prev", 62.09, (1.83, -0.01, 3.66)), ("chain-prev", 58.54, (-1.66, -3.85, 0.61)), ("desc", 59.31, (-0.95, -2.68, 0.84)),
                ("default", 57.62, (-2.60, -4.78, -0.57)), ("shuffled-desc", 56.25, (-3.93, -5.67, -2.23))],
    "meld": [("nocue", 68.97, None), ("lf-all", 69.20, (0.26, -1.64, 2.15)), ("chain-prev", 65.81, (-3.19, -5.34, -1.03))],
}
for ds, arms in ABL.items():
    for arm, v, d in arms:
        ROWS.append(("Table V / Sec. VI", f"{ds} {'no-clue control (nocue)' if arm == 'nocue' else arm} (short protocol, dev)", ds, f"Abl_{ds}_{arm}", None, v, f"Abl_{ds}_nocue" if d else None, None, d))


def locate(folder, epoch):
    eps = mrt.epochs(os.path.join(OUT, folder))
    if not eps:
        return None, None, None
    e = max(eps) if epoch is None else epoch
    if e not in eps:
        return None, None, None
    return e, eps[e][0], eps[e][1]


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true", help="recompute paired-bootstrap deltas and 95 %% intervals (slow)")
    args = ap.parse_args()
    lines = ["# Where every number in the report comes from\n",
             "Generated by `results/report_number_map.py`. Each row gives the report location, the reported value, the value recomputed from the saved "
             "prediction file, and the file. Paths are relative to `new/combined/`; `<epoch>_<score>` in a file name is the epoch index and the score the "
             "trainer printed for that epoch.\n",
             "Weighted F1 is recomputed from the prediction rows (final label parsed from each row's output). Deltas are paired-bootstrap differences "
             "(1000 resamples, 95 % percentile interval) between two prediction files over the same test rows; "
             + ("recomputed here with a fixed seed (tolerance 0.3 on the interval)" if args.ci else "run with `--ci` to recompute them") + ".\n",
             "| Report location | Quantity | Reported | Recomputed | Source file | Reference file | Check |", "|---|---|---|---|---|---|---|"]
    bad = 0
    cache = {}
    for loc, qty, ds, folder, epoch, rep, ref_folder, ref_epoch, delta in ROWS:
        labels = mrt.cfg(ds)["labels"]
        e, header, path = locate(folder, epoch)
        if path is None:
            lines.append(f"| {loc} | {qty} | {rep} | - | `{folder}` (missing) | | MISSING |")
            bad += 1
            continue
        if path not in cache:
            cache[path] = mrt.preds(path, labels)
        p, g = cache[path]
        w = mrt.an.wf1(g, p)
        cells, ok = [], True
        if rep is not None:
            cells.append(f"{rep:.2f}")
            okv = abs(w - rep) <= SCORE_TOL + 0.005
            ok &= okv
            rec = f"{w:.2f}"
        else:
            cells.append(f"final epoch {e}")
            rec = f"{w:.2f}"
        ref_txt = ""
        if delta is not None and ref_folder:
            re_, rh, rpath = locate(ref_folder, ref_epoch)
            ref_txt = f"`{rel(rpath)}`" if rpath else "(missing)"
            cells.append(f"{delta[0]:+.2f} [{delta[1]:+.2f}, {delta[2]:+.2f}]")
            if args.ci and rpath:
                if rpath not in cache:
                    cache[rpath] = mrt.preds(rpath, labels)
                pr, gr = cache[rpath]
                if len(pr) == len(p) and (gr == g).all():
                    m, lo, hi = mrt.boot(g, p, pr, np.random.RandomState(0))
                    okd = max(abs(m - delta[0]), abs(lo - delta[1]), abs(hi - delta[2])) <= CI_TOL
                    ok &= okd
                    rec += f" ; {m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                else:
                    ok = False
                    rec += " ; row mismatch with reference"
            else:
                rec += " ; (not recomputed)"
        lines.append(f"| {loc} | {qty} | {' ; '.join(cells)} | {rec} | `{rel(path)}` | {ref_txt} | {'MATCH' if ok else 'MISMATCH'} |")
        bad += (not ok)
    lines += ["", f"**{len(ROWS) - bad} of {len(ROWS)} rows match.**\n", NOT_FROM_PREDS]
    path = os.path.join(HERE, "REPORT_NUMBERS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(ROWS) - bad}/{len(ROWS)} rows match -> {path}")


NOT_FROM_PREDS = """## Numbers that do not come from a prediction file

| Number in the report | Where it comes from |
|---|---|
| Reported scores 44.29 / 70.11 / 70.46 (Table I) | the CoE paper (ACL 2025, `2025.acl-long.1148`), results table |
| Paper setup: fp16, LoRA rank 16, 20 ERC epochs, three-run average, V100 cluster | the CoE paper, Section 5.1.3 (training setup) |
| Dataset sizes (Section IV): EmoryNLP 9,934 / 1,344 / 1,328; MELD 9,989 / 1,109 / 2,610; IEMOCAP 5,758 / 1,622 / 1,622 | count of ERC rows per split in `base/source_data/<dataset>/{train,dev,test}.json` (rows whose prompt asks for the emotion of the current utterance) |
| Training setup (rank 8, alpha 16, lr 2e-4, batch 2 x 8 accumulation, max length 1400, ZeRO-2, 4-bit NF4) | defaults in `base/run_local_combined.py` and `base/Truebaseline/src/erc_trainer.py`; stage-1 epochs / data share in `base/run_strategy_b.py` |
| Cost (about 23 GPU-hours EmoryNLP pipeline, 12 h per EmoryNLP/MELD arm, 4-5 h IEMOCAP) | measured and estimated times in `README.md`, section 5 |
| Ensemble remark in the Discussion (+0.19, not significant, with final epochs) | `base+chain/ensemble_eval.py`, MELD, final-epoch option |
| Chain-variant definitions (Section III-B) | `base+chain/chain_data.py` (`VARIANTS` and the prompt text for each) |
| Prompts in the appendix | `base/build_coe_data.py` (ERC, speaker, role-play, rationale prompts) and `base/generate_scenes.py` (scene prompt) |

## How to reproduce a row by hand

1. Open the source file listed for the row; the first line holds the trainer's weighted-F1 for that epoch, followed by a classification report.
2. The table value is the weighted F1 recomputed from the individual predictions in the file (`make_results_table.py` uses the same functions).
3. For a delta, compare the source file with the reference file on the same rows (paired bootstrap, `make_results_table.py: boot`).
"""

if __name__ == "__main__":
    main()
