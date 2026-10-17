#!/usr/bin/env python3
"""GPU ablation (NOT run automatically): which threshold configuration gives the LLM the most useful clues?

Every arm starts from the same stage-1 adapter, fine-tunes on the training part of the ERC data for the same budget and
is scored on the held-out dev rows (never the test set). Arms:
  nocue      same fine-tune, no clue                  -> what plain extra training gives
  shuffled   default thresholds, clues on random turns -> what 'extra text' gives
  default    the dataset's original recipe
  + the top-k configs of the CPU screen (results/{dataset}_screen.csv) or --configs "name" ...
Rough cost per arm on the 3060: ~1.2 h for EmoryNLP/MELD at --percent 0.3, plus ~10-20 min dev evaluation.
Results: python summarize_ablation.py --dataset D   (paired bootstrap against nocue and shuffled).

  python run_ablation.py --dataset iemocap --top_k 2 --percent 0.3
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
import build_ablation_data as bld  # noqa: E402
import configs  # noqa: E402
import run_strategy_b as rsb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--tag", default="", help="base run tag ('' or _scene for IEMOCAP)")
    ap.add_argument("--top_k", type=int, default=2, help="screen shortlist size")
    ap.add_argument("--configs", nargs="*", default=None, help="explicit config names instead of the shortlist")
    ap.add_argument("--forms", nargs="*", default=[], choices=list(bld.FORM_ARMS),
                    help="SpeechCueLLM-style arms to add, e.g. --forms desc shuffled_desc (each needs its shuffled_* control)")
    ap.add_argument("--speechcue_ablation", action="store_true",
                    help="add the full component ablation of the SpeechCueLLM-style clue (preset for this dataset)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--percent", type=float, default=0.3, help="share of the training part used per arm")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    rsb.DRY = a.dry_run
    rsb.MP = ["--model_path", a.model_path] if a.model_path else []

    s1 = os.path.join(rsb.OUT, f"Stage1_aux_{a.dataset}{a.tag}", f"{a.dataset}_coe{a.tag}_aux", "final_checkpoint")
    if not os.path.exists(s1) and not a.dry_run:
        sys.exit(f"stage-1 adapter missing: {s1}")
    if a.configs is not None:
        cfgs = a.configs
    elif os.path.exists(os.path.join(bld.HERE, "results", f"{a.dataset}_screen.csv")):
        cfgs = configs.shortlist(a.dataset, a.top_k)
    else:
        sys.exit(f"no screen results for {a.dataset}: run screen_thresholds.py --dataset {a.dataset} or pass --configs")
    forms = list(a.forms) + (bld.preset_for(a.dataset)[0] if a.speechcue_ablation else [])
    arms = ["nocue", "shuffled", "default"] + list(dict.fromkeys(cfgs)) + list(dict.fromkeys(forms))
    print("arms:", arms)
    for i, arm in enumerate(arms):
        data = bld.arm_dir(a.dataset, arm, a.tag)
        if not os.path.exists(os.path.join(rsb.DATA, data, "train.json")):
            if a.dry_run:
                print(f"[dry-run] would build {data}")
            else:
                bld.build_arm(a.dataset, arm, a.tag)
        rsb.train_stage(f"Abl_{a.dataset}{a.tag}_{configs.slug(arm)}", data, a.dataset, a.epochs, a.percent, 29530 + i,
                        start_ckpt=s1, extra=["--eval_percent", "1.0"])


if __name__ == "__main__":
    main()
