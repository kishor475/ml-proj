#!/usr/bin/env python3
"""Acoustic-clue ERC runs, directly comparable to the base run.

Each arm starts from the SAME stage-1 (auxiliary) adapter as the base stage 2 and uses the same ERC hyper-parameters;
only the prompt differs (extra acoustic clue). Arms:
  clue      real acoustic clues            (the experiment)
  shuffled  clues assigned to random turns (control: separates 'uses the audio' from 'extra text / extra training')
The base arm (no clues) is the existing  Stage2_erc_{dataset}{tag}  run.

Resumable like base/run_strategy_b.py: re-running the same command skips finished stages and continues an interrupted
one from its latest saved epoch adapter.

  python run_acoustic.py --dataset iemocap --variants clue shuffled --erc_epochs 3
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
import acoustic_clues as ac  # noqa: E402
import build_acoustic_data as bad  # noqa: E402
import run_strategy_b as rsb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--tag", default="", help="same tag as the base run ('' or _scene for IEMOCAP)")
    ap.add_argument("--variants", nargs="+", default=["clue", "shuffled"], choices=["clue", "shuffled"])
    ap.add_argument("--config", default=None, help="threshold config chosen by the ablation (name or slug); default = original recipe")
    ap.add_argument("--form", default="impression", choices=ac.FORMS,
                    help="impression (original) | description (SpeechCueLLM headline form) | both")
    ap.add_argument("--audio_context", action="store_true", help="SpeechCueLLM audio_context tags on the last 3 context turns")
    ap.add_argument("--erc_epochs", type=int, default=3, help="must equal the base run's --erc_epochs")
    ap.add_argument("--erc_percent", type=float, default=1.0)
    ap.add_argument("--eval_percent", type=float, default=1.0)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    rsb.DRY = a.dry_run
    rsb.MP = ["--model_path", a.model_path] if a.model_path else []

    s1 = os.path.join(rsb.OUT, f"Stage1_aux_{a.dataset}{a.tag}", f"{a.dataset}_coe{a.tag}_aux", "final_checkpoint")
    if not os.path.exists(s1) and not a.dry_run:
        sys.exit(f"stage-1 adapter missing: {s1}\nFinish the base run (run_strategy_b.py --dataset {a.dataset}) first.")
    cfg, slug = ac.parse_config(a.config, a.dataset) if a.config else (None, None)
    for i, v in enumerate(a.variants):
        data = bad.data_name(a.dataset, a.tag, v, slug, a.form, a.audio_context)
        if not os.path.exists(os.path.join(rsb.DATA, data, "train.json")):
            if a.dry_run:
                print(f"[dry-run] would build {data}")
            else:
                bad.build(a.dataset, v, a.tag, cfg=cfg, slug=slug, form=a.form, ctx=a.audio_context)
        rsb.train_stage(f"Stage2_acoustic_{v}{bad._FORM_TAG[a.form]}{'_ctx' if a.audio_context else ''}{'_' + slug if slug else ''}_{a.dataset}{a.tag}", data, a.dataset, a.erc_epochs, a.erc_percent,
                        29520 + i, start_ckpt=s1, extra=["--eval_percent", str(a.eval_percent)])


if __name__ == "__main__":
    main()
