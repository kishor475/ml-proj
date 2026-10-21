#!/usr/bin/env python3
"""Relationship-clue ERC run, directly comparable to the base run (same stage-1 adapter, same ERC hyper-parameters; only the prompt differs).

  python run_rel.py --dataset emorynlp --erc_epochs 3            # base + relationships
  python run_rel.py --dataset meld --acoustic --erc_epochs 3     # base + relationships + acoustic clues
  python run_rel.py --dataset iemocap --tag _scene --erc_epochs 3   # IEMOCAP: relationship per scene (needs the transcript alignment)
Compare Stage2_rel_<D>[ _acoustic] with Stage2_erc_<D> (base) via base+chain/analyze_chain.py. Resumable: re-run the same command.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
sys.path.insert(0, os.path.join(HERE, "..", "base+acoustic"))
import build_acoustic_data as bad  # noqa: E402
import build_rel_data as brd  # noqa: E402
import run_strategy_b as rsb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--tag", default="", help="same tag as the base run ('' or _scene for IEMOCAP)")
    ap.add_argument("--acoustic", action="store_true", help="add the relationship block to the acoustic-clue prompts")
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
    if a.acoustic and not os.path.exists(os.path.join(rsb.DATA, brd.src_name(a.dataset, a.tag, True), "train.json")):
        if a.dry_run:
            print(f"[dry-run] would build {brd.src_name(a.dataset, a.tag, True)}")
        else:
            bad.build(a.dataset, "clue", a.tag)
    data = brd.out_name(a.dataset, a.tag, a.acoustic)
    if not os.path.exists(os.path.join(rsb.DATA, data, "train.json")):
        if a.dry_run:
            print(f"[dry-run] would build {data}")
        else:
            brd.build(a.dataset, a.tag, a.acoustic)
    rsb.train_stage(f"Stage2_rel{'_acoustic' if a.acoustic else ''}_{a.dataset}{a.tag}", data, a.dataset, a.erc_epochs,
                    a.erc_percent, 29560, start_ckpt=s1, extra=["--eval_percent", str(a.eval_percent)])


if __name__ == "__main__":
    main()
