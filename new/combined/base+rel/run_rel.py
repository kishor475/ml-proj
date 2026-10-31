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
import build_relmap_data as brm  # noqa: E402
import relmap  # noqa: E402
import run_strategy_b as rsb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--tag", default="", help="same tag as the base run ('' or _scene for IEMOCAP)")
    ap.add_argument("--map", default="kb", choices=["kb", "user"],
                    help="kb = the small built-in knowledge base (relations.py); user = the hand-built map files (EmoryNLP / MELD)")
    ap.add_argument("--style", default="detail", choices=relmap.STYLES, help="with --map user: how each line is written")
    ap.add_argument("--acoustic", action="store_true", help="add the relationship block to the acoustic-clue prompts")
    ap.add_argument("--erc_epochs", type=int, default=3, help="must equal the base run's --erc_epochs")
    ap.add_argument("--erc_percent", type=float, default=1.0)
    ap.add_argument("--eval_percent", type=float, default=1.0)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--suffix", default="", help="run-name suffix of the base-model variant (e.g. _v01): start from Stage1_aux_<D>_v01 and write Stage2_*_<D>_v01")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    rsb.DRY = a.dry_run
    rsb.MP = ["--model_path", a.model_path] if a.model_path else []

    s1 = os.path.join(rsb.OUT, f"Stage1_aux_{a.dataset}{a.tag}{a.suffix}", f"{a.dataset}_coe{a.tag}_aux", "final_checkpoint")
    if not os.path.exists(s1) and not a.dry_run:
        sys.exit(f"stage-1 adapter missing: {s1}\nFinish the base run (run_strategy_b.py --dataset {a.dataset}) first.")
    if a.acoustic and not os.path.exists(os.path.join(rsb.DATA, brd.src_name(a.dataset, a.tag, True), "train.json")):
        if a.dry_run:
            print(f"[dry-run] would build {brd.src_name(a.dataset, a.tag, True)}")
        else:
            bad.build(a.dataset, "clue", a.tag)
    if a.map == "user":
        if a.dataset == "iemocap" or a.acoustic:
            sys.exit("--map user is for EmoryNLP / MELD and is not combined with --acoustic")
        data = brm.out_name(a.dataset, a.style, a.tag)
        if not os.path.exists(os.path.join(rsb.DATA, data, "train.json")):
            if a.dry_run:
                print(f"[dry-run] would build {data}")
            else:
                brm.build(a.dataset, a.style, a.tag)
        rsb.train_stage(f"Stage2_relmap_{a.style}_{a.dataset}{a.tag}{a.suffix}", data, a.dataset, a.erc_epochs, a.erc_percent, 29561,
                        start_ckpt=s1, extra=["--eval_percent", str(a.eval_percent)])
        return
    data = brd.out_name(a.dataset, a.tag, a.acoustic)
    if not os.path.exists(os.path.join(rsb.DATA, data, "train.json")):
        if a.dry_run:
            print(f"[dry-run] would build {data}")
        else:
            brd.build(a.dataset, a.tag, a.acoustic)
    rsb.train_stage(f"Stage2_rel{'_acoustic' if a.acoustic else ''}_{a.dataset}{a.tag}{a.suffix}", data, a.dataset, a.erc_epochs,
                    a.erc_percent, 29560, start_ckpt=s1, extra=["--eval_percent", str(a.eval_percent)])


if __name__ == "__main__":
    main()
