#!/usr/bin/env python3
"""Chain-of-Emotion ERC runs, directly comparable to the base run (and to base+acoustic).

Each arm starts from the SAME stage-1 adapter as the base stage 2 with the same ERC hyper-parameters; only the prompt
instruction and the target (chain + label) differ. Compare against Stage2_erc_<D>  (base)  and, with --acoustic, against
Stage2_acoustic_clue_<D>.  Resumable: re-run the same command.

  python run_chain.py --dataset iemocap --variants self all prev --erc_epochs 3
  python run_chain.py --dataset iemocap --variants all --acoustic --erc_epochs 3      # chain + acoustic clues
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base"))
sys.path.insert(0, os.path.join(HERE, "..", "base+acoustic"))
sys.path.insert(0, os.path.join(HERE, "..", "base+rel"))
import acoustic_clues as ac  # noqa: E402
import build_acoustic_data as bad  # noqa: E402
import build_chain_data as bcd  # noqa: E402
import build_rel_data as brd  # noqa: E402
import chain_data as cd  # noqa: E402
import run_strategy_b as rsb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--tag", default="", help="same tag as the base run ('' or _scene for IEMOCAP)")
    ap.add_argument("--variants", nargs="+", default=["self", "all", "prev"], choices=cd.VARIANTS)
    ap.add_argument("--acoustic", action="store_true", help="chain on top of the acoustic-clue prompts (base+chain+acoustic)")
    ap.add_argument("--rel", action="store_true", help="chain on top of the character-relationship prompts (base+rel); combine with --acoustic for all three clues")
    ap.add_argument("--acoustic_config", default=None, help="with --acoustic: use clues made with this ablation-chosen threshold config (name or slug)")
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
    acoustic = a.acoustic or bool(a.acoustic_config)
    cfg, slug = ac.parse_config(a.acoustic_config, a.dataset) if a.acoustic_config else (None, None)
    acx = slug if slug else acoustic            # False | True | config slug
    if acoustic and not os.path.exists(os.path.join(rsb.DATA, bcd.src_name(a.dataset, a.tag, acx), "train.json")):
        if a.dry_run:
            print(f"[dry-run] would build {bcd.src_name(a.dataset, a.tag, acx)}")
        else:
            bad.build(a.dataset, "clue", a.tag, cfg=cfg, slug=slug)
    if a.rel and not os.path.exists(os.path.join(rsb.DATA, bcd.src_name(a.dataset, a.tag, acx, True), "train.json")):
        if a.dry_run:
            print(f"[dry-run] would build {bcd.src_name(a.dataset, a.tag, acx, True)}")
        else:
            if acx not in (False, True):
                sys.exit("--rel with --acoustic_config is not supported; use the default acoustic recipe")
            brd.build(a.dataset, a.tag, bool(acx))
    for i, v in enumerate(a.variants):
        data = bcd.out_name(a.dataset, a.tag, v, acx, a.rel)
        if not os.path.exists(os.path.join(rsb.DATA, data, "train.json")):
            if a.dry_run:
                print(f"[dry-run] would build {data}")
            else:
                bcd.build(a.dataset, v, a.tag, acx, a.rel)
        name = f"Stage2_chain_{v}{('_acoustic' + ('_' + slug if slug else '')) if acoustic else ''}{'_rel' if a.rel else ''}_{a.dataset}{a.tag}"
        rsb.train_stage(name, data, a.dataset, a.erc_epochs, a.erc_percent, 29540 + i, start_ckpt=s1,
                        extra=["--eval_percent", str(a.eval_percent), "--max_new_tokens", str(cd.MAX_NEW_TOKENS[v])])


if __name__ == "__main__":
    main()
