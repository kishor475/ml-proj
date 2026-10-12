#!/usr/bin/env python3
"""CoE Strategy B driver (Mixed Auxiliary):
  0. build data            (build_coe_data.py; scenes from generate_scenes.py)
  1. seed rationales       (generate_seed_rationales.py)
  2. STEeR loop            (run_local_steer.py)  -> memory.json = real self-generated rationales
  3. rebuild aux data with the STEeR rows added
  4. stage 1: joint auxiliary training (speaker-ID + role-play + STEeR), no ERC
  5. stage 2: ERC fine-tuning starting from the stage-1 adapter, evaluated on the test set
Each step is skipped if its output already exists, so the script can be re-run to resume.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "Truebaseline", "outputs")
DATA = os.path.join(HERE, "Truebaseline", "data")


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="emorynlp", choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--tag", default="", help="'' (persona+scene) | _noscene | _nopersona | _scene (IEMOCAP only: transcript-derived scene)")
    ap.add_argument("--steer_epochs", type=int, default=4)
    ap.add_argument("--infer_per_epoch", type=int, default=1500)
    ap.add_argument("--aux_epochs", type=int, default=2)
    ap.add_argument("--aux_percent", type=float, default=1.0)
    ap.add_argument("--erc_epochs", type=int, default=5)
    ap.add_argument("--erc_percent", type=float, default=1.0)
    ap.add_argument("--eval_percent", type=float, default=1.0, help="share of test set scored in stage 2")
    ap.add_argument("--skip_steer", action="store_true", help="aux stage without STEeR rows")
    args = ap.parse_args()
    base = f"{args.dataset}_coe{args.tag}"
    flags = {"_noscene": ["--no_scene"], "_nopersona": ["--no_persona"], "_scene": ["--iemocap_scene"]}.get(args.tag, [])

    if not os.path.exists(os.path.join(DATA, f"{base}_erc", "train.json")):
        run([PY, "build_coe_data.py", "--dataset", args.dataset] + flags)

    memory = None
    if not args.skip_steer:
        steer_dir = f"{base}_steer"
        if not os.path.exists(os.path.join(DATA, steer_dir, "trainbegin.json")):
            run([PY, "generate_seed_rationales.py", "--dataset", args.dataset, "--steer_dir", steer_dir])
        memory = os.path.join(OUT, f"STEeR_{args.dataset}{args.tag}", steer_dir, "memory.json")
        if not os.path.exists(os.path.join(OUT, f"STEeR_{args.dataset}{args.tag}", steer_dir, "final_checkpoint")):
            run([PY, "run_local_steer.py", "--dataset", args.dataset, "--data_subdir", steer_dir, "--run_name", f"STEeR_{args.dataset}{args.tag}",
                 "--epochs", str(args.steer_epochs), "--infer_per_epoch", str(args.infer_per_epoch)])
        run([PY, "build_coe_data.py", "--dataset", args.dataset, "--steer_rows", memory] + flags)

    s1 = os.path.join(OUT, f"Stage1_aux_{args.dataset}{args.tag}", f"{base}_aux", "final_checkpoint")
    if not os.path.exists(s1):
        run([PY, "run_local_combined.py", "--data_subdir", f"{base}_aux", "--run_name", f"Stage1_aux_{args.dataset}{args.tag}",
             "--dataset", args.dataset, "--epochs", str(args.aux_epochs), "--data_percent", str(args.aux_percent),
             "--no_eval", "--port", "29511"])
    run([PY, "run_local_combined.py", "--data_subdir", f"{base}_erc", "--run_name", f"Stage2_erc_{args.dataset}{args.tag}",
         "--dataset", args.dataset, "--epochs", str(args.erc_epochs), "--data_percent", str(args.erc_percent),
         "--eval_percent", str(args.eval_percent), "--checkpoint_dir", s1, "--port", "29512"])


if __name__ == "__main__":
    main()
