#!/usr/bin/env python3
"""CoE Strategy B driver (Mixed Auxiliary):
  0. build data            (build_coe_data.py; scenes from generate_scenes.py)
  1. seed rationales       (generate_seed_rationales.py)
  2. STEeR loop            (run_local_steer.py)  -> memory.json = real self-generated rationales
  3. rebuild aux data with the STEeR rows added
  4. stage 1: joint auxiliary training (speaker-ID + role-play + STEeR), no ERC
  5. stage 2: ERC fine-tuning starting from the stage-1 adapter, evaluated on the test set
Re-running the same command resumes: finished steps are skipped, and an interrupted training stage continues
from its latest saved epoch adapter (optimizer state and LR schedule restart; the STEeR loop restarts its
model from scratch but keeps the rationales it already collected).
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "Truebaseline", "outputs")
DATA = os.path.join(HERE, "Truebaseline", "data")


DRY = False
MP = []  # ["--model_path", path] when given


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    if not DRY:
        subprocess.run(cmd, check=True, cwd=HERE)


def latest_epoch(out_dir):
    """Highest k with a saved lora_epoch_k adapter in out_dir, or None."""
    if not os.path.isdir(out_dir):
        return None
    ks = [int(m.group(1)) for n in os.listdir(out_dir) for m in [re.fullmatch(r"lora_epoch_(\d+)", n)] if m]
    return max(ks) if ks else None


def train_stage(name, data_sub, dataset, epochs, percent, port, start_ckpt=None, extra=(), lr=None):
    """Run one training stage; skip it if finished, continue from the latest epoch adapter if interrupted."""
    out = os.path.join(OUT, name, data_sub)
    k = latest_epoch(out)
    # finished stage: skip, unless MORE epochs were requested than were trained -> extend it from its last epoch adapter
    if os.path.exists(os.path.join(out, "final_checkpoint")) and (k is None or k + 1 >= epochs):
        print(f"[skip] {name}: final_checkpoint exists", flush=True)
        return
    ckpt, offset = start_ckpt, 0
    if k is not None:
        offset = k + 1
        ckpt = os.path.join(out, f"lora_epoch_{k}")
        if offset >= epochs:
            print(f"[skip] {name}: all {epochs} epochs already saved (lora_epoch_{k})", flush=True)
            return
        print(f"[resume] {name}: continuing after epoch {k} from {ckpt}; {epochs - offset} epoch(s) left", flush=True)
    cmd = [PY, "run_local_combined.py", "--data_subdir", data_sub, "--run_name", name, "--dataset", dataset,
           "--epochs", str(epochs - offset), "--epoch_offset", str(offset), "--data_percent", str(percent),
           "--port", str(port)] + MP + list(extra)
    if ckpt:
        cmd += ["--checkpoint_dir", ckpt]
    if lr:
        cmd += ["--lr", str(lr)]
    run(cmd)


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
    ap.add_argument("--erc_lr", default=None, help="learning rate for stage 2 (default 2e-4); use ~1e-4 when extending a finished run")
    ap.add_argument("--eval_percent", type=float, default=1.0, help="share of test set scored in stage 2")
    ap.add_argument("--skip_steer", action="store_true", help="aux stage without STEeR rows")
    ap.add_argument("--model_path", default=None, help="local Mistral-7B-Instruct-v0.3 folder (default: /home/user2/Mistral-7B-Instruct-v0.3)")
    ap.add_argument("--dry_run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()
    global DRY, MP
    DRY = args.dry_run
    MP = ["--model_path", args.model_path] if args.model_path else []
    base = f"{args.dataset}_coe{args.tag}"
    flags = {"_noscene": ["--no_scene"], "_nopersona": ["--no_persona"], "_scene": ["--iemocap_scene"]}.get(args.tag, [])

    if not os.path.exists(os.path.join(DATA, f"{base}_erc", "train.json")):
        run([PY, "build_coe_data.py", "--dataset", args.dataset] + flags)

    memory = None
    if not args.skip_steer:
        steer_dir = f"{base}_steer"
        if not os.path.exists(os.path.join(DATA, steer_dir, "trainbegin.json")):
            run([PY, "generate_seed_rationales.py", "--dataset", args.dataset, "--steer_dir", steer_dir] + MP)
        memory = os.path.join(OUT, f"STEeR_{args.dataset}{args.tag}", steer_dir, "memory.json")
        if not os.path.exists(os.path.join(OUT, f"STEeR_{args.dataset}{args.tag}", steer_dir, "final_checkpoint")):
            run([PY, "run_local_steer.py", "--dataset", args.dataset, "--data_subdir", steer_dir, "--run_name", f"STEeR_{args.dataset}{args.tag}",
                 "--epochs", str(args.steer_epochs), "--infer_per_epoch", str(args.infer_per_epoch)] + MP)
        run([PY, "build_coe_data.py", "--dataset", args.dataset, "--steer_rows", memory] + flags)

    s1 = os.path.join(OUT, f"Stage1_aux_{args.dataset}{args.tag}", f"{base}_aux", "final_checkpoint")
    train_stage(f"Stage1_aux_{args.dataset}{args.tag}", f"{base}_aux", args.dataset, args.aux_epochs,
                args.aux_percent, 29511, extra=["--no_eval"])
    # stage 2 starts from the stage-1 adapter; if interrupted it continues from its own latest epoch
    train_stage(f"Stage2_erc_{args.dataset}{args.tag}", f"{base}_erc", args.dataset, args.erc_epochs,
                args.erc_percent, 29512, start_ckpt=s1, extra=["--eval_percent", str(args.eval_percent)], lr=args.erc_lr)


if __name__ == "__main__":
    main()
