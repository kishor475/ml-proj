#!/usr/bin/env python3
"""
==============================================================================
LOCAL STEeR RUNNER (RTX 3060 12GB VRAM + Mistral-7B-Instruct-v0.3)
==============================================================================
Runs Self-Taught Emotion Rationale Reasoning (STEeR) curriculum learning locally.
Paper: 2025.acl-long.1148 "CoE: A Clue of Emotion Framework for ERC"

STEeR Process:
  1. Starts with seed rationales from trainbegin.json copied into memory.json.
  2. Trains model on memory.json.
  3. Predicts on train4infer.json; samples where model prediction matches gold
     emotion are accepted and added to memory.json with generated rationale.
  4. Evaluates on test set.
==============================================================================
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run STEeR locally on desktop GPU")
    parser.add_argument("--dataset", type=str, default="emorynlp", choices=["iemocap", "meld", "emorynlp"],
                        help="Dataset to train on (default: emorynlp)")
    parser.add_argument("--data_subdir", type=str, default=None,
                        help="Data folder under Truebaseline/data (e.g. emorynlp_coe_steer); default: --dataset")
    parser.add_argument("--run_name", type=str, default="STEeR", help="Output folder name under Truebaseline/outputs")
    parser.add_argument("--infer_per_epoch", type=int, default=1500, help="pool examples scored per STEeR epoch")
    parser.add_argument("--infer_batch", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--model_path", type=str, default="/home/user2/Mistral-7B-Instruct-v0.3",
                        help="Path to local pretrained Mistral model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of STEeR epochs")
    parser.add_argument("--data_percent", type=float, default=1.0,
                        help="Fraction of memory data to train per epoch")
    parser.add_argument("--eval_percent", type=float, default=0.05,
                        help="Fraction of test data to evaluate each epoch")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-GPU train batch size (2 fits 12GB VRAM)")
    parser.add_argument("--accum_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_batch", type=int, default=2,
                        help="Eval batch size")
    parser.add_argument("--max_length", type=int, default=1400,
                        help="Max sequence length")
    parser.add_argument("--lr", type=str, default="2e-4",
                        help="Learning rate")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Shortcut for smoke test (epochs=1, eval_percent=0.02)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to saved LoRA adapter checkpoint (e.g. from baseline)")
    parser.add_argument("--port", type=int, default=29501,
                        help="DeepSpeed master port")
    parser.add_argument("--use_deepspeed", action="store_true", default=True,
                        help="Use DeepSpeed launcher")

    args = parser.parse_args()

    if args.smoke_test:
        print("\n>>> STEeR SMOKE TEST MODE ACTIVATED <<<")
        args.epochs = 1
        args.data_percent = 0.5
        args.eval_percent = 0.02

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "Truebaseline")
    data_dir = os.path.join(base_dir, "data", args.data_subdir or args.dataset)
    out_dir = os.path.join(base_dir, "outputs", args.run_name, args.data_subdir or args.dataset)
    ds_config = os.path.join(base_dir, "config", "deepspeed_config.json")
    steer_trainer = os.path.join(base_dir, "src", "steer_trainer.py")

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("LOCAL STEeR RUNNER (Self-Taught Emotion Rationale Reasoning)")
    print(f"Dataset      : {args.dataset}")
    print(f"Model Path   : {args.model_path}")
    print(f"Data Dir     : {data_dir}")
    print(f"Output Dir   : {out_dir}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch_size} (accum: {args.accum_steps})")
    print(f"Checkpoint   : {args.checkpoint_dir}")
    print("=" * 70)

    # Sanity checks
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path '{args.model_path}' does not exist!")
        sys.exit(1)

    trainbegin_file = os.path.join(data_dir, "trainbegin.json")
    if not os.path.exists(trainbegin_file):
        print(f"ERROR: Required seed dataset '{trainbegin_file}' not found!")
        sys.exit(1)

    # Prefer venv python
    py_exec = sys.executable
    candidates = [
        os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".venv", "bin", "python")),
        os.path.abspath(os.path.join(script_dir, "..", "..", ".venv", "bin", "python")),
    ]
    for c in candidates:
        if os.path.exists(c):
            py_exec = c
            print(f"Using virtual environment python: {py_exec}")
            break

    deepspeed_bin = os.path.join(os.path.dirname(py_exec), "deepspeed")
    if os.path.exists(deepspeed_bin) and args.use_deepspeed:
        launch_cmd = [
            deepspeed_bin,
            "--num_gpus=1",
            f"--master_port={args.port}"
        ]
    else:
        launch_cmd = [py_exec]

    cmd = launch_cmd + [
        steer_trainer,
        "--dataset", args.dataset,
        "--model_name_or_path", args.model_path,
        "--data_dir", data_dir,
        "--output_dir", out_dir,
        "--max_length", str(args.max_length),
        "--batch_size", str(args.batch_size),
        "--deepspeed_config", ds_config,
        "--gradient_accumulation_steps", str(args.accum_steps),
        "--eval_batch_size", str(args.eval_batch),
        "--num_train_epochs", str(args.epochs),
        "--data_percent", str(args.data_percent),
        "--eval_percent", str(args.eval_percent),
        "--save_steps", "100000",
        "--lora", "True",
        "--learning_rate", str(args.lr),
        "--do_train", "True",
        "--do_eval", "True",
        "--infer_per_epoch", str(args.infer_per_epoch),
        "--infer_batch_size", str(args.infer_batch),
        "--max_new_tokens", str(args.max_new_tokens),
        "--gradient_checkpointing"
    ]

    if args.checkpoint_dir:
        cmd += ["--checkpoint_dir", os.path.abspath(args.checkpoint_dir)]

    print("\nExecuting command:")
    print(" ".join(cmd))
    print("-" * 70)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{base_dir}:{base_dir}/src:" + env.get("PYTHONPATH", "")
    # DeepSpeed FusedAdam JIT-loads a CUDA op and needs the venv's `ninja` on PATH (even when the venv isn't activated)
    env["PATH"] = os.path.dirname(py_exec) + os.pathsep + env.get("PATH", "")

    proc = subprocess.run(cmd, cwd=base_dir, env=env)
    if proc.returncode == 0:
        print("\n" + "=" * 70)
        print("STEeR training finished successfully!")
        print(f"Expanded memory & checkpoints saved to: {out_dir}")
        print("=" * 70)
    else:
        print(f"\nExecution exited with code: {proc.returncode}")
        sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
