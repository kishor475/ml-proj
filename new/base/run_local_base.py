#!/usr/bin/env python3
"""
==============================================================================
LOCAL CoE BASELINE RUNNER (RTX 3060 12GB VRAM + Mistral-7B-Instruct-v0.3)
==============================================================================
Runs the single joint True CoE Baseline stage locally without Kaggle dependencies.
Paper: 2025.acl-long.1148 "CoE: A Clue of Emotion Framework for ERC"

Tailored for single NVIDIA GeForce RTX 3060 (12 GB VRAM):
  - QLoRA 4-bit NF4
  - Gradient Checkpointing
  - Micro batch size 2, gradient accumulation 8 (effective batch size 16)
  - DeepSpeed ZeRO-2 fp16 (1 GPU)
==============================================================================
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run CoE Baseline locally on desktop GPU")
    parser.add_argument("--dataset", type=str, default="iemocap", choices=["iemocap", "meld", "emorynlp"],
                        help="Dataset to train on (default: iemocap)")
    parser.add_argument("--model_path", type=str, default="/home/user2/Mistral-7B-Instruct-v0.3",
                        help="Path to local pretrained Mistral model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--data_percent", type=float, default=0.5,
                        help="Fraction of training data to use (use 0.002 for quick smoke test)")
    parser.add_argument("--eval_percent", type=float, default=0.02,
                        help="Fraction of test data to score each epoch")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-GPU train batch size (2 fits 12GB VRAM safely)")
    parser.add_argument("--accum_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_batch", type=int, default=2,
                        help="Eval batch size")
    parser.add_argument("--max_length", type=int, default=1400,
                        help="Max sequence length")
    parser.add_argument("--lr", type=str, default="2e-4",
                        help="Learning rate")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Shortcut for smoke test (epochs=1, data_percent=0.002, eval_percent=0.02)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to saved LoRA adapter checkpoint (e.g. from Kaggle)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation using checkpoint without training")
    parser.add_argument("--port", type=int, default=29500,
                        help="DeepSpeed master port")
    parser.add_argument("--use_deepspeed", action="store_true", default=True,
                        help="Use DeepSpeed launcher (default: True)")
    
    args = parser.parse_args()

    if args.smoke_test:
        print("\n>>> SMOKE TEST MODE ACTIVATED <<<")
        args.epochs = 1
        args.data_percent = 0.002
        args.eval_percent = 0.02

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "Truebaseline")
    data_dir = os.path.join(base_dir, "data", args.dataset)
    out_dir = os.path.join(base_dir, "outputs", "CoE_True_Baseline_joint", args.dataset)
    ds_config = os.path.join(base_dir, "config", "deepspeed_config.json")
    erc_trainer = os.path.join(base_dir, "src", "erc_trainer.py")

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("LOCAL CoE BASELINE RUNNER")
    print(f"Dataset      : {args.dataset}")
    print(f"Model Path   : {args.model_path}")
    print(f"Data Dir     : {data_dir}")
    print(f"Output Dir   : {out_dir}")
    print(f"Epochs       : {args.epochs}")
    print(f"Data Percent : {args.data_percent}")
    print(f"Eval Percent : {args.eval_percent}")
    print(f"Batch Size   : {args.batch_size} (accum: {args.accum_steps})")
    print("=" * 70)

    # Sanity check model path
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path '{args.model_path}' does not exist!")
        sys.exit(1)

    # Sanity check data dir
    train_file = os.path.join(data_dir, "train.json")
    if not os.path.exists(train_file):
        print(f"ERROR: Data file '{train_file}' not found!")
        sys.exit(1)

    # Detect python executable (prefer active venv)
    py_exec = sys.executable
    venv_py = os.path.abspath(os.path.join(script_dir, "..", "..", ".venv", "bin", "python"))
    if os.path.exists(venv_py):
        py_exec = venv_py
        print(f"Using virtual environment python: {py_exec}")

    # Build command
    # Single GPU RTX 3060: run deepspeed launcher with --num_gpus 1
    deepspeed_bin = os.path.join(os.path.dirname(py_exec), "deepspeed")
    if os.path.exists(deepspeed_bin) and args.use_deepspeed:
        launch_cmd = [
            deepspeed_bin,
            "--num_gpus=1",
            f"--master_port={args.port}"
        ]
    else:
        # Fallback to direct python execution with deepspeed config
        launch_cmd = [py_exec]

    do_train_val = "False" if args.eval_only else "True"
    do_eval_val = "True"

    cmd = launch_cmd + [
        erc_trainer,
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
        "--do_train", do_train_val,
        "--do_eval", do_eval_val,
        "--gradient_checkpointing"
    ]

    if args.checkpoint_dir:
        cmd += ["--checkpoint_dir", args.checkpoint_dir]

    print("\nExecuting command:")
    print(" ".join(cmd))
    print("\n" + "-" * 70)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{base_dir}:{base_dir}/src:" + env.get("PYTHONPATH", "")
    
    proc = subprocess.run(cmd, cwd=base_dir, env=env)
    if proc.returncode == 0:
        print("\n" + "=" * 70)
        print("Training finished successfully!")
        print(f"Predictions & scores saved to: {out_dir}")
        print("=" * 70)
    else:
        print(f"\nExecution exited with error code: {proc.returncode}")
        sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
