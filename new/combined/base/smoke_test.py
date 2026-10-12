#!/usr/bin/env python3
"""
Quick smoke test for Combined (Mixed Strategy B) and STEeR
Uses data_percent=0.002, eval_percent=0.02, 1 epoch on EmoryNLP with Mistral-7B-Instruct-v0.3
"""
import sys
import subprocess
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Smoke test for combined / STEeR")
    parser.add_argument("--mode", choices=["combined", "steer"], default="combined",
                        help="Which mode to smoke test (default: combined)")
    parser.add_argument("--dataset", default="emorynlp", choices=["emorynlp", "iemocap", "meld"])
    args, extra_args = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.mode == "steer":
        runner = os.path.join(script_dir, "run_local_steer.py")
    else:
        runner = os.path.join(script_dir, "run_local_combined.py")

    cmd = [sys.executable, runner, "--smoke_test", "--dataset", args.dataset] + extra_args
    print(f"Running {args.mode} smoke test...")
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
