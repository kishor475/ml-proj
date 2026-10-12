#!/usr/bin/env python3
"""
Quick smoke test for local CoE Ours (Acoustic + Relational + Chain-of-Emotion)
Uses data_percent=0.002, eval_percent=0.02, 1 epoch on IEMOCAP with Mistral-7B-Instruct-v0.3
"""
import sys
import subprocess
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
runner = os.path.join(script_dir, "run_local_ours.py")

cmd = [sys.executable, runner, "--smoke_test", "--dataset", "iemocap"] + sys.argv[1:]
print("Running smoke test for Ours (Acoustic + Relational + Chain)...")
proc = subprocess.run(cmd)
sys.exit(proc.returncode)
