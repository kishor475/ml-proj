#!/bin/bash
cd /home/user2/ml_proj/new/combined/base
PY=/home/user2/ml_proj/.venv/bin/python
$PY -I run_local_combined.py --smoke_test --data_subdir emorynlp_coe_aux --run_name smoke_s1 --dataset emorynlp --no_eval --port 29541 > smoke_s1.log 2>&1 || { echo S1FAIL >> smoke_s1.log; exit 1; }
S1=Truebaseline/outputs/smoke_s1/emorynlp_coe_aux/final_checkpoint
$PY -I run_local_combined.py --smoke_test --data_subdir emorynlp_coe_erc --run_name smoke_s2 --dataset emorynlp --checkpoint_dir $S1 --port 29542 > smoke_s2.log 2>&1 || echo S2FAIL >> smoke_s2.log
echo CHAINDONE > smoke_chain.done
