"""
==============================================================================
OUR METHOD (baseline + additions) — Kaggle Notebook
==============================================================================
Same workflow as the true baseline (ONE joint run of src/erc_trainer.py on
train.json = ERC + Speaker Identification + Role-Play, persona + history),
PLUS our additions:
  ➕ Acoustic clues   ("### Acoustic Clues:"   from SpeechCueLLM impressions)
  ➕ Relational clues ("### Relational Clues:" speaker-relationship text)
  ➕ Chain-of-Emotion ERC target:
        "Emotion Flow: a -> b -> c. \nCurrent Emotion: X"

Still NOT included: STEeR, scene clues.

Data is already generated with these additions (no target stripping here!).
Hardware adaptations: QLoRA 4-bit NF4, FP16 DeepSpeed.
==============================================================================
"""

# ─── CELL 1: Install ──────────────────────────────────────────────────────────
# !pip install -q -U bitsandbytes transformers peft accelerate datasets
# !pip install -q -U deepspeed
# !pip install -q -U torchao
# Then RESTART the kernel once so the new versions load.


# ─── CELL 2: HuggingFace login ────────────────────────────────────────────────
# from kaggle_secrets import UserSecretsClient
# from huggingface_hub import login
# login(token=UserSecretsClient().get_secret("HF_TOKEN"))


# ─── CELL 3: Copy project to a writable dir (adjust dataset name) ─────────────
# !cp -r /kaggle/input/<your-dataset-name>/Truebaseline /kaggle/working/Truebaseline


# ─── CELL 4: Configuration ───────────────────────────────────────────────────
import os, json

DATASET    = "meld"          # "meld" | "emorynlp" | "iemocap"
MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.1"
BASE_DIR   = "/kaggle/working/Truebaseline"
DATA_DIR   = f"{BASE_DIR}/data/{DATASET}"
OUT_DIR    = f"{BASE_DIR}/outputs/CoE_Ours_joint/{DATASET}"
DS_CONFIG  = f"{BASE_DIR}/config/deepspeed_config.json"

NUM_GPUS     = 2
MASTER_PORT  = 29500
EPOCHS       = 3        # earlier run used 3
DATA_PERCENT = 0.5      # earlier run used 0.5 (set ~0.05 for a quick smoke test)
EVAL_PERCENT = 0.02     # fraction of test.json scored each epoch (earlier run: 0.02)
LR           = "2e-4"
BATCH_SIZE   = 8
ACCUM_STEPS  = 4
EVAL_BATCH   = 4
MAX_LENGTH   = 1400

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Dataset: {DATASET} | Model: {MODEL_PATH} | Epochs: {EPOCHS} | Data: {DATA_PERCENT}")


# ─── CELL 5: Patch erc_trainer.py (hardware / correctness only) ──────────────
"""
  1. QLoRA 4-bit loading                         (hardware)
  2. prepare_model_for_kbit_training before LoRA (hardware)
  3. Gradient-checkpointing guard for QLoRA      (correctness)
  4. Skip ZeRO checkpoint load for LoRA          (correctness)
  5. Drop empty optimizer param groups           (correctness)
  6. classification_report with missing labels   (correctness)
  7. Save LoRA adapter every epoch               (so STEeR etc. can resume later)
  8. Score only EVAL_PERCENT of test.json/epoch  (time budget)
"""

def swap(code, old, new, name):
    if new in code:
        print(f"  [{name}] already applied")
        return code
    if old not in code:
        print(f"  [{name}] NO MATCH — check manually")
        return code
    print(f"  [{name}] applied")
    return code.replace(old, new, 1)

erc_path = f"{BASE_DIR}/src/erc_trainer.py"
with open(erc_path, "r") as f:
    code = f.read()
print("Patching erc_trainer.py...")

# (QLoRA loading is already built into src/erc_trainer.py)

code = swap(code,
    'if args.lora:\n    model = get_peft_model(model, lora_config)',
    '''if args.lora:
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()''',
    "kbit training")

# (gradient-checkpointing guard is already built into src/erc_trainer.py)

code = swap(code,
    'if args.checkpoint_dir is not None:\n    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint',
    'if args.checkpoint_dir is not None and not args.lora:\n    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint',
    "ckpt guard")

code = swap(code,
    "    return optimizer_grouped_parameters",
    "    return [g for g in optimizer_grouped_parameters if len(g['params']) > 0]",
    "empty param groups")

code = swap(code,
    "metrics.classification_report(golds, preds, target_names=target_names, digits=digits)",
    "metrics.classification_report(golds, preds, labels=list(range(len(target_names))), target_names=target_names, digits=digits, zero_division=0)",
    "report fix")

code = swap(code,
    '''            model.eval()
            targets = list(df_dev["output"])''',
    '''            if int(args.local_rank) in (-1, 0):
                model.module.save_pretrained(os.path.join(args.output_dir, f'lora_epoch_{epoch}'))
            model.eval()
            targets = list(df_dev["output"])''',
    "per-epoch adapter save")

code = swap(code,
    "df_dev = read_data(dev_file, percent=1.0",
    f"df_dev = read_data(dev_file, percent={EVAL_PERCENT}",
    "eval subset")

with open(erc_path, "w") as f:
    f.write(code)
print("erc_trainer.py done.\n")


# ─── CELL 6: DeepSpeed config ────────────────────────────────────────────────
with open(DS_CONFIG, "r") as f:
    ds = json.load(f)
ds["zero_optimization"]["stage"] = 2
ds["zero_optimization"]["offload_optimizer"] = {"device": "none"}
ds["fp16"] = {"enabled": True}
ds["bf16"] = {"enabled": False}
for k in ["bfloat16", "optimizer", "scheduler", "train_micro_batch_size_per_gpu", "train_batch_size"]:
    ds.pop(k, None)
with open(DS_CONFIG, "w") as f:
    json.dump(ds, f, indent=4)
print("DeepSpeed config ready.\n")


# ─── CELL 7: Verify the additions are present (DO NOT strip anything) ────────
n = {"flow": 0, "acoustic": 0, "relational": 0, "total": 0}
with open(f"{DATA_DIR}/train.json", encoding="utf-8") as f:
    for l in f:
        r = json.loads(l)
        n["total"] += 1
        n["flow"] += "Emotion Flow:" in r["target"]
        n["acoustic"] += "### Acoustic Clues:" in r["input"]
        n["relational"] += "### Relational Clues:" in r["input"]
print(n)
assert n["flow"] > 0 and n["acoustic"] > 0 and n["relational"] > 0, "additions missing from data!"
print()


# ─── CELL 8: Sanity check one record of each kind ────────────────────────────
with open(f"{DATA_DIR}/train.json", encoding="utf-8") as f:
    recs = [json.loads(next(f)) for _ in range(3)]
for r in recs:
    print("TARGET:", repr(r["target"][:80]))
    print("INPUT :", r["input"][:200].replace("\n", " | "), "...\n")
# Expect: ERC target = "Emotion Flow: ... Current Emotion: X", and ERC/speaker/
# role-play INPUTs contain "### Acoustic Clues:" and "### Relational Clues:".


# ─── CELL 9: Train (single joint stage) ──────────────────────────────────────
cmd = (
    f"cd {BASE_DIR} && deepspeed --num_gpus={NUM_GPUS} --master_port={MASTER_PORT} "
    f"src/erc_trainer.py "
    f'--dataset "{DATASET}" '
    f'--model_name_or_path "{MODEL_PATH}" '
    f'--data_dir "{DATA_DIR}" '
    f'--output_dir "{OUT_DIR}" '
    f"--max_length {MAX_LENGTH} "
    f"--batch_size {BATCH_SIZE} "
    f'--deepspeed_config "{DS_CONFIG}" '
    f"--gradient_accumulation_steps {ACCUM_STEPS} "
    f"--eval_batch_size {EVAL_BATCH} "
    f"--num_train_epochs {EPOCHS} "
    f"--data_percent {DATA_PERCENT} "
    f"--save_steps 100000 "
    f"--lora True "
    f"--learning_rate {LR} "
    f"--do_train True "
    f"--do_eval True "
    f"--gradient_checkpointing"
)
print(cmd)
# Run it in its own notebook cell so output streams:
# !{cmd}

print(f"\nScores: {OUT_DIR}/preds_for_eval_*.text   |   Adapters: {OUT_DIR}/lora_epoch_*")
