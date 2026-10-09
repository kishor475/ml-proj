import os
import re
import shutil
import py_compile

src = "src/multimodal_trainer.py"
backup = "src/multimodal_trainer.original.py"

if not os.path.exists(backup):
    shutil.copy(src, backup)

with open(backup, "r", encoding="utf-8") as f:
    content = f.read()

# Safest way to replace
def rep(old, new):
    global content
    assert old in content, f"anchor not found: {old[:60]}"
    content = content.replace(old, new)


# 2. Fix the Memory crash
rep("df_train = read_data(memory_file, percent=args.data_percent, random_seed=args.seed)", 
    "df_train = read_data(train_file, percent=args.data_percent, random_seed=args.seed)")

# 3. FIX THE GPU VRAM LIMIT WITH 4-BIT QLoRA & DEVICE MAP
old_load = "model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True).half()"
new_load = """from transformers import BitsAndBytesConfig
        import torch
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map={"": int(os.getenv("LOCAL_RANK", "0"))},
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )"""
rep(old_load, new_load)

# 4. FIX TRUNCATION SIDE BUG
rep("if tokenizer.pad_token is None:",
    "tokenizer.truncation_side = 'left'\nif tokenizer.pad_token is None:")

# 5. FIX THE BATCH SIZE MATH BUG
bad_batch_code = """deepspeed_config["train_batch_size"] = args.batch_size
deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps"""
good_batch_code = """deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
deepspeed_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * world_size"""
rep(bad_batch_code, good_batch_code)

# 6. FIX THE LR SCHEDULER BUG
bad_lr_code = "t_total = math.ceil(len(train_dataset) / args.batch_size) * args.num_train_epochs"
good_lr_code = "t_total = math.ceil(len(train_dataset) / (args.batch_size * args.gradient_accumulation_steps * world_size)) * args.num_train_epochs"
rep(bad_lr_code, good_lr_code)

# 7. FIX THE OPTIMIZER CRASH
rep("    return optimizer_grouped_parameters\n", 
    "    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g['params']) > 0]\n    return optimizer_grouped_parameters\n")

# 8. Remove self-training loop but KEEP Eval (Evaluate ONLY on last 2 epochs to save time)
pattern = r'if epoch>=1:\n.*?df_train4infer = read_data.*?targets = list\(df_dev\["output"\]\)'
assert len(re.findall(pattern, content, flags=re.DOTALL)) == 1, "step 8 anchor not found"
content = re.sub(
    pattern,
    'if epoch >= args.num_train_epochs - 2:\n                targets = list(df_dev["output"])',
    content, count=1, flags=re.DOTALL,
)

# 9. PREVENT DOUBLE CLIPPING BUG
rep("                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)\n", "")

# 10. Multi-GPU Eval Bug
rep("eval_sampler = DistributedSampler(dev_dataset, shuffle=False)", "eval_sampler = SequentialSampler(dev_dataset)")

# 11. SAVE LORA WEIGHTS IMMEDIATELY
rep("            model.eval()", """            if args.local_rank in (-1, 0):
                model.module.save_pretrained(os.path.join(args.output_dir, f'lora_epoch_{epoch}'))
            model.eval()""")

# 12. EmoryNLP Label Capitalization Fix
rep("'emorynlp': ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared'],",
    "'emorynlp': ['joyful', 'mad', 'peaceful', 'neutral', 'sad', 'powerful', 'scared'],")

# 13. BULLETPROOF EVALUATION CRASH FIX
pattern2 = re.compile(r"^( *)golds \+= \[emotional_label_dict\[targets\[index\]\]\]$", re.M)
def fix(m):
    i = m.group(1)
    lines = [
        "target_str = targets[index]",
        "if 'Current Emotion: ' in target_str:",
        "    target_emo = target_str.split('Current Emotion: ')[-1].strip()",
        "else:",
        "    target_emo = target_str.strip()",
        "target_emo = target_emo.lower()",
        "if target_emo not in emotional_label_dict:",
        "    words = target_emo.split()",
        "    target_emo = words[-1].replace('.', '').lower() if words else ''",
        "    if target_emo not in emotional_label_dict:",
        "        print('bad target:', target_str)",
        "        target_emo = list(emotional_label_dict.keys())[0]",
        "golds += [emotional_label_dict[target_emo]]",
        "if 'Current Emotion: ' in answer:",
        "    answer = answer.split('Current Emotion: ')[-1].strip()",
        "answer = answer.lower()",
    ]
    return "\n".join(i + l for l in lines)
content, n = pattern2.subn(fix, content)
assert n == 1, "step 13 did not apply"

# 14. CLASSIFICATION REPORT CRASH FIX
rep("metrics.classification_report(golds, preds, target_names=target_names, digits=digits)",
    "metrics.classification_report(golds, preds, labels=list(range(len(target_names))), target_names=target_names, digits=digits, zero_division=0)")

# 15. WORKER THREAD COUNT BUG
rep("num_workers=8", "num_workers=2")

# Save the permanently patched file safely
with open(src, "w", encoding="utf-8", newline="\n") as f:
    f.write(content)

py_compile.compile(src, cfile="check.pyc", doraise=True)
print("patched and syntax ok")
