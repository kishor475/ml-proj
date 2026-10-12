# =============================================================================
# steer_trainer.py -- Self-paced / "memory"-based ERC trainer (STEeR)
#
# Same QLoRA + DeepSpeed setup as erc_trainer.py, but training data is curated
# between epochs (a self-training / curriculum loop):
#   * memory.json      = examples used for training; starts from trainbegin.json
#   * train4infer.json = training examples not yet 'learned'
# From epoch 1 onward, after each training epoch the model predicts every
# example in train4infer.json. Correctly predicted ones (with the model's own
# answer containing the rationale as target) are appended to memory.json;
# the rest stay in train4infer.json. Then the test set is evaluated.
# =============================================================================

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
    LlamaConfig,
    LlamaTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    StoppingCriteriaList
)
import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_of_script = os.path.dirname(current_script_dir)
sys.path.insert(0, parent_dir_of_script)

import pdb
import torch.nn.functional as F
import torch
import torch.nn as nn
import deepspeed
from dataclasses import dataclass, asdict
import pandas as pd
import json
import logging
import math
import random
import re
import warnings
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, DistributedSampler
import numpy as np
from data_utils.data_utils import *
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.optim import AdamW, Adam
from typing import List, Dict
from peft import LoraConfig, get_peft_model
import argparse
from datetime import datetime
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
import shutil
import pickle
import torch.distributed as dist
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import time

# ------------- 2. Multi-GPU / Memory helpers ---------------------------------
def gather_data_from_all_ranks(local_data, world_size, local_rank):
    if world_size == 1 or local_rank == -1:
        return local_data
    
    local_data_bytes = pickle.dumps(local_data)
    local_data_tensor = torch.ByteTensor(list(local_data_bytes)).cuda()
    local_size = torch.tensor([len(local_data_bytes)], dtype=torch.long).cuda()
    size_list = [torch.tensor([0], dtype=torch.long).cuda() for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = max(size.item() for size in size_list)
    
    if len(local_data_tensor) < max_size:
        padding = torch.zeros(max_size - len(local_data_tensor), dtype=torch.uint8).cuda()
        local_data_tensor = torch.cat([local_data_tensor, padding])
    
    gathered_tensors = [torch.zeros(max_size, dtype=torch.uint8).cuda() for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_data_tensor)
    
    if local_rank == 0:
        gathered_data = []
        for i, (tensor, size) in enumerate(zip(gathered_tensors, size_list)):
            valid_tensor = tensor[:size.item()]
            rank_data = pickle.loads(bytes(valid_tensor.cpu().numpy()))
            gathered_data.extend(rank_data)
        return gathered_data
    return None

def save_memory_data_distributed(memory, others, memory_file, train4infer_file, world_size, local_rank):
    if world_size == 1 or local_rank in (-1, 0):
        if memory:
            with open(memory_file, 'a', encoding='utf-8') as file:
                for example in memory:
                    file.write(json.dumps(example, ensure_ascii=False) + "\n")
            print(f"Added {len(memory)} new examples to training set ({memory_file})")
        
        if others is not None:
            with open(train4infer_file, 'w', encoding='utf-8') as file:
                for example in others:
                    file.write(json.dumps(example, ensure_ascii=False) + "\n")
            print(f"Updated unlearned inference pool with {len(others)} examples ({train4infer_file})")

# ------------------- 3. Label + metric helpers -------------------------------
def find_emotional_label(text, dataset):
    """Final label of a rationale: last <Label>, else 'label ... : Label' / 'is Label'. None if absent."""
    if not text:
        return None
    names = [k for k in get_labels_attr(dataset)[0] if k == k.lower()]
    alt = "|".join(re.escape(n) for n in names)
    for pat in (r"<\s*(" + alt + r")\s*>",
                r"label[^\n<>:]{0,20}?(?:is|:)\s*\**<?(" + alt + r")\b",
                r"\bis\s+(" + alt + r")\b"):
        hits = re.findall(pat, text, re.I)
        if hits:
            return hits[-1].lower()
    return None

_LABEL_SENT = re.compile(r"\s*(?:The emotional label of|Emotional Label|Emotion Label|The emotion label)", re.I)

def normalize_rationale(ans, prompt, label_name):
    """'[Rationale]\n<body> The emotional label of <spk: "utt"> is <Label>.' - the seed format.
    Returns None if no usable body remains."""
    body = re.sub(r"^\s*(\[Rationale\]|Rationale\s*:)\s*", "", ans, flags=re.I)
    m = _LABEL_SENT.search(body)
    if m:
        body = body[:m.start()]
    body = " ".join(body.split())
    sp = re.search(r'Please analyze the emotional state of <(.*?): "(.*)"> by first', prompt, re.S)
    if len(body) < 40 or sp is None:
        return None
    return f'[Rationale]\n{body} The emotional label of <{sp.group(1)}: "{sp.group(2)}"> is <{label_name}>.'

def get_labels_attr(dataset):
    d = dataset.lower()
    if 'iemocap' in d:
        label_list = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
        label_str = "'happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'"
    elif 'emorynlp' in d:
        label_list = ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared']
        label_str = "'Joyful','Mad','Peaceful', 'Neutral','Sad','Powerful','Scared'"
    else:
        label_list = ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry']
        label_str = "'neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'"

    emotional_label_dict = {text_label: num_label for num_label, text_label in enumerate(label_list)}
    emotional_label_dict.update({text_label.lower(): num_label for num_label, text_label in enumerate(label_list)})
    return emotional_label_dict, label_str

def report_score(dataset, golds, preds, mode='test'):
    d = dataset.lower()
    if 'iemocap' in d:
        target_names = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
        digits = 6
    elif 'emorynlp' in d:
        target_names = ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared']
        digits = 7
    else:
        target_names = ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry']
        digits = 7

    res = {
        'Acc_SA': round(accuracy_score(golds, preds) * 100, 3),
        'F1_SA': round(f1_score(golds, preds, average='weighted', zero_division=0) * 100, 3),
        'mode': mode
    }
    res_matrix = metrics.classification_report(
        golds, preds,
        labels=list(range(len(target_names))),
        target_names=target_names,
        digits=digits,
        zero_division=0
    )
    return res, res_matrix     

def match_text(text, word_set_):
    if text is None:
        return []
    len_text = len(text)
    s_idx = 0
    match_res = []
    word_set_lower = {w.lower(): w for w in word_set_}
    while s_idx < len_text:
        cache = []
        span_length = 1
        while span_length < 15 and s_idx + span_length <= len_text:
            span = text[s_idx: s_idx + span_length].strip('<>').lower()
            if span in word_set_lower:
                cache.append(word_set_lower[span])
            span_length += 1
        if len(cache) > 0:
            match_res.append(cache[-1])
            s_idx += len(cache[-1])
        else:
            s_idx += 1
    return match_res

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1].lower() == s2[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]

def optimize_output(output, label_set):
    min_distance = float('inf')
    optimized_output = list(label_set)[0]
    clean_out = output.strip('<>').strip()
    for label in label_set:
        distance = edit_distance(clean_out, label)
        if distance < min_distance:
            min_distance = distance
            optimized_output = label
    return optimized_output

# ----------------- 4. CLI Arguments ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
world_size = int(os.getenv("WORLD_SIZE", '1'))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--gradient_checkpointing", action='store_true')
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--mode", type=str, default="sft")
parser.add_argument("--deepspeed_config", type=str, default="./data_utils/deepspeed_config.json")
parser.add_argument("--num_train_epochs", default=5, type=int)
parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
parser.add_argument("--warmup_ratio", default=0.1, type=float)
parser.add_argument('--local_rank', default=-1)
parser.add_argument("--warmup_steps", default=None, type=int)
parser.add_argument("--learning_rate", default=2e-4, type=float)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--max_length", default=1400, type=int)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--save_steps', default=100000, type=int)
parser.add_argument("--zero_shot", default=False)
parser.add_argument("--lora", default=True)
parser.add_argument("--lora_dim", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_module_name", type=str, default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--eval_batch_size", default=2, type=int)
parser.add_argument("--top_k", default=None, type=int)
parser.add_argument("--num_beams", default=1, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--clip_norm", type=float, default=1.0)
parser.add_argument("--temp", type=float, default=None)
parser.add_argument("--do_train", default=True)
parser.add_argument("--do_eval", default=True)
parser.add_argument("--offload_optimizer", action='store_true')
parser.add_argument("--eval_percent", default=0.02, type=float)
parser.add_argument("--data_percent", default=1.0, type=float)
parser.add_argument("--max_new_tokens", default=200, type=int, help="rationale generation budget")
parser.add_argument("--infer_per_epoch", default=1500, type=int, help="pool examples scored per STEeR epoch")
parser.add_argument("--infer_batch_size", default=16, type=int)

args = parser.parse_args()
do_sample = args.top_k is not None or args.top_p is not None or args.num_beams > 1 or args.temp is not None

if isinstance(args.do_train, str):
    args.do_train = (args.do_train.lower() == 'true')
if isinstance(args.do_eval, str):
    args.do_eval = (args.do_eval.lower() == 'true')
if isinstance(args.lora, str):
    args.lora = (args.lora.lower() == 'true')

model_args = {
    "dataset": args.dataset,
    "model_name_or_path": args.model_name_or_path,
    "checkpoint_dir": args.checkpoint_dir,
    "data_dir": args.data_dir,
    "max_seq_length": args.max_seq_length,
    "batch_size": args.batch_size,
    "eval_batch_size": args.eval_batch_size,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "learning_rate": args.learning_rate,
    "num_train_epochs": args.num_train_epochs,
    "save_steps": args.save_steps,
    "output_dir": args.output_dir,
    "max_length": args.max_length,
    "warmup_ratio": args.warmup_ratio,
    "warmup_steps": args.warmup_steps,
    "weight_decay": args.weight_decay,
    "lora": args.lora,
    "lora_dim": args.lora_dim,
    "lora_dropout": args.lora_dropout,
    "lora_alpha": args.lora_alpha,
    "lora_module_name": args.lora_module_name,
    "num_beams": args.num_beams,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "do_sample": do_sample,
    "seed": args.seed,
    "do_train": args.do_train,
    "do_eval": args.do_eval,
    "offload_optimizer": args.offload_optimizer,
    "deepspeed_config": args.deepspeed_config,
    "zero_shot": args.zero_shot,
    "mode": args.mode,
    "gradient_checkpointing": args.gradient_checkpointing,
    "data_percent": args.data_percent,
    "max_new_tokens": args.max_new_tokens,
    "infer_per_epoch": args.infer_per_epoch,
    "infer_batch_size": args.infer_batch_size,
    "eval_percent": args.eval_percent,
    "local_rank": args.local_rank
}
args_obj = ModelArgs()
args_obj.update(model_args)
args = args_obj

os.makedirs(args.output_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=args.lora_dim,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=args.lora_module_name.split(","),
    bias='none',
)

with open(args.deepspeed_config, 'r', encoding='utf-8') as f:
    deepspeed_config = json.load(f)
deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
deepspeed_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * max(1, world_size)

def getOptimizerGroup(model):
    no_decay = ["bias", "LayerNorm.weight"]
    return [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

def _get_input_dict(batch):
    return {
        "input_ids": batch["input_ids"].to(device),
        "labels": batch["labels"].to(device),
        "attention_mask": batch["attention_mask"].to(device)
    }

# ----------------- 5. Model + Tokenizer Loading ------------------------------
config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    device_map={"": int(args.local_rank)} if int(args.local_rank) != -1 else "auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<|pad|>"
tokenizer.padding_side = "left"

if args.lora:
    from peft import prepare_model_for_kbit_training, PeftModel
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.checkpoint_dir is not None and not os.path.exists(os.path.join(args.checkpoint_dir, "adapter_config.json")):
        raise FileNotFoundError(f"--checkpoint_dir has no LoRA adapter (adapter_config.json): {args.checkpoint_dir}")
    if args.checkpoint_dir is not None:
        print(f"***** Loading existing LoRA adapter from {args.checkpoint_dir} *****")
        model = PeftModel.from_pretrained(model, args.checkpoint_dir, is_trainable=args.do_train)
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# ----------------- 6. Data Files for STEeR ------------------------------------
dev_file = os.path.join(args.data_dir, "test.json")
memory_file = os.path.join(args.output_dir, "memory.json")
trainbegin_file = os.path.join(args.data_dir, "trainbegin.json")
train4infer_file = os.path.join(args.output_dir, "train4infer.json")

# memory.json = rationale training set (starts from the seed rationales)
if not os.path.exists(memory_file):
    if not os.path.exists(trainbegin_file):
        raise FileNotFoundError(f"Missing required seed dataset: {trainbegin_file} (run generate_seed_rationales.py)")
    shutil.copyfile(trainbegin_file, memory_file)
    print(f"Seeded memory from {trainbegin_file}")
# train4infer.json = pool of training examples without an accepted rationale yet
if not os.path.exists(train4infer_file):
    shutil.copyfile(os.path.join(args.data_dir, "train4infer.json"), train4infer_file)
    print(f"Initialized inference pool ({train4infer_file})")

df_dev = None
dev_dataset = None
if args.do_eval and os.path.exists(dev_file):
    df_dev = read_data(dev_file, percent=args.eval_percent, random_seed=args.seed)

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

@torch.no_grad()
def generate_answers(prompts, batch_size):
    """Greedy rationale+label generation; returns only the new tokens, in input order."""
    model.eval()
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    answers = [None] * len(prompts)
    for k in tqdm(range(0, len(order), batch_size), desc="Generating"):
        idx = order[k:k + batch_size]
        enc = tokenizer([prompts[i] for i in idx], max_length=args.max_length - args.max_new_tokens,
                        truncation=True, padding=True, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        for i, o in zip(idx, out):
            answers[i] = tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return answers

# ----------------- 7. Main Loop ----------------------------------------------
if __name__ == "__main__":
    emotional_label_dict, emotional_label_str = get_labels_attr(dataset=args.dataset)

    # One DeepSpeed engine for the whole run (memory.json grows, so batches are built per epoch).
    optimizer_class = DeepSpeedCPUAdam if deepspeed_config["zero_optimization"]["offload_optimizer"]["device"] == "cpu" else FusedAdam
    optimizer = optimizer_class(getOptimizerGroup(model=model), lr=args.learning_rate, betas=[0.9, 0.95])
    # dataset size changes every epoch, so use warmup + constant LR instead of a fixed-horizon decay
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps or 10)
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, config=deepspeed_config)
    model = model_engine
    train_collator = Seq2SeqCollator(args, tokenizer, mode="train")

    best_f1_score = 0
    for epoch in range(args.num_train_epochs):
        print(f"\n{'='*70}\nSTEeR EPOCH {epoch} / {args.num_train_epochs}\n{'='*70}")

        # (1) fine-tune on the current rationale set  (M_n <- train(M_{n-1}, D_train))
        df_train = read_data(memory_file, percent=args.data_percent, random_seed=args.seed + epoch)
        loader = DataLoader(Seq2SeqDataset(args, df_train, mode='train'), batch_size=args.batch_size,
                            shuffle=True, collate_fn=train_collator)
        model.train()
        it = tqdm(loader, desc=f"Training on memory ({len(df_train)} rationales)")
        for batch in it:
            loss = model(**_get_input_dict(batch)).loss
            model.backward(loss)
            model.step()
            it.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        (model.module if hasattr(model, 'module') else model).save_pretrained(os.path.join(args.output_dir, f"lora_epoch_{epoch}"))

        # (2) generate rationales for part of the pool; keep the ones whose final label is correct
        pool = read_jsonl(train4infer_file)
        random.Random(args.seed + epoch).shuffle(pool)
        batch_pool, rest = pool[:args.infer_per_epoch], pool[args.infer_per_epoch:]
        if batch_pool:
            answers = generate_answers([r["input"] for r in batch_pool], args.infer_batch_size)
            accepted, failed = [], []
            for r, ans in zip(batch_pool, answers):
                gold = r["target"].strip().strip('<>').lower()
                norm = normalize_rationale(ans, r["input"], r["target"].strip().strip('<>')) \
                    if find_emotional_label(ans, args.dataset) == gold else None
                if norm is not None:
                    accepted.append({"input": r["input"], "target": norm})
                else:
                    failed.append(r)
            write_jsonl(memory_file, accepted, mode="a")
            write_jsonl(train4infer_file, rest + failed)
            print(f"Accepted {len(accepted)}/{len(batch_pool)} rationales "
                  f"({100*len(accepted)/len(batch_pool):.1f}%); memory +{len(accepted)}, pool now {len(rest) + len(failed)}")

        # (3) monitor on the test set (rationale mode)
        if df_dev is not None:
            targets = list(df_dev["output"])
            answers = generate_answers(list(df_dev["input"]), args.infer_batch_size)
            golds, preds = [], []
            for tgt, ans in zip(targets, answers):
                golds.append(emotional_label_dict[tgt.strip().strip('<>')])
                lab = find_emotional_label(ans, args.dataset)
                preds.append(emotional_label_dict[lab] if lab else -1)
            score, res_matrix = report_score(dataset=args.dataset, golds=golds, preds=preds)
            print(f"Epoch {epoch} Test Weighted-F1: {score['F1_SA']}% | Accuracy: {score['Acc_SA']}%")
            with open(os.path.join(args.output_dir, f"preds_for_eval_{epoch}_{score['F1_SA']}.text"), 'w', encoding='utf-8') as f:
                f.write(json.dumps(score) + f"\n{res_matrix}\n\n")
                f.write(json.dumps([{"input": i, "output": a, "target": t} for i, a, t in zip(df_dev["input"], answers, targets)],
                                   indent=2, ensure_ascii=False))
            if score['F1_SA'] > best_f1_score:
                best_f1_score = score['F1_SA']
                (model.module if hasattr(model, 'module') else model).save_pretrained(os.path.join(args.output_dir, "best_checkpoint"))

    (model.module if hasattr(model, 'module') else model).save_pretrained(os.path.join(args.output_dir, "final_checkpoint"))
    print(f"\nSTEeR Training Complete! Best Weighted-F1: {best_f1_score}%  | memory: {len(read_jsonl(memory_file))} rationales")
