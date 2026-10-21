# =============================================================================
# erc_trainer.py -- Emotion Recognition in Conversation (ERC) & Multi-task Trainer
#
# Fine-tunes (4-bit QLoRA + DeepSpeed) or evaluates a decoder-only LLM
# (e.g. Mistral-7B-Instruct) on joint multi-task datasets (ERC + Speaker ID +
# Role-Play + STEeR Emotion Reasoning).
# =============================================================================

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
    LlamaConfig,
    LlamaTokenizer,
    get_linear_schedule_with_warmup,
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
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import torch.distributed as dist

# ----------------- 2. Label & Metric Helpers ---------------------------------
def get_labels_attr(dataset):
    d = dataset.lower()
    if 'emorynlp' in d:
        label_list = ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared']
        label_str = "'Joyful','Mad','Peaceful', 'Neutral','Sad','Powerful','Scared'"
    elif 'iemocap' in d:
        label_list = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
        label_str = "'happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'"
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

    res = {}
    res['Acc_SA'] = accuracy_score(golds, preds)
    res['F1_SA'] = f1_score(golds, preds, average='weighted', zero_division=0)
    res['mode'] = mode
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v * 100, 3)

    res_matrix = metrics.classification_report(
        golds, preds,
        labels=list(range(len(target_names))),
        target_names=target_names,
        digits=digits,
        zero_division=0
    )
    return res, res_matrix     
    
def current_emotion_text(text):
    """Text that holds the answer. With a 'Current Emotion:' marker only the FIRST LINE after the last marker counts (so label-first
    answers 'Current Emotion: X\\nEmotion Flow: ...' are scored on X); without a marker the whole text, as before."""
    if "Current Emotion:" in text:
        return text.split("Current Emotion:")[-1].strip().split("\n")[0].strip()
    return text.strip()


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

# ----------------- 3. Runtime settings & CLI args ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
world_size = int(os.getenv("WORLD_SIZE", '1'))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
parser.add_argument("--data_dir", type=str, required=True, help="Input data directory")
parser.add_argument("--model_name_or_path", type=str, required=True, help="Pretrained model path")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="LoRA checkpoint path")
parser.add_argument("--gradient_checkpointing", action='store_true')
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
parser.add_argument("--mode", type=str, default="sft")
parser.add_argument("--deepspeed_config", type=str, default="./data_utils/deepspeed_config.json")
parser.add_argument("--num_train_epochs", default=3, type=int)
parser.add_argument("--epoch_offset", default=0, type=int, help="resume: number of epochs already done (only changes file names / logs)")
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
parser.add_argument("--few_shot", type=bool, default=False)
parser.add_argument('--statistic_mode', default='True')
parser.add_argument("--offload_optimizer", action='store_true')
parser.add_argument("--emotion_prediction", default=False)
parser.add_argument("--eval_percent", default=0.02, type=float)
parser.add_argument("--data_percent", default=1.0, type=float)
parser.add_argument("--max_new_tokens", default=16, type=int, help="generation budget at eval (ERC labels are short)")
parser.add_argument("--beta", default=1.0)
parser.add_argument("--theta", default=1.0)

args = parser.parse_args()
do_sample = args.top_k is not None or args.top_p is not None or args.num_beams > 1 or args.temp is not None

if isinstance(args.do_train, str):
    args.do_train = (args.do_train.lower() == 'true')
if isinstance(args.do_eval, str):
    args.do_eval = (args.do_eval.lower() == 'true')
if isinstance(args.lora, str):
    args.lora = (args.lora.lower() == 'true')
if isinstance(args.emotion_prediction, str):
    args.emotion_prediction = (args.emotion_prediction.lower() == 'true')

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
    "epoch_offset": args.epoch_offset,
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
    "statistic_mode": args.statistic_mode,
    "mode": args.mode,
    "emotion_prediction": args.emotion_prediction,
    "beta": args.beta,
    "theta": args.theta,
    "gradient_checkpointing": args.gradient_checkpointing,
    "data_percent": args.data_percent,
    "max_new_tokens": args.max_new_tokens,
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

# ----------------- 6. Data Preparation ---------------------------------------
train_file = os.path.join(args.data_dir, "train.json")
dev_file = os.path.join(args.data_dir, "test.json")

train_dataset, dev_dataset = None, None
train_collator, dev_collator = None, None

if args.do_train:
    df_train = read_data(train_file, percent=args.data_percent, random_seed=args.seed)
    train_dataset = Seq2SeqDataset(args, df_train, mode='train')
    train_collator = Seq2SeqCollator(args, tokenizer, mode="train")

if args.do_eval and os.path.exists(dev_file):
    df_dev = read_data(dev_file, percent=args.eval_percent, random_seed=args.seed)
    dev_dataset = Seq2SeqDataset(args, df_dev, mode='dev')
    dev_collator = Seq2SeqCollator(args, tokenizer, mode="dev")

# ----------------- 7. DeepSpeed Initialization -------------------------------
if args.do_train:
    # DeepSpeed steps the LR scheduler once per accumulated optimizer step, not per micro-batch
    t_total = math.ceil(len(train_dataset) / (args.batch_size * args.gradient_accumulation_steps * max(1, world_size))) * args.num_train_epochs
    warmup_steps = math.ceil(t_total * args.warmup_ratio) if args.warmup_steps is None else args.warmup_steps
    args.warmup_steps = warmup_steps

    optimizer_grouped_parameters = getOptimizerGroup(model=model)
    optimizer_class = DeepSpeedCPUAdam if deepspeed_config["zero_optimization"]["offload_optimizer"]["device"] == "cpu" else FusedAdam
    optimizer = optimizer_class(optimizer_grouped_parameters, lr=args.learning_rate, betas=[0.9, 0.95])
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=t_total, num_warmup_steps=warmup_steps)

    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=deepspeed_config,
        collate_fn=train_collator
    )
    model = model_engine

# ----------------- 8. Main Loop ----------------------------------------------
if __name__ == "__main__":
    emotional_label_dict, emotional_label_str = get_labels_attr(dataset=args.dataset)
    base_labels = [k for k in emotional_label_dict.keys() if k == k.capitalize() or k == k.lower()]

    if args.do_train:
        global_steps = 0
        best_f1_score = 0
        if args.epoch_offset:  # resuming: keep the best F1 of the epochs finished before the interruption
            _prev = [float(m.group(1)) for f in os.listdir(args.output_dir)
                     for m in [re.fullmatch(r"preds_for_eval_\d+_([\d.]+)\.text", f)] if m]
            best_f1_score = max(_prev, default=0)
            print(f"Resuming after {args.epoch_offset} finished epochs; best F1 so far {best_f1_score}%")
        for epoch in range(args.num_train_epochs):
            ep = epoch + args.epoch_offset  # epoch number used in logs and file names
            model.train()
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running epoch {ep} / {args.num_train_epochs + args.epoch_offset}",
                disable=False
            )
            for step, batch in enumerate(batch_iterator):
                batch = _get_input_dict(batch)
                outputs = model(**batch)
                loss = outputs.loss

                model.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                model.step()

                batch_iterator.set_description(
                    f"Epoch {ep}/{args.num_train_epochs + args.epoch_offset} | Loss: {loss.item():.4f}"
                )

                if step % args.gradient_accumulation_steps == 0:
                    global_steps += 1

            if int(args.local_rank) in (-1, 0):
                adapter_save_dir = os.path.join(args.output_dir, f'lora_epoch_{ep}')
                (model.module if hasattr(model, 'module') else model).save_pretrained(adapter_save_dir)
                print(f"Saved LoRA checkpoint to {adapter_save_dir}")

            if args.do_eval and dev_dataset is not None:
                model.eval()
                targets = list(df_dev["output"])
                eval_sampler = SequentialSampler(dev_dataset)
                eval_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, sampler=eval_sampler, collate_fn=dev_collator, num_workers=2)
                all_outputs = []
                eval_inputs_iter = []

                print(f"\n***** Evaluating Epoch {ep} *****\n")
                for eval_batch in tqdm(eval_dataloader):
                    eval_batch = eval_batch.to(device)
                    eval_inputs_iter.extend(eval_batch["input_ids"])
                    with torch.no_grad():
                        if "token_type_ids" in eval_batch:
                            eval_batch.pop("token_type_ids")
                        outputs = model.generate(
                            **eval_batch,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    outputs[outputs[:, :] < 0] = tokenizer.pad_token_id
                    all_outputs.extend(outputs)

                eval_inputs_iter = [tokenizer.decode(e_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for e_id in eval_inputs_iter]
                outs = [tokenizer.decode(o_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o_id in all_outputs]
                preds_for_eval = []
                all_answers = []

                for index, o in enumerate(outs):
                    this_input = eval_inputs_iter[index]
                    answer = o.replace(this_input, "").strip() if this_input in o else o.strip()
                    all_answers.append(answer)
                    preds_for_eval.append({
                        "index": index,
                        "input": this_input,
                        "output": answer,
                        "target": targets[index]
                    })

                preds = []
                golds = []
                confuse_case = []
                for index, answer in enumerate(all_answers):
                    raw_gold = current_emotion_text(targets[index]).strip('<>')
                    gold_label_num = emotional_label_dict.get(raw_gold, emotional_label_dict.get(raw_gold.lower(), 0))
                    golds.append(gold_label_num)

                    match_res = match_text(current_emotion_text(answer), base_labels)
                    if match_res:
                        pred_label_num = emotional_label_dict.get(match_res[0], emotional_label_dict.get(match_res[0].lower(), 0))
                    else:
                        nearest = optimize_output(current_emotion_text(answer), base_labels)
                        pred_label_num = emotional_label_dict.get(nearest, emotional_label_dict.get(nearest.lower(), 0))
                        confuse_case.append(index)
                    preds.append(pred_label_num)

                score, res_matrix = report_score(dataset=args.dataset, golds=golds, preds=preds)
                print(f"Epoch {ep} Test Weighted-F1: {score['F1_SA']}% | Accuracy: {score['Acc_SA']}%")

                preds_path = os.path.join(args.output_dir, f"preds_for_eval_{ep}_{score['F1_SA']}.text")
                with open(preds_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(score) + f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{res_matrix}\n\n")
                    f.write(json.dumps(preds_for_eval, indent=2, ensure_ascii=False))

                if score['F1_SA'] > best_f1_score:
                    best_f1_score = score['F1_SA']
                    best_dir = os.path.join(args.output_dir, "best_checkpoint")
                    (model.module if hasattr(model, 'module') else model).save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    print(f"--> New best model saved to {best_dir} (F1: {best_f1_score}%)")

        if int(args.local_rank) in (-1, 0):
            final_dir = os.path.join(args.output_dir, "final_checkpoint")
            (model.module if hasattr(model, 'module') else model).save_pretrained(final_dir)
            print(f"Saved final LoRA adapter to {final_dir}")
        print(f"\nTraining Complete! Best Weighted-F1: {best_f1_score}%")
