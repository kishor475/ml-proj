#!/usr/bin/env python3
"""Generate a short synthetic scene description per EmoryNLP dialogue with the
local Mistral (4-bit). Uses ONLY the first few utterances (no labels, no later
turns) and asks for physical setting only, so no emotion or future context leaks.
Resumable: results are cached in Truebaseline/data/emorynlp_scenes.json.

NOTE: these scenes are synthetic (the paper used frontier models); report them as such.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from coe_clues import cfg, load_dialogues, load_scenes, scenes_path

UNKNOWN = "Unknown setting"
BANNED = re.compile(
    r"\b(happy|sad|angry|upset|tense|nervous|excited|joy\w*|mad|scared|afraid|anxious|"
    r"emotion\w*|mood|feel\w*|argu\w*|fight\w*|laugh\w*|cry\w*|worried|annoyed|frustrat\w*)\b", re.I)

INSTRUCTION = (
    "Below are the opening lines of a scene from the TV sitcom Friends. In ONE short sentence "
    "(max 20 words), state only the physical setting where it takes place, and who is present if "
    "clear. Do not mention emotions, mood, plot or relationships. If the setting is neither stated "
    f"nor clearly implied, answer exactly: {UNKNOWN}\n\nOpening lines:\n"
)


def clean(ans):
    ans = ans.strip().split("\n")[0].strip().strip('"')
    if not ans or BANNED.search(ans) or len(ans.split()) > 30:
        return UNKNOWN
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="emorynlp", choices=["emorynlp", "meld"])
    ap.add_argument("--model_path", default="/home/user2/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--n_lines", type=int, default=3, help="opening utterances shown to the model")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only process first N dialogues (testing)")
    args = ap.parse_args()
    if not cfg(args.dataset)["scene"]:
        sys.exit(f"{args.dataset} has no scene clue")
    SCENES_PATH = scenes_path(args.dataset)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    todo = []
    scenes = load_scenes(args.dataset)
    for split in ("train", "dev", "test"):
        for d in load_dialogues(split, args.dataset):
            if d["id"] not in scenes:
                todo.append(d)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(scenes)} cached, {len(todo)} to generate")
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True))
    model.eval()

    def prompt(d):
        lines = "\n".join(t["line"] for t in d["turns"][: args.n_lines])
        return tok.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION + lines}], tokenize=False, add_generation_prompt=True)

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i: i + args.batch_size]
        enc = tok([prompt(d) for d in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        for d, o in zip(batch, out):
            raw = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            scenes[d["id"]] = {"scene": clean(raw), "raw": raw.strip()}
        with open(SCENES_PATH, "w", encoding="utf-8") as f:
            json.dump(scenes, f, indent=1, ensure_ascii=False)
        print(f"{min(i + args.batch_size, len(todo))}/{len(todo)}")

    unk = sum(v["scene"] == UNKNOWN for v in scenes.values())
    print(f"done: {len(scenes)} scenes, {unk} 'Unknown setting' ({100 * unk / len(scenes):.0f}%)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
