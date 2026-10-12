#!/usr/bin/env python3
"""Create trainbegin.json (seed rationales for STEeR) with the local Mistral.

The paper seeds STEeR with GPT-4o rationales (kept only when GPT-4o's own label is correct).
Without API access we use STaR-style *rationalisation*: the model is shown the gold label and
asked to justify it from persona/scene/history. The final sentence ("The emotional label of
<spk: "utt"> is <X>.") is appended programmatically so the format is exact. Seeds are class
balanced and removed from the STEeR pool. This is a deviation from the paper - report it.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from coe_clues import DATA_DIR, cfg

ONE_SHOT = (
    "Example rationale (for a different utterance, emotion Powerful): In the dialogue, Phoebe starts feeling "
    "vulnerable about her date but transitions to confidence after receiving support from her friends. Her "
    "assertive statement, indicating a desire for action and control, reflects an empowered emotional state."
)
BAD = re.compile(r"annotat|gold|label|the correct|<|>", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="emorynlp", choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--steer_dir", default=None, help="default: {dataset}_coe_steer")
    ap.add_argument("--model_path", default="/home/user2/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--per_label", type=int, default=10)
    ap.add_argument("--oversample", type=int, default=2, help="candidates per kept seed, for filtering")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    LABELS = cfg(args.dataset)["labels"]
    d = os.path.join(DATA_DIR, args.steer_dir or f"{args.dataset}_coe_steer")
    pool = [json.loads(l) for l in open(os.path.join(d, "train4infer.json"), encoding="utf-8")]
    rng = random.Random(args.seed)
    by_label = {l: [r for r in pool if r["target"] == l] for l in LABELS}
    cands = []
    for l in LABELS:
        rng.shuffle(by_label[l])
        cands += by_label[l][: args.per_label * args.oversample]
    print(f"{len(cands)} candidates for {args.per_label * len(LABELS)} seeds")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True))
    model.eval()

    def parse(r):
        ctx, _, ask = r["input"].partition("Please analyze the emotional state of")
        m = re.match(r'\s*<(.*?): "(.*)"> by first', ask, re.S)
        return ctx.strip(), m.group(1), m.group(2)

    def prompt(r):
        ctx, spk, utt = parse(r)
        txt = (f"{ctx}\n\nThe emotion of the current utterance is known to be {r['target']}. In 2-3 sentences, explain "
               f"why {spk} feels {r['target']} here, using the persona, scene and dialogue context. Describe the "
               f"reasoning only: do not repeat the utterance verbatim, and do not mention labels or annotations.\n{ONE_SHOT}")
        return tok.apply_chat_template([{"role": "user", "content": txt}], tokenize=False, add_generation_prompt=True)

    kept = {l: [] for l in LABELS}
    for i in range(0, len(cands), args.batch_size):
        batch = cands[i:i + args.batch_size]
        enc = tok([prompt(r) for r in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=140, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, o in zip(batch, out):
            txt = " ".join(tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).split())
            txt = re.sub(r"^(Rationale|Explanation)\s*:\s*", "", txt, flags=re.I)
            if not (40 <= len(txt) <= 700) or BAD.search(txt) or not txt.endswith("."):
                continue
            _, spk, utt = parse(r)
            if len(kept[r["target"]]) < args.per_label:
                kept[r["target"]].append((r, f"[Rationale]\n{txt} The emotional label of <{spk}: \"{utt}\"> is <{r['target']}>."))
        print(f"{min(i + args.batch_size, len(cands))}/{len(cands)}  kept " + str({k: len(v) for k, v in kept.items()}))

    seeds = [{"input": r["input"], "target": t} for l in LABELS for r, t in kept[l]]
    seed_inputs = {s["input"] for s in seeds}
    with open(os.path.join(d, "trainbegin.json"), "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    rest = [r for r in pool if r["input"] not in seed_inputs]
    with open(os.path.join(d, "train4infer.json"), "w", encoding="utf-8") as f:
        for r in rest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(seeds)} seeds -> trainbegin.json; pool now {len(rest)}", flush=True)
    del model
    torch.cuda.empty_cache()
    os._exit(0)  # the process otherwise lingers and keeps the GPU


if __name__ == "__main__":
    main()
