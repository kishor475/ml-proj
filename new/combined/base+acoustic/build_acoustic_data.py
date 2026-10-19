#!/usr/bin/env python3
"""Build ERC datasets with acoustic clues (one extra '### Acoustic Clues:' block per prompt).

Input : the base ERC data  Truebaseline/data/{dataset}_coe{base_tag}_erc/{train,dev,test}.json  (built by base/build_coe_data.py)
Output: Truebaseline/data/{dataset}_coe{base_tag}_acoustic_{variant}[_<config slug>]_erc/{train,dev,test}.json
Variants:
  clue      the real clues, thresholds fitted on TRAIN only with the dataset's default recipe (see acoustic_clues.DEFAULTS)
  shuffled  control: the same clue sentences assigned to random turns (same marginal distribution, no real audio information)
Only ERC rows get clues: the auxiliary stage and the STEeR rationales of the base run are reused unchanged.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import acoustic_clues as ac  # noqa: E402
import acoustic_data as ad  # noqa: E402


_FORM_TAG = {"impression": "", "description": "_desc", "both": "_both"}


def data_name(dataset, base_tag, variant, slug=None, form="impression", ctx=False):
    return (f"{dataset}_coe{base_tag}_acoustic_{variant}{_FORM_TAG[form]}{'_ctx' if ctx else ''}"
            f"{'_' + slug if slug else ''}_erc")


def context_tags(rows, bundles, k, n=3):
    """Tags of the previous (up to n) turns of the same dialogue, oldest first."""
    tags = []
    for j in range(max(0, k - n), k):
        if rows[j]["dialogue"] == rows[k]["dialogue"]:
            tags.append(bundles[j][1])
    return tags


def build(dataset, variant, base_tag="", cfg=None, out_name=None, seed=0, slug=None, form="impression", ctx=False):
    cfg = cfg or ac.default_config(dataset)
    src = ad.base_erc_dir(dataset, base_tag)
    if not os.path.exists(os.path.join(src, "train.json")):
        sys.exit(f"missing {src}; run base/build_coe_data.py --dataset {dataset} first")
    rows = {s: ac.load_dataset(dataset, s) for s in ("train", "dev", "test")}
    if ctx and dataset == "iemocap":
        sys.exit("--audio_context is not supported for IEMOCAP (its history contains unlabelled utterances)")
    out = os.path.join(ad.DATA_DIR, out_name or data_name(dataset, base_tag, variant, slug, form, ctx))
    print(f"{dataset} / {variant} / {cfg} -> {os.path.basename(out)}")
    for split in ("train", "dev", "test"):
        base_rows = ad.read_rows(os.path.join(src, f"{split}.json"))
        assert len(base_rows) == len(rows[split]), f"{split}: base ERC rows {len(base_rows)} != aligned rows {len(rows[split])}"
        if form == "impression" and not ctx:      # original path: output stays byte-identical to earlier runs
            texts = ac.clue_texts(cfg, rows["train"], rows[split])
            if variant == "shuffled":
                texts = ad.shuffled(texts, seed)
            ad.write_rows(os.path.join(out, f"{split}.json"),
                          [{"input": ad.add_clue(b["input"], t), "target": b["target"]} for b, t in zip(base_rows, texts)])
            continue
        # SpeechCueLLM-style: clue text (+ optional tags on the last 3 context turns); None = no clue for that turn
        bundles = ac.clue_bundle(cfg, rows["train"], rows[split], form, min_duration=ac.MIN_DURATION)
        per_turn = [(t, context_tags(rows[split], bundles, k) if ctx else [], c) for k, (t, c) in enumerate(bundles)]
        if variant == "shuffled":                  # control: whole bundles go to random turns
            per_turn = ad.shuffled(per_turn, seed)
        out_rows = []
        for b, (text, htags, ctag) in zip(base_rows, per_turn):
            p = ad.add_clue(b["input"], text) if text else b["input"]
            if ctx:
                p = ad.add_context(p, htags, ctag)
            out_rows.append({"input": p, "target": b["target"]})
        ad.write_rows(os.path.join(out, f"{split}.json"), out_rows)
    return os.path.basename(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--variant", default="clue", choices=["clue", "shuffled"])
    ap.add_argument("--base_tag", default="", help="'' or _scene (IEMOCAP scene variant) - which base ERC data to extend")
    ap.add_argument("--config", default=None, help="threshold config from the ablation, e.g. \"5c p10-30-70-90 | speaker_pct | 8feat\" (default: the dataset's original recipe)")
    ap.add_argument("--form", default="impression", choices=ac.FORMS,
                    help="impression = interpretive sentence (our original); description = SpeechCueLLM's compact factual form (headline setting); both")
    ap.add_argument("--audio_context", action="store_true", help="SpeechCueLLM audio_context: tag the last 3 context turns with pitch/variation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg, slug = ac.parse_config(a.config, a.dataset) if a.config else (None, None)
    build(a.dataset, a.variant, a.base_tag, cfg=cfg, seed=a.seed, slug=slug, form=a.form, ctx=a.audio_context)
