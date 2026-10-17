#!/usr/bin/env python3
"""Datasets for one ablation arm. The held-out DEV rows are written as test.json so the trainer's per-epoch evaluation
scores them; the real test set is never used.

arm = nocue | shuffled | default | <config name from the screen, e.g. '5c p10-30-70-90 | speaker_pct | 8feat'>
Output: Truebaseline/data/{dataset}_coe{base_tag}_abl_{arm-slug}/{train,test,dev}.json
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "base+acoustic"))
import acoustic_clues as ac  # noqa: E402
import acoustic_data as ad  # noqa: E402
import build_acoustic_data as bad  # noqa: E402
import configs  # noqa: E402


# SpeechCueLLM-style arms: name -> (form, audio_context tags, shuffled control). Each form has its own shuffled control.
GROUPS = {"volume": ("avg_intensity", "intensity_variation"), "pitch": ("avg_pitch", "pitch_std"), "rate": ("articulation_rate",)}


def _arm(form="description", ctx=False, shuf=False, drop=None, perc=None, norm=None):
    return dict(form=form, ctx=ctx, shuf=shuf, drop=drop, perc=perc, norm=norm)


# Component ablation of the SpeechCueLLM-style clue. Reference ("full") = description + audio_context + 5 features + 5-class
# percentiles (10/25/75/90) + the dataset's default normalisation. Every other arm changes ONE component of it.
FORM_ARMS = {
    # with audio_context tags (EmoryNLP / MELD)
    "desc_ctx": _arm(ctx=True),                                   # FULL reference
    "shuffled_desc_ctx": _arm(ctx=True, shuf=True),               # control for the reference: same text on random turns
    "desc": _arm(),                                               # - context tags
    "shuffled_desc": _arm(shuf=True),                             # control for "desc"
    "desc_ctx_nov": _arm(ctx=True, drop="volume"),                # - volume features (intensity, intensity variation)
    "desc_ctx_nop": _arm(ctx=True, drop="pitch"),                 # - pitch features (pitch, pitch variation; also drops the tags)
    "desc_ctx_nor": _arm(ctx=True, drop="rate"),                  # - speech-rate feature
    "desc_ctx_3c": _arm(ctx=True, perc=(25, 75)),                 # 5-class -> 3-class percentile cutoffs
    "desc_ctx_spk": _arm(ctx=True, norm="speaker_pct"),           # global -> per-speaker percentiles
    "both_ctx": _arm(form="both", ctx=True),                      # + impression sentence
    # without tags (also the IEMOCAP set, which has no audio_context): reference = "desc"
    "desc_nov": _arm(drop="volume"), "desc_nop": _arm(drop="pitch"), "desc_nor": _arm(drop="rate"),
    "desc_3c": _arm(perc=(25, 75)), "desc_spk": _arm(norm="speaker_pct"), "both": _arm(form="both"),
    "shuffled_both": _arm(form="both", shuf=True),
}
CONTROL_OF = {"desc": "shuffled_desc", "desc_ctx": "shuffled_desc_ctx", "both": "shuffled_both"}
# preset: the full component ablation for a dataset, and its reference arm
PRESET = {"ctx": ["desc_ctx", "shuffled_desc_ctx", "desc", "shuffled_desc", "desc_ctx_nov", "desc_ctx_nop", "desc_ctx_nor",
                  "desc_ctx_3c", "desc_ctx_spk", "both_ctx"],
          "noctx": ["desc", "shuffled_desc", "desc_nov", "desc_nop", "desc_nor", "desc_3c", "desc_spk", "both"]}
REFERENCE = {"ctx": "desc_ctx", "noctx": "desc"}


def preset_for(dataset):
    k = "noctx" if dataset == "iemocap" else "ctx"
    return PRESET[k], REFERENCE[k]


def arm_dir(dataset, arm, base_tag=""):
    return f"{dataset}_coe{base_tag}_abl_{configs.slug(arm)}"


def build_arm(dataset, arm, base_tag="", seed=0):
    train, dev = ad.holdout_split(dataset)
    src = ad.base_erc_dir(dataset, base_tag)
    base = {s: ad.read_rows(os.path.join(src, f"{s}.json")) for s in ("train", "dev")}
    if arm in FORM_ARMS:
        spec = FORM_ARMS[arm]
        form, ctx, shuf = spec["form"], spec["ctx"], spec["shuf"]
        if ctx and dataset == "iemocap":
            sys.exit("audio_context arms are not supported for IEMOCAP")
        over = {}
        if spec["drop"]:
            over["features"] = tuple(f for f in ac.FEATS5 if f not in GROUPS[spec["drop"]])
        if spec["perc"]:
            over["percentiles"] = spec["perc"]
        if spec["norm"]:
            over["norm"] = spec["norm"]
        cfg = ac.default_config(dataset, **over)
        def per_turn(rs):
            b = ac.clue_bundle(cfg, train, rs, form, min_duration=ac.MIN_DURATION)
            pt = [(t, bad.context_tags(rs, b, k) if ctx else [], c) for k, (t, c) in enumerate(b)]
            return ad.shuffled(pt, seed + (1 if rs is dev else 0)) if shuf else pt
        def form_rows(rs):
            out = []
            for r, (text, htags, ctag) in zip(rs, per_turn(rs)):
                b = base[r["src"]][r["idx"]]
                p = ad.add_clue(b["input"], text) if text else b["input"]
                out.append({"input": ad.add_context(p, htags, ctag) if ctx else p, "target": b["target"]})
            return out
        d = os.path.join(ad.DATA_DIR, arm_dir(dataset, arm, base_tag))
        tr_rows, dv_rows = form_rows(train), form_rows(dev)
        print(f"{dataset} arm '{arm}': train {len(tr_rows)}  held-out dev {len(dv_rows)} -> {os.path.basename(d)}")
        for name, rows_ in (("train", tr_rows), ("test", dv_rows), ("dev", dv_rows)):
            ad.write_rows(os.path.join(d, f"{name}.json"), rows_)
        return os.path.basename(d)
    if arm in ("nocue",):
        txt_tr = txt_dv = None
    else:
        cfg = ac.default_config(dataset) if arm in ("default", "shuffled") else configs.parse(arm, dataset)
        txt_tr = ac.clue_texts(cfg, train, train)          # thresholds fitted on the training part only
        txt_dv = ac.clue_texts(cfg, train, dev)
        if arm == "shuffled":
            txt_tr, txt_dv = ad.shuffled(txt_tr, seed), ad.shuffled(txt_dv, seed + 1)
    def rows(rs, txts):
        out = []
        for i, r in enumerate(rs):
            b = base[r["src"]][r["idx"]]
            out.append({"input": b["input"] if txts is None else ad.add_clue(b["input"], txts[i]), "target": b["target"]})
        return out
    d = os.path.join(ad.DATA_DIR, arm_dir(dataset, arm, base_tag))
    tr_rows, dv_rows = rows(train, txt_tr), rows(dev, txt_dv)
    print(f"{dataset} arm '{arm}': train {len(tr_rows)}  held-out dev {len(dv_rows)} -> {os.path.basename(d)}")
    ad.write_rows(os.path.join(d, "train.json"), tr_rows)
    ad.write_rows(os.path.join(d, "test.json"), dv_rows)
    ad.write_rows(os.path.join(d, "dev.json"), dv_rows)
    return os.path.basename(d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base_tag", default="")
    a = ap.parse_args()
    build_arm(a.dataset, a.arm, a.base_tag)
