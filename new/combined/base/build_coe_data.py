#!/usr/bin/env python3
"""Build CoE-format data (persona + scene clues, Appendix E prompt layout) for emorynlp / meld / iemocap.

Writes three data dirs under Truebaseline/data/ (each with train/dev/test.json):
  {dataset}_coe{tag}_aux    stage 1 of Strategy B : speaker-ID + role-play (+ real STEeR rows if given)
  {dataset}_coe{tag}_erc    stage 2 of Strategy B : ERC only
  {dataset}_coe{tag}_mixed  everything in one file (the "Multi-Task" strategy, for comparison)
plus {dataset}_coe{tag}_steer/ (train4infer.json pool, dev/test) for the STEeR loop.
dev/test are always ERC-only. Clues a dataset does not have (IEMOCAP: persona, scene) are left out
automatically. Scenes come from generate_scenes.py unless --no_scene.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from coe_clues import (DATA_DIR, IEMOCAP_SCENES, cfg, label_str, load_dialogues, load_iemocap_turn_scenes,
                       load_scenes, persona_block)


def clue_blocks(a, scene):
    out = ""
    if a.persona:
        out += f"### Speaker Persona:\n{persona_block(a.dataset)}\n"
    if a.scene:
        out += f"### Scene:\n{scene}\n"
    return out


def erc_row(a, t, scene):
    inp = (
        "As an expert in Conversational Emotion Recognition (ERC), your task is to predict the emotion "
        "by analyzing the " + a.clue_names + "Dialogue History information.\n"
        + clue_blocks(a, scene)
        + f"### Dialogue History:\n{t['history']}\n### Current Utterance:\n{t['line']}\n\n"
        "Based on the context, what is the emotion of the current utterance?\n"
        f"Choose from: {label_str(a.dataset)}\nAnswer with just the emotion label."
    )
    return {"input": inp, "target": t["emotion"]}


def steer_row(a, t, scene):
    inp = (
        "As an expert in Conversational Emotion Recognition (ERC), your task is to predict the emotion "
        "by analyzing the " + a.clue_names + "Dialogue History information.\n"
        + clue_blocks(a, scene)
        + f"### Dialogue History:\n{t['history']}\n### Current Utterance:\n{t['line']}\n\n"
        f"Please analyze the emotional state of <{t['speaker']}: \"{t['text']}\"> by first providing your "
        f"rationale based on the given context, then select the emotional label from {label_str(a.dataset)}."
    )
    return {"input": inp, "target": t["emotion"]}


def speaker_row(a, t):
    # Paper: clues are System + Speaker Persona only; candidates are the six main characters.
    cands = ", ".join(cfg(a.dataset)["main"])
    inp = ("You are now an expert in Conversational Emotion Recognition. Your task is to predict the speaker "
           "by analyzing sentences. Here's a list of individuals and their Speaker Persona, which will assist "
           f"you in understanding them better.\n### Speaker Persona:\n{persona_block(a.dataset)}\n"
           f"Please select the Speaker of <\"{t['text']}\", Emotion: {t['emotion']}> from <{cands}>.")
    return {"input": inp, "target": t["speaker"]}


def roleplay_row(a, t, scene):
    inp = (
        "You are now an expert role-play actor. Analyze the character's " + a.rp_names + "Dialogue History "
        "to predict their most likely next line.\n"
        + clue_blocks(a, scene)
        + f"### Dialogue History:\n{t['history']}\n"
        f"You will stay in character whenever possible, and generate responses in <{t['emotion']}> emotion "
        f"as if you were <{t['speaker']}>."
    )
    return {"input": inp, "target": t["text"]}


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {len(rows):6d} rows -> {os.path.relpath(path, DATA_DIR)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="emorynlp", choices=["emorynlp", "meld", "iemocap"])
    ap.add_argument("--no_persona", action="store_true")
    ap.add_argument("--no_scene", action="store_true")
    ap.add_argument("--iemocap_scene", action="store_true",
                    help="IEMOCAP only: add a per-turn scene clue recovered from the original transcripts "
                         "(deviation from the paper, written to iemocap_coe_scene_*)")
    ap.add_argument("--steer_rows", default=None,
                    help="JSONL of real STEeR rows (memory.json, new prompt format) to add to the aux data")
    a = ap.parse_args()
    c = cfg(a.dataset)
    if a.iemocap_scene and a.dataset != "iemocap":
        sys.exit("--iemocap_scene only applies to --dataset iemocap")
    tag = ("_nopersona" if a.no_persona else "") + ("_noscene" if a.no_scene else "") + ("_scene" if a.iemocap_scene else "")
    base = f"{a.dataset}_coe{tag}"
    a.persona = bool(c["persona"]) and not a.no_persona
    a.scene = (c["scene"] and not a.no_scene) or a.iemocap_scene
    a.clue_names = ("Speaker Persona, " if a.persona else "") + ("Scene and " if a.scene else "")
    if a.clue_names and not a.clue_names.endswith(" and "):
        a.clue_names = a.clue_names.rstrip(", ") + " and "

    # role-play header keeps the paper's wording ("[Speaker_Persona], [Scene], and [History_dialogue]")
    a.rp_names = a.clue_names.replace("Scene and ", "Scene, and ") if a.persona and a.scene else a.clue_names
    splits = {s: load_dialogues(s, a.dataset) for s in ("train", "dev", "test")}
    scenes = load_scenes(a.dataset) if a.scene and not a.iemocap_scene else {}
    turn_scenes = load_iemocap_turn_scenes() if a.iemocap_scene else {}
    if a.scene and not a.iemocap_scene:
        missing = [d["id"] for ds in splits.values() for d in ds if d["id"] not in scenes]
        if missing:
            sys.exit(f"{len(missing)} dialogues have no scene yet; run generate_scenes.py --dataset {a.dataset} (or --no_scene)")

    def sc(d, i=0):
        if a.iemocap_scene:
            return IEMOCAP_SCENES[turn_scenes[d["id"]][i]]
        scene = scenes.get(d["id"], {}).get("scene", "")
        # collapse the spelling variants the generator produced for the same place
        return re.sub(r"^Central Perk( Coffee ?(house|shop)| Cafe)?", "Central Perk coffeehouse", scene, flags=re.I)

    erc = {s: [erc_row(a, t, sc(d, i)) for d in ds for i, t in enumerate(d["turns"])] for s, ds in splits.items()}

    aux, steer_pool = [], []
    for d in splits["train"]:
        for i, t in enumerate(d["turns"]):
            steer_pool.append(steer_row(a, t, sc(d, i)))
            if t["speaker"] not in c["main"]:
                continue
            if c["speaker_task"]:
                aux.append(speaker_row(a, t))
            if t["history"]:  # paper App. B (F1): no role-play on a dialogue's first utterance
                aux.append(roleplay_row(a, t, sc(d, i)))
    if a.steer_rows:
        with open(a.steer_rows, encoding="utf-8") as f:
            aux += [json.loads(l) for l in f]

    print(f"Writing {base}_* (persona={a.persona}, scene={a.scene})")
    for name, train in (("aux", aux), ("erc", erc["train"]), ("mixed", aux + erc["train"])):
        out = os.path.join(DATA_DIR, f"{base}_{name}")
        write(os.path.join(out, "train.json"), train)
        write(os.path.join(out, "dev.json"), erc["dev"])
        write(os.path.join(out, "test.json"), erc["test"])
    steer_dir = os.path.join(DATA_DIR, f"{base}_steer")
    seeds_path = os.path.join(steer_dir, "trainbegin.json")
    if os.path.exists(seeds_path):  # seed rationales are already in the training memory, keep them out of the pool
        with open(seeds_path, encoding="utf-8") as f:
            seed_inputs = {json.loads(l)["input"] for l in f if l.strip()}
        steer_pool = [r for r in steer_pool if r["input"] not in seed_inputs]
    write(os.path.join(steer_dir, "train4infer.json"), steer_pool)
    write(os.path.join(steer_dir, "dev.json"), erc["dev"])
    write(os.path.join(steer_dir, "test.json"), [steer_row(a, t, sc(d, i)) for d in splits["test"] for i, t in enumerate(d["turns"])])


if __name__ == "__main__":
    main()
