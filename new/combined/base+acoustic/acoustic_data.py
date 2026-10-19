"""Helpers shared by base+acoustic and acoustic_ablation: put the acoustic clue into the ERC prompts."""
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import acoustic_clues as ac  # noqa: E402
from coe_clues import DATA_DIR  # noqa: E402  (acoustic_clues puts ../base on sys.path)

CLUE_HEADER = "### Acoustic Clues:"


def add_clue(prompt, text):
    """Insert the clue after the current utterance and mention it in the task sentence."""
    marker = "\n\nBased on the context, what is the emotion"
    assert marker in prompt and "Dialogue History information." in prompt, "unexpected ERC prompt layout"
    prompt = prompt.replace("Dialogue History information.", "Dialogue History and Acoustic Clues information.", 1)
    return prompt.replace(marker, f"\n{CLUE_HEADER}\n{text}" + marker, 1)


_CTX_TAG = re.compile(r" \((?:very low|low|moderate|high|very high) pitch(?: with (?:very low|low|moderate|high|very high) variation)?\)$")


def add_context(prompt, hist_tags, cur_tag):
    """SpeechCueLLM 'audio_context': tag the last (up to) 3 context utterances and the current one with e.g. '(high pitch with
    low variation)'. hist_tags = tags of the previous turns, oldest first; None entries are skipped."""
    h0 = prompt.index("### Dialogue History:\n") + len("### Dialogue History:\n")
    h1 = prompt.index("### Current Utterance:\n")
    c0 = h1 + len("### Current Utterance:\n")
    c1 = prompt.index("\n", c0)
    lines = prompt[h0:h1].rstrip("\n").split("\n") if prompt[h0:h1].strip() else []
    for k, tag in enumerate(reversed(hist_tags)):
        idx = len(lines) - 1 - k
        if idx >= 0 and tag:
            lines[idx] += " " + tag
    hist = "\n".join(lines) + "\n" if lines else prompt[h0:h1]   # empty history: keep the original (blank) line untouched
    cur = prompt[c0:c1] + (" " + cur_tag if cur_tag else "")
    return prompt[:h0] + hist + prompt[h1:c0] + cur + prompt[c1:]


def strip_context(prompt):
    """Inverse of add_context (used by the integrity checks)."""
    return "\n".join(_CTX_TAG.sub("", l) for l in prompt.split("\n"))


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {len(rows):6d} rows -> {os.path.relpath(path, DATA_DIR)}")


def base_erc_dir(dataset, base_tag=""):
    return os.path.join(DATA_DIR, f"{dataset}_coe{base_tag}_erc")


def shuffled(texts, seed=0):
    """Control: the same clue sentences (incl. 'no audio' share) assigned to random turns."""
    t = list(texts); random.Random(seed).shuffle(t)
    return t


def holdout_split(dataset):
    """(train_rows, dev_rows) used for the ablation. EmoryNLP/MELD: their dev sets. IEMOCAP has no dev
    (its dev file equals test), so every 5th training dialogue is held out. Test is never touched.
    Each row gets 'src' ('train'|'dev') and 'idx' = its position in that split's base ERC file."""
    tr = ac.load_dataset(dataset, "train")
    for i, r in enumerate(tr):
        r["src"], r["idx"] = "train", i
    if dataset == "iemocap":
        ids = sorted({r["dialogue"] for r in tr}, key=lambda s: int(s.split(":")[1]))
        held = set(ids[::5])
        return [r for r in tr if r["dialogue"] not in held], [r for r in tr if r["dialogue"] in held]
    dv = ac.load_dataset(dataset, "dev")
    for i, r in enumerate(dv):
        r["src"], r["idx"] = "dev", i
    return tr, dv
