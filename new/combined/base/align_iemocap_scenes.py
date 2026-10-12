#!/usr/bin/env python3
"""Recover the IEMOCAP scene of every utterance from the original transcripts.

Our IEMOCAP ERC rows carry no scene id, and one "dialogue" in them is several recordings concatenated.
For each dialogue we segment its turns into runs of one scene (Viterbi over the 15 scenes, emission =
does this utterance text occur in a transcript of that scene, switch penalty 3) and write the per-turn
scene ids to Truebaseline/data/iemocap_turn_scenes.json. Only scene ids are stored, no transcript text.
Checked on our data: for improvised scenes (unique text per recording) every run maps to a single
transcript file (mean purity 0.95, min run purity > 0.8).

Usage: python align_iemocap_scenes.py --transcripts /path/with/Session1..5/*.txt
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from coe_clues import DATA_DIR, load_dialogues

SCENES = ['impro0%d' % i for i in range(1, 9)] + [
    'script01_1', 'script01_2', 'script01_3', 'script02_1', 'script02_2', 'script03_1', 'script03_2']
SWITCH = 3.0

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def scene_of(fname):  # Ses01F_impro05a -> impro05, Ses05M_script01_1b -> script01_1
    return re.sub(r"[ab]$", "", re.sub(r"^Ses\d\d[FM]_", "", fname))


def viterbi(texts, idx):
    K = len(SCENES)
    emis = []
    for t in texts:
        hit = idx.get(norm(t))
        emis.append([0.0] * K if not hit else [0.0 if s in hit else 1.0 for s in SCENES])
    cost, back = [emis[0][:]], []
    for i in range(1, len(texts)):
        prev = cost[-1]
        best = min(range(K), key=lambda k: prev[k])
        cur, bk = [], []
        for k in range(K):
            if prev[k] <= prev[best] + SWITCH:
                cur.append(prev[k] + emis[i][k]); bk.append(k)
            else:
                cur.append(prev[best] + SWITCH + emis[i][k]); bk.append(best)
        cost.append(cur); back.append(bk)
    k = min(range(K), key=lambda k: cost[-1][k])
    path = [k]
    for bk in reversed(back):
        k = bk[k]; path.append(k)
    return [SCENES[k] for k in reversed(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True, help="folder containing Session1..5 with the *.txt transcripts")
    a = ap.parse_args()
    idx = collections.defaultdict(set)
    files = glob.glob(os.path.join(a.transcripts, "Session*", "*.txt"))
    if not files:
        sys.exit(f"no transcripts under {a.transcripts}/Session*/")
    for p in files:
        sc = scene_of(os.path.basename(p)[:-4])
        for line in open(p, encoding="utf-8", errors="ignore"):
            m = re.match(r"(\S+) \[[\d.\-]+\]: (.*)", line.strip())
            if m and norm(m.group(2)):
                idx[norm(m.group(2))].add(sc)
    out, runs = {}, 0
    for split in ("train", "dev", "test"):
        for d in load_dialogues(split, "iemocap"):
            out[d["id"]] = viterbi([t["text"] for t in d["turns"]], idx)
            runs += 1 + sum(x != y for x, y in zip(out[d["id"]], out[d["id"]][1:]))
    path = os.path.join(DATA_DIR, "iemocap_turn_scenes.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"{len(out)} dialogues, {runs} scene runs from {len(files)} transcript files -> {path}")
    print(collections.Counter(s for p in out.values() for s in p).most_common())


if __name__ == "__main__":
    main()
