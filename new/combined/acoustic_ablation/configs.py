"""Threshold configurations for the acoustic ablation: parse / name them and shortlist them from the CPU screen."""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "base+acoustic"))
import acoustic_clues as ac  # noqa: E402

_PAT = re.compile(r"^(5c|3c) p([\d-]+)(?: \(original\))? \| (global|speaker_pct|speaker) \| (5feat|8feat)$")


def parse(name, dataset):
    """'5c p10-30-70-90 | speaker_pct | 8feat' (or its slug) -> ClueConfig; shared with the real runs via acoustic_clues."""
    return ac.parse_config(name, dataset)[0]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower().replace("(original)", "")).strip("-")


def shortlist(dataset, top_k):
    """Best top_k configs of the CPU screen (controls/ceiling rows skipped)."""
    path = os.path.join(HERE, "results", f"{dataset}_screen.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} missing; run screen_thresholds.py --dataset {dataset} first")
    rows = [r for r in csv.DictReader(open(path)) if _PAT.match(r["config"])]
    return [r["config"] for r in rows[:top_k]]
