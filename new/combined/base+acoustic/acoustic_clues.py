"""Acoustic clues for CoE: align per-utterance audio features to our ERC turns, categorise them with
configurable percentile thresholds, and verbalise them with the SpeechCueLLM-style impression template
(copied from preprocessing/postprocess_emorynlp.py so the default config reproduces the original clues).

Alignment (verified on our data):
  MELD     text + order against the audio CSV rows                      (99.9-100 % of turns)
  EmoryNLP text + order against the original emotion-detection JSON, id -> sea*_ep*_sc*_utt*.mp4 (100 %; ~75 % have audio)
  IEMOCAP  per-recording Viterbi over text membership, then monotone text alignment inside a recording;
           validated by agreement between the audio CSV's emotion code and our label.
"""
import csv
import difflib
import json
import math
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "base"))
from coe_clues import load_dialogues  # noqa: E402

ROOT = os.path.realpath(os.path.join(HERE, "..", "..", ".."))
AUDIO_DIR = os.environ.get("ACOUSTIC_AUDIO_DIR", os.path.join(ROOT, "audio"))
EMORY_JSON_DIR = os.environ.get("EMORYNLP_JSON_DIR", os.path.join(ROOT, "emotion-detection"))

FEATS5 = ["avg_pitch", "pitch_std", "avg_intensity", "intensity_variation", "articulation_rate"]  # used by the original clue text
FEATS8 = ["duration", "avg_intensity", "intensity_variation", "avg_pitch", "pitch_std", "pitch_range",
          "articulation_rate", "mean_hnr"]

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def _read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def _feats(row):
    return {f: _num(row.get(f)) for f in FEATS8}


# ----------------------------------------------------------------------------- alignment
def align_meld(split):
    P = _read(os.path.join(AUDIO_DIR, f"meld_processed_5class_{split}_audio_features.csv"))
    R = {r["filename"]: r for r in _read(os.path.join(AUDIO_DIR, f"meld_{split}_audio_features.csv"))}
    turns = [(d, t) for d in load_dialogues(split, "meld") for t in d["turns"]]
    sm = difflib.SequenceMatcher(None, [norm(t["text"]) for _, t in turns], [norm(p["Utterance"]) for p in P], autojunk=False)
    out = [None] * len(turns)
    for b in sm.get_matching_blocks():
        for k in range(b.size):
            r = R.get(P[b.b + k]["filename"])
            if r:
                out[b.a + k] = _feats(r)
    return turns, out


def align_emorynlp(split):
    js = {"train": "trn", "dev": "dev", "test": "tst"}[split]
    J = json.load(open(os.path.join(EMORY_JSON_DIR, f"emotion-detection-{js}.json"), encoding="utf-8"))
    R = {r["filename"]: r for r in _read(os.path.join(AUDIO_DIR, "emorynlp_audio_features.csv"))}
    labels = {'Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared'}
    ref = [u for ep in J["episodes"] for sc in ep["scenes"] for u in sc["utterances"] if u.get("emotion", "") in labels]
    turns = [(d, t) for d in load_dialogues(split, "emorynlp") for t in d["turns"]]
    assert len(ref) == len(turns) and all(norm(u.get("transcript", "")) == norm(t["text"]) for u, (_, t) in zip(ref, turns)), \
        "EmoryNLP JSON does not line up with our turns"
    out = []
    for u in ref:
        p = u["utterance_id"].split("_")
        fn = f"sea{int(p[0][1:])}_ep{int(p[1][1:])}_sc{int(p[2][1:])}_utt{int(p[3][1:])}.mp4"
        out.append(_feats(R[fn]) if fn in R else None)
    return turns, out


_IEMO_CODE = {"neu": "neutral", "fru": "frustrated", "ang": "angry", "sad": "sad", "hap": "happy", "exc": "excited"}


def align_iemocap(split, switch=3.0):
    rows = _read(os.path.join(AUDIO_DIR, "iemocap_audio_features.csv"))
    modes = ("train", "valid") if split == "train" else ("test",)
    vids = defaultdict(list)
    for r in rows:
        if r["mode"] in modes and r["emotion"] in _IEMO_CODE:
            vids[r["video_id"]].append(r)
    names = sorted(vids)
    for v in names:
        vids[v].sort(key=lambda r: int(r["segment_id"]))
    vtext = {v: [norm(r["text"]) for r in vids[v]] for v in names}
    vset = {v: set(vtext[v]) for v in names}
    turns = [(d, t) for d in load_dialogues(split, "iemocap") for t in d["turns"]]
    out = [None] * len(turns)
    pos = 0
    for d in load_dialogues(split, "iemocap"):
        n = len(d["turns"]); texts = [norm(t["text"]) for t in d["turns"]]
        K = len(names)
        emis = [[0.0 if texts[i] in vset[v] else 1.0 for v in names] for i in range(n)]
        cost = [emis[0][:]]; back = []
        for i in range(1, n):
            prev = cost[-1]; best = min(range(K), key=prev.__getitem__)
            cur, bk = [], []
            for k in range(K):
                if prev[k] <= prev[best] + switch:
                    cur.append(prev[k] + emis[i][k]); bk.append(k)
                else:
                    cur.append(prev[best] + switch + emis[i][k]); bk.append(best)
            cost.append(cur); back.append(bk)
        k = min(range(K), key=cost[-1].__getitem__); path = [k]
        for bk in reversed(back):
            k = bk[k]; path.append(k)
        path.reverse()
        i = 0
        while i < n:                       # runs of one recording -> monotone text alignment inside it
            j = i
            while j < n and path[j] == path[i]:
                j += 1
            v = names[path[i]]
            sm = difflib.SequenceMatcher(None, texts[i:j], vtext[v], autojunk=False)
            for b in sm.get_matching_blocks():
                for q in range(b.size):
                    r = vids[v][b.b + q]
                    f = _feats(r); f["_vid"] = v; f["_code"] = _IEMO_CODE[r["emotion"]]
                    out[pos + i + b.a + q] = f
            i = j
        pos += n
    return turns, out


ALIGN = {"meld": align_meld, "emorynlp": align_emorynlp, "iemocap": align_iemocap}


def speaker_key(dataset, turn, feats):
    if dataset == "iemocap":  # actor = session + gender
        v = (feats or {}).get("_vid")
        return f"{v[:5]}_{turn['speaker']}" if v else turn["speaker"]
    return turn["speaker"]


def load_dataset(dataset, split):
    """-> list of dict(turn, dialogue, feats|None, speaker) in the same order as our flattened ERC turns."""
    turns, feats = ALIGN[dataset](split)
    return [dict(dialogue=d["id"], turn=t, feats=f, speaker=speaker_key(dataset, t, f),
                 gender=t["speaker"] if dataset == "iemocap" else None) for (d, t), f in zip(turns, feats)]


# ----------------------------------------------------------------------------- thresholds -> categories -> text
NAMES = {3: ["low", "medium", "high"], 5: ["very_low", "low", "medium", "high", "very_high"]}


def quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    x = (len(sorted_vals) - 1) * q / 100.0
    lo = int(math.floor(x)); hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (x - lo)


class ClueConfig:
    """percentiles: cut points (3 classes -> 2 cuts, 5 classes -> 4 cuts); features;
    norm: 'global'      percentiles of the raw values over the whole training set (original EmoryNLP clues)
          'speaker_pct' percentiles of the raw values per speaker, global cuts for speakers with < min_speaker_n turns
                        (original MELD / SpeechCueLLM clues)
          'speaker'     per-speaker z-scores, then global percentiles of the z-scores"""
    def __init__(self, percentiles=(10, 25, 75, 90), features=FEATS5, norm="global", min_speaker_n=30, speaker_field="speaker"):
        assert len(percentiles) in (2, 4)
        self.percentiles, self.features, self.norm, self.min_speaker_n = tuple(percentiles), tuple(features), norm, min_speaker_n
        self.speaker_field = speaker_field  # which row field identifies the speaker for the per-speaker options
        self.names = NAMES[len(percentiles) + 1]

    def __repr__(self):
        return f"p{'-'.join(map(str, self.percentiles))}|{len(self.features)}f|{self.norm}|{self.speaker_field}"


def fit(cfg, rows):
    """Fit cut points on training rows only (zero / missing values ignored, like the original script)."""
    stats = {"speaker": {}}
    def vals(sub, f):
        return sorted(x for r in sub if r["feats"] and (x := r["feats"].get(f)) not in (None, 0))
    if cfg.norm == "speaker":
        by = defaultdict(list)
        for r in rows:
            by[r[cfg.speaker_field]].append(r)
        for s, sub in by.items():
            if len(sub) >= cfg.min_speaker_n:
                stats["speaker"][s] = {f: (sum(v) / len(v), (sum((x - sum(v) / len(v)) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5 or 1.0)
                                       for f in cfg.features if (v := vals(sub, f))}
    if cfg.norm == "speaker_pct":
        by = defaultdict(list)
        for r in rows:
            by[r[cfg.speaker_field]].append(r)
        stats["speaker_cuts"] = {s_: {f: [quantile(v, p) for p in cfg.percentiles] for f in cfg.features if (v := vals(sub, f))}
                                 for s_, sub in by.items() if len(sub) >= cfg.min_speaker_n}
    glob = {f: vals(rows, f) for f in cfg.features}
    stats["global_cuts"] = {f: [quantile(glob[f], p) for p in cfg.percentiles] for f in cfg.features}
    if cfg.norm == "speaker":   # cut points are then percentiles of the per-speaker z-scores
        z = {f: [] for f in cfg.features}
        for r in rows:
            for f in cfg.features:
                x = _value(cfg, stats, r, f)
                if x is not None:
                    z[f].append(x)
        stats["z_cuts"] = {f: [quantile(sorted(z[f]), p) for p in cfg.percentiles] for f in cfg.features}
    return stats


def _value(cfg, stats, row, f):
    x = row["feats"].get(f) if row["feats"] else None
    if x in (None, 0):
        return None
    if cfg.norm == "speaker":
        s = stats["speaker"].get(row[cfg.speaker_field], {}).get(f)
        if s:
            return (x - s[0]) / s[1]
        # speaker too rare: fall back to global z
        return None if "z_fallback" not in stats else x
    return x


def categorize(cfg, stats, row):
    """-> {feature: category name or 'none'}"""
    out = {}
    for f in cfg.features:
        x = _value(cfg, stats, row, f)
        if x is None:
            out[f] = "none"; continue
        if cfg.norm == "speaker":
            cuts = stats["z_cuts"][f]
        elif cfg.norm == "speaker_pct":
            cuts = stats["speaker_cuts"].get(row[cfg.speaker_field], {}).get(f) or stats["global_cuts"][f]
        else:
            cuts = stats["global_cuts"][f]
        k = sum(x > c for c in cuts)       # '<= cut' belongs to the lower class, as in the original script
        out[f] = cfg.names[k]
    return out


_LEVEL = {"very_low": ("high", "very low"), "low": ("medium", "low"), "medium": ("low", "medium"),
          "high": ("medium", "high"), "very_high": ("high", "very high")}
_CERT = {"high": "", "medium": "likely ", "low": "may "}
NO_AUDIO = "No distinct acoustic variation observed in transcript."
# (low phrase, high phrase) for the three features beyond the original five; medium values are not mentioned
_EXTRA_PHRASES = {"duration": ("short duration", "long duration"),
                  "pitch_range": ("narrow pitch range", "wide pitch range"),
                  "mean_hnr": ("rough or breathy voice quality", "clear voice quality")}


def impression(cats):
    """Verbalise like postprocess_emorynlp.generate_impression (pitch, pitch variation, volume, volume variation, rate)."""
    if cats is None or all(v == "none" for v in cats.values()):
        return NO_AUDIO
    lv = lambda f: _LEVEL.get(cats.get(f, "medium"), ("low", "medium")) if cats.get(f, "none") != "none" else ("low", "medium")
    pc, pitch = lv("avg_pitch"); vc, pvar = lv("pitch_std"); ic, vol = lv("avg_intensity")
    ivc, vvar = lv("intensity_variation"); rc, rate = lv("articulation_rate")
    c = lambda k: _CERT[k]
    if pitch in ("high", "very high"): p = f"{c(pc)}uses a higher pitch"
    elif pitch in ("low", "very low"): p = f"{c(pc)}uses a lower pitch"
    else: p = "has a moderate pitch"
    if pvar in ("high", "very high"): p += f" with {c(vc)}noticeable variation, suggesting expressiveness"
    elif pvar in ("low", "very low"): p += f" that {c(vc)}remains steady, potentially indicating calmness or seriousness"
    else: p += " with typical variation"
    if vol in ("high", "very high"): v = f"{c(ic)}speaking loudly, which might indicate excitement, confidence, or urgency"
    elif vol in ("low", "very low"): v = f"{c(ic)}speaking softly, possibly suggesting calmness, shyness, or caution"
    else: v = "using a moderate volume"
    if vvar in ("high", "very high"): v += f", with {c(ivc)}significant volume changes"
    elif vvar in ("low", "very low"): v += f", with {c(ivc)}little volume variation"
    else: v += ", with normal volume variation"
    if rate in ("high", "very high"): r = f"{c(rc)}talking quickly, which could indicate excitement, urgency, or nervousness"
    elif rate in ("low", "very low"): r = f"{c(rc)}talking slowly, possibly suggesting thoughtfulness, hesitation, or calmness"
    else: r = "speaking at a moderate pace"
    text = f"The target speaker {p}, while {v}, and is {r}."
    # the original template only covers 5 features; configs that also categorise duration / pitch range / HNR get one extra
    # sentence (otherwise an '8feat' config would produce exactly the same text as its '5feat' twin)
    extra = []
    for feat, (lo, hi) in _EXTRA_PHRASES.items():
        c = cats.get(feat, "none")
        if c in ("very_low", "low"):
            extra.append("a " + ("very " if c == "very_low" else "") + lo)
        elif c in ("high", "very_high"):
            extra.append("a " + ("very " if c == "very_high" else "") + hi)
    if extra:
        text += " It has " + (", ".join(extra[:-1]) + " and " + extra[-1] if len(extra) > 1 else extra[0]) + "."
    return text


# Default clue settings per dataset = how the original clues were made (see module docstring):
#   EmoryNLP global percentiles, MELD per-speaker percentiles (>=100 turns, else global), IEMOCAP per-gender (best reproduction, ~54 %).
DEFAULTS = {
    "emorynlp": dict(norm="global"),
    "meld": dict(norm="speaker_pct", min_speaker_n=100),
    "iemocap": dict(norm="speaker_pct", min_speaker_n=100, speaker_field="gender"),
}


def default_config(dataset, **override):
    kw = dict(DEFAULTS[dataset]); kw.update(override)
    return ClueConfig(**kw)


_NAME = re.compile(r"^(5c|3c) p([\d-]+)(?: \(original\))? \| (global|speaker_pct|speaker) \| (5feat|8feat)$")
_SLUG = re.compile(r"^(5c|3c)-p((?:\d+-)*\d+)-(speaker-pct|speaker|global)-(5feat|8feat)$")


def parse_config(name, dataset, min_speaker_n=100):
    """'5c p10-30-70-90 | speaker_pct | 8feat' (as in the screen CSV) or its slug '5c-p10-30-70-90-speaker-pct-8feat'
    -> (ClueConfig, canonical slug). IEMOCAP per-speaker options use gender, as in its default."""
    n = name.strip()
    m = _NAME.match(n) or _SLUG.match(n)
    if not m:
        raise ValueError(f"cannot parse clue config {name!r}; expected e.g. '5c p10-30-70-90 | speaker_pct | 8feat'")
    cuts = tuple(int(x) for x in m.group(2).split("-"))
    norm = m.group(3).replace("-", "_")
    feats = FEATS5 if m.group(4) == "5feat" else FEATS8
    kw = dict(percentiles=cuts, features=feats, norm=norm, min_speaker_n=min_speaker_n)
    if dataset == "iemocap":
        kw["speaker_field"] = "gender"
    slug = f"{m.group(1)}-p{m.group(2)}-{norm.replace('_', '-')}-{m.group(4)}"
    return ClueConfig(**kw), slug


def clue_texts(cfg, fit_rows, rows):
    """Impression text for every row, thresholds fitted on fit_rows (train) only."""
    st = fit(cfg, fit_rows)
    return [impression(categorize(cfg, st, r)) if r["feats"] is not None else NO_AUDIO for r in rows]


# ----------------------------------------------------------------------------- SpeechCueLLM-style 'description'
# Mirrors sp/SpeechCueLLM/feature_utils/postprocess_audio_feature_meld.generate_concise_description (5-class terms; 3-class names
# fall into the same table). The paper's headline setting is "text + audio description": compact, factual, no emotion words.
_TERM = {"very_low": "very low", "low": "low", "medium": "moderate", "high": "high", "very_high": "very high"}
MIN_DURATION = 0.5   # SpeechCueLLM adds no speech text for clips of <= 0.5 s (MELD)


def description(cats):
    """'Target speech characteristics: very high volume with moderate variation, very low pitch with very low variation, ...'"""
    g = lambda f: cats.get(f, "none")
    parts = []
    if g("avg_intensity") != "none":
        v = g("intensity_variation")
        parts.append(f"{_TERM[g('avg_intensity')]} volume" + (f" with {_TERM[v]} variation" if v != "none" else ""))
    if g("avg_pitch") != "none":
        pv = g("pitch_std") if g("pitch_std") != "none" else g("pitch_range")
        parts.append(f"{_TERM[g('avg_pitch')]} pitch" + (f" with {_TERM[pv]} variation" if pv != "none" else ""))
    if g("articulation_rate") != "none":
        parts.append(f"{_TERM[g('articulation_rate')]} speaking rate")
    return "Target speech characteristics: " + ", ".join(parts) + "." if parts else "Insufficient data to describe speech characteristics."


def context_phrase(cats):
    """Short tag appended to context utterances (SpeechCueLLM 'audio_context'): '(high pitch with low variation)'."""
    if cats.get("avg_pitch", "none") == "none":
        return None
    pv = cats.get("pitch_std", "none")
    return f"({_TERM[cats['avg_pitch']]} pitch" + (f" with {_TERM[pv]} variation)" if pv != "none" else ")")


FORMS = ("impression", "description", "both")


def clue_bundle(cfg, fit_rows, rows, form="impression", min_duration=None):
    """Per row: (clue text or None, context tag or None). form: impression (our original), description (SpeechCueLLM headline),
    both (description then impression, as the SpeechCueLLM flags allow). Rows without usable audio get no text for the
    description forms; the impression form keeps the fixed 'no acoustic variation' sentence like before."""
    st = fit(cfg, fit_rows)
    out = []
    for r in rows:
        f = r["feats"]
        if f is None or (min_duration is not None and (f.get("duration") or 0) <= min_duration):
            out.append((NO_AUDIO if form == "impression" else None, None))
            continue
        cats = categorize(cfg, st, r)
        text = {"impression": impression, "description": description,
                "both": lambda c: description(c) + " " + impression(c)}[form](cats)
        out.append((text, context_phrase(cats)))
    return out
