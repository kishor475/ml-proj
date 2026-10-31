"""Chain-of-Emotion data transforms.

The model must WRITE a chain of emotions for the dialogue history (supervised with the gold labels of those turns) and then
the emotion of the current utterance. The input never contains gold emotions, so nothing leaks at test time; the chain the
model generates at test time is its own. The final answer is always on the last line as 'Current Emotion: <label>', which
the existing trainer/evaluator already parses.

variants
  self  emotions of the CURRENT SPEAKER's previous utterances in the history window      (the earlier base+chain design)
  all   emotion of EVERY utterance in the history window, speaker-tagged, oldest first
  prev  emotion of the previous utterance only (chain of length 1)
"""
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "base"))
from coe_clues import split_speaker  # noqa: E402

VARIANTS = ("self", "all", "prev", "all_known", "lf_all", "lf_prev")
# all_known: like 'all' but history lines WITHOUT a gold emotion (IEMOCAP: ~23 %) are skipped instead of written as '?'
#            (those targets are unpredictable and were the likely reason 'all' scored worst on IEMOCAP)
# lf_*     : LABEL FIRST - "Current Emotion: X" then the chain as auxiliary supervision. The label no longer depends on a
#            self-generated chain (no error propagation), but training still sees the chain targets (multi-task effect).
# generation budget needed at evaluation (tokens): chain + label
MAX_NEW_TOKENS = {"prev": 32, "self": 64, "all": 128, "all_known": 128, "lf_all": 24, "lf_prev": 24}   # lf_*: only the first line is scored

OLD_INSTRUCTION = "Answer with just the emotion label."
NEW_INSTRUCTION = {
    "self": ("First list the emotions of this speaker's previous utterances in the dialogue history (oldest first, "
             "'None' if there are none), then give the emotion of the current utterance.\n"
             "Answer in the format:\nEmotion Flow: <emotion> -> <emotion> -> ...\nCurrent Emotion: <label>"),
    "all": ("First list the emotion of each utterance in the dialogue history (oldest first, '?' if unknown), then give "
            "the emotion of the current utterance.\n"
            "Answer in the format:\nEmotion Flow: <speaker>: <emotion> -> <speaker>: <emotion> -> ...\nCurrent Emotion: <label>"),
    "all_known": ("First list the emotion of each utterance in the dialogue history that has a clear emotion (oldest first, 'None' if there "
                  "are none), then give the emotion of the current utterance.\n"
                  "Answer in the format:\nEmotion Flow: <speaker>: <emotion> -> <speaker>: <emotion> -> ...\nCurrent Emotion: <label>"),
    "lf_all": ("First give the emotion of the current utterance, then list the emotion of each earlier utterance in the dialogue history that "
               "has a clear emotion (oldest first).\n"
               "Answer in the format:\nCurrent Emotion: <label>\nEmotion Flow: <speaker>: <emotion> -> <speaker>: <emotion> -> ..."),
    "lf_prev": ("First give the emotion of the current utterance, then the emotion of the previous utterance ('none' if there is none, "
                "'unknown' if unclear).\nAnswer in the format:\nCurrent Emotion: <label>\nPrevious Emotion: <emotion>"),
    "prev": ("First give the emotion of the previous utterance ('none' if there is none, 'unknown' if unclear), then the emotion of the current "
             "utterance.\nAnswer in the format:\nPrevious Emotion: <emotion>\nCurrent Emotion: <label>"),
}


def gold_history(d, i, max_back=4):
    """[(speaker, emotion|None)] for each line of turn i's history window, oldest first.
    EmoryNLP/MELD histories are exactly the previous turns; IEMOCAP histories also contain unlabelled utterances that
    are not turns of ours (emotion None)."""
    turns = d["turns"]
    lines = [l for l in turns[i]["history"].split("\n") if l.strip()]
    out, j = [], i - 1
    for line in reversed(lines):
        spk = split_speaker(line)[0]
        hit = None
        for k in range(j, max(j - max_back, -1), -1):
            if turns[k]["line"] == line:
                hit = k; break
        if hit is None:
            out.append((spk, None))
        else:
            out.append((turns[hit]["speaker"], turns[hit]["emotion"])); j = hit - 1
    return out[::-1]


def chain_target(variant, speaker, hist, label):
    if variant == "self":
        flow = [e for s, e in hist if s == speaker and e]
        return f"Emotion Flow: {' -> '.join(flow) if flow else 'None'}\nCurrent Emotion: {label}"
    if variant == "all":
        flow = [f"{s}: {e or '?'}" for s, e in hist]
        return f"Emotion Flow: {' -> '.join(flow) if flow else 'None'}\nCurrent Emotion: {label}"
    if variant in ("prev", "lf_prev"):
        prev = "none" if not hist else (hist[-1][1] or "unknown")   # none = no previous turn, unknown = previous turn unlabelled
        if variant == "lf_prev":
            return f"Current Emotion: {label}\nPrevious Emotion: {prev}"
        return f"Previous Emotion: {prev}\nCurrent Emotion: {label}"
    if variant in ("all_known", "lf_all"):
        flow = [f"{s}: {e}" for s, e in hist if e]
        body = f"Emotion Flow: {' -> '.join(flow) if flow else 'None'}"
        return f"Current Emotion: {label}\n{body}" if variant == "lf_all" else f"{body}\nCurrent Emotion: {label}"
    raise ValueError(variant)


def chain_prompt(prompt, variant):
    assert prompt.rstrip().endswith(OLD_INSTRUCTION), "unexpected ERC prompt ending"
    return prompt.rstrip()[: -len(OLD_INSTRUCTION)] + NEW_INSTRUCTION[variant]


def split_chain(text):
    """-> (chain part, final label) of a (generated or gold) chain answer. Handles both orders:
    chain first ('Emotion Flow: ...\nCurrent Emotion: X') and label first ('Current Emotion: X\nEmotion Flow: ...')."""
    s = text.lstrip()
    if s.startswith("Current Emotion:"):
        first, _, rest = s[len("Current Emotion:"):].partition("\n")
        return rest.strip(), first.strip()
    head, sep, tail = text.rpartition("Current Emotion:")
    return (head.strip(), tail.strip()) if sep else ("", text.strip())
