"""Shared helpers for building CoE (Clue of Emotion) data: per-dataset config,
personas, dialogue reconstruction from the flat ERC rows, and the scene cache."""
import json
import os
import re

HERE = os.path.dirname(os.path.realpath(__file__))
# ERC-only rows of the standard EmoryNLP / MELD / IEMOCAP splits (copied from the baseline pipeline's data)
BASE_DATA = os.path.join(HERE, "source_data")
DATA_DIR = os.path.join(HERE, "Truebaseline", "data")

# Paper Appendix E (speaker-recognition prompt), keyed by the full character names.
PERSONAS = {
    "Chandler Bing": "Known for his wit and sarcasm, works in advertising, group's humorist with logical insights.",
    "Ross Geller": "A paleontologist and professor, adds intellectual depth through family ties with Monica and past relationship with Rachel.",
    "Joey Tribbiani": "An aspiring actor, the group's innocent and charming member, deepening bond with Chandler through loyalty and simplicity.",
    "Phoebe Buffay": "A masseuse and musician, introduces eccentricity and unique wisdom, offering empathy to the group.",
    "Monica Geller": "A chef, known for organization, group's anchor with hospitality, directly linking to Ross as his sister.",
    "Rachel Green": "Progresses from a waitress to fashion, influencing group dynamics, especially her evolving relationship with Ross.",
}
FULL = list(PERSONAS)
SHORT = {n.split()[0]: n for n in FULL}  # MELD uses first names only

DATASETS = {
    "emorynlp": dict(
        title="EmoryNLP",
        labels=['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared'],
        main=FULL, persona={n: PERSONAS[n] for n in FULL}, scene=True, speaker_task=True),
    "meld": dict(
        title="MELD",
        labels=['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'],
        main=list(SHORT), persona={s: PERSONAS[f] for s, f in SHORT.items()}, scene=True, speaker_task=True),
    # IEMOCAP: two anonymous speakers (F/M), no personas, no scene (paper notes both are absent).
    # Speaker-ID is skipped: with F/M alternating it only teaches turn-taking, and there is no persona to link to.
    "iemocap": dict(
        title="IEMOCAP",
        labels=['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
        main=["F", "M"], persona={}, scene=False, speaker_task=False),
}


def cfg(dataset):
    return DATASETS[dataset]


def label_str(dataset):
    return ", ".join(f"'{l}'" for l in cfg(dataset)["labels"])


_ROW = re.compile(
    r"### Dialogue History:\n(.*?)### Current Utterance:\n(.*?)\n\nBased on the context", re.S)


def split_speaker(line):
    name, _, text = line.partition(":")
    return name.strip(), text.strip()


def load_dialogues(split, dataset="emorynlp"):
    """Group the flat ERC rows into dialogues (an empty history starts a new one).
    Each turn keeps its own original history text: for IEMOCAP the history contains
    unlabelled utterances that have no row, so it cannot be rebuilt from the rows."""
    rows = []
    with open(os.path.join(BASE_DATA, dataset, f"{split}.json"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "what is the emotion of the current" in r["input"]:
                rows.append(r)
    dialogues = []
    for r in rows:
        hist, cur = _ROW.search(r["input"]).groups()
        if hist == "":
            dialogues.append({"id": f"{split}:{len(dialogues)}", "turns": []})
        spk, text = split_speaker(cur.strip())
        dialogues[-1]["turns"].append({
            "speaker": spk, "text": text, "line": cur.strip(),
            "emotion": r["target"].strip(), "history": hist.rstrip("\n")})
    return dialogues


def scenes_path(dataset):
    return os.path.join(DATA_DIR, f"{dataset}_scenes.json")


def load_scenes(dataset="emorynlp"):
    p = scenes_path(dataset)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def persona_block(dataset="emorynlp", names=None):
    pers = cfg(dataset)["persona"]
    names = names or list(pers)
    return "\n".join(f"{n}: {pers[n]}" for n in names)


# IEMOCAP scene clue (opt-in variant, see build_coe_data.py --iemocap_scene). Setting / premise only:
# deliberately no target emotions, percentages or mood words from the dataset documentation.
IEMOCAP_SCENES = {
    "impro01": "A government office: a person who has waited in line to get an ID speaks with a clerk who says the paperwork or the line is wrong.",
    "impro02": "At home: one partner tells the other they have received orders for a year-long military deployment, starting tomorrow.",
    "impro03": "Two friends talk: one has just got engaged or married (in Las Vegas) and shows off the ring.",
    "impro04": "Two people talk about a long job search: one has been unemployed for three years, the other suggests job boards and new options.",
    "impro05": "An airport baggage claim: a traveler's luggage has not arrived and a baggage agent offers a complaint form and a small reimbursement.",
    "impro06": "Two friends talk after the death of a young mutual friend, Johnny.",
    "impro07": "Two friends talk: one has just opened a big envelope with an acceptance letter from a university.",
    "impro08": "A customer phones a company's automated phone menu and then an agent to sort out a credit card or service problem.",
    "script01_1": "From the play 'All My Sons': early morning in the backyard, a storm has snapped the apple tree planted for the missing son Larry.",
    "script01_2": "From the play 'All My Sons': a mother asks why Ann Deever, her missing son Larry's former fiancee, was invited to visit.",
    "script01_3": "From the play 'All My Sons': Chris and Ann talk privately in the garden about the past, the family and their future.",
    "script02_1": "From the play 'Grunions': a couple wait on a California beach at night for the grunion fish to come ashore.",
    "script02_2": "From the play 'Grunions': on the beach at night, he reads meaning into the wind and their relationship while she brushes it off.",
    "script03_1": "From the play 'Private Lives': divorced lovers who ran off together hide in a Paris flat from their new spouses; a phone call startles them.",
    "script03_2": "From the play 'Private Lives': over cocktails, two divorced lovers talk about past relationships and a diamond brooch.",
}


def load_iemocap_turn_scenes():
    """{dialogue id: [scene id per turn]} from align_iemocap_scenes.py."""
    p = os.path.join(DATA_DIR, "iemocap_turn_scenes.json")
    if not os.path.exists(p):
        raise SystemExit("missing iemocap_turn_scenes.json; run align_iemocap_scenes.py --transcripts <dir> first")
    with open(p) as f:
        return json.load(f)
