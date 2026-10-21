"""Relationship knowledge for the 'Character Relationships' clue.

Friends datasets (EmoryNLP, MELD): a who's-who map. For every recognised side character in the dialogue window the clue states who they
are *by name* ("Carol Willick is Ross Geller's ex-wife."), plus the sibling tie among the main cast when both siblings are present.
Windows with no recognised character get no clue at all (the earlier constant "a group of close friends" carried no information).

IEMOCAP: one relationship sentence per scene (needs iemocap_turn_scenes.json, i.e. the same transcript alignment as the `_scene`
variant). Wording is deliberately emotion-free: the earlier version said "a frustrated customer" / "an angry customer", which
gives the label away.

NOTE: the Friends facts below are the curated list from the earlier base+chain+acous+rel experiment (names added so the clue says
WHO is meant), plus a few entries added when the show facts are certain. Review/extend FRIENDS_KB before relying on it: a wrong fact
becomes a wrong training prompt. Ambiguous first names (Paul, Julie, Mark, Bonnie, ...) are intentionally left out.
"""
import re

# display name -> (aliases, relation phrase). Aliases are matched on the normalised speaker name (lower case, no punctuation/honorific).
FRIENDS_KB = {
    "Carol Willick": (["carol willick", "carol"], "Ross Geller's ex-wife"),
    "Susan Bunch": (["susan bunch", "susan"], "Carol Willick's partner"),
    "Barry Farber": (["barry farber", "barry"], "Rachel Green's ex-fiance"),
    "Richard Burke": (["richard burke", "richard"], "Monica Geller's older ex-boyfriend"),
    "Janice Litman Goralnik": (["janice litman goralnik", "janice"], "Chandler Bing's on-and-off girlfriend"),
    "Emily Waltham": (["emily waltham", "emily"], "Ross Geller's British ex-wife"),
    "Stephen Waltham": (["stephen waltham"], "Emily Waltham's father"),
    "Andrea Waltham": (["andrea waltham"], "Emily Waltham's mother"),
    "Judy Geller": (["judy geller", "mrs geller"], "the mother of Ross Geller and Monica Geller"),
    "Jack Geller": (["jack geller", "mr geller"], "the father of Ross Geller and Monica Geller"),
    "Frank Buffay Jr.": (["frank buffay jr", "frank buffay", "frank"], "Phoebe Buffay's half-brother"),
    "Ursula Buffay": (["ursula buffay", "ursula"], "Phoebe Buffay's twin sister"),
    "Mike Hannigan": (["mike hannigan", "mike"], "Phoebe Buffay's romantic partner"),
    "David": (["david"], "Phoebe Buffay's scientist boyfriend"),
    "Gunther": (["gunther"], "the manager of the Central Perk coffee shop"),
    "Mr. Treeger": (["mr treeger", "treeger"], "the building superintendent"),
    "Mr. Heckles": (["mr heckles", "heckles"], "the downstairs neighbor"),
    "Estelle Leonard": (["estelle leonard", "estelle"], "Joey Tribbiani's talent agent"),
    "Mr. Tribbiani": (["mr tribbiani"], "Joey Tribbiani's father"),
    "Mona": (["mona"], "Ross Geller's girlfriend"),
    "Pete Becker": (["pete becker", "peter becker", "pete"], "Monica Geller's millionaire boyfriend"),
    "Tag Jones": (["tag jones", "tag"], "Rachel Green's assistant and boyfriend"),
    "Joshua Burgin": (["joshua burgin", "joshua"], "Rachel Green's client and boyfriend"),
    "Dr. Green": (["dr green"], "Rachel Green's father"),
    "Amy Green": (["amy green", "amy"], "Rachel Green's sister"),
    "Jill Green": (["jill green", "jill"], "Rachel Green's sister"),
    "Charlie Wheeler": (["charlie wheeler", "charlie"], "a professor whom both Ross Geller and Joey Tribbiani dated"),
    "Ben Geller": (["ben geller", "ben"], "Ross Geller's son"),
    "Eddie Menuek": (["eddie menuek", "eddie"], "Chandler Bing's roommate"),
    "Kathy": (["kathy"], "Chandler Bing's girlfriend"),
}
_ALIAS = {a: d for d, (al, _) in FRIENDS_KB.items() for a in al}
MAIN_CAST = {"ross": "Ross Geller", "monica": "Monica Geller", "chandler": "Chandler Bing",
             "joey": "Joey Tribbiani", "rachel": "Rachel Green", "phoebe": "Phoebe Buffay"}
SIBLINGS = ("Ross Geller", "Monica Geller")

# IEMOCAP: scene id -> relationship between the two speakers (no emotion words)
IEMOCAP_REL = {
    "impro01": "The speakers are strangers: a person at a government office and the clerk serving them.",
    "impro02": "The speakers are partners; one of them is about to be deployed.",
    "impro03": "The speakers are close friends.",
    "impro04": "The speakers are close: one has been unemployed for three years and the other is trying to help.",
    "impro05": "The speakers are strangers: a traveler and an airline baggage agent.",
    "impro06": "The speakers are close friends of the person who has died.",
    "impro07": "The speakers are close friends.",
    "impro08": "The speakers are strangers: a customer and a customer-service agent.",
    "script01_1": "The speakers are family members: parents and their adult son (from the play 'All My Sons').",
    "script01_2": "The speakers are family members (from the play 'All My Sons').",
    "script01_3": "The speakers are Chris and Ann, old friends with a romantic connection (from the play 'All My Sons').",
    "script02_1": "The speakers are a couple, Augie and Carla (from the play 'Grunions').",
    "script02_2": "The speakers are a couple, Augie and Carla (from the play 'Grunions').",
    "script03_1": "The speakers are divorced spouses, Elyot and Amanda, who have run away together (from the play 'Private Lives').",
    "script03_2": "The speakers are divorced spouses, Elyot and Amanda (from the play 'Private Lives').",
}


def _norm(name):
    n = re.sub(r"[#.]", "", name.lower())
    return re.sub(r"\s+", " ", n).strip()


def recognise(speaker):
    """speaker name -> ('side', display) | ('main', display) | None"""
    n = _norm(speaker)
    if n in _ALIAS:
        return "side", _ALIAS[n]
    first = n.split(" ")[0] if n else ""
    if first in MAIN_CAST and n.split(" ")[-1] in (first, MAIN_CAST[first].split()[-1].lower()):
        return "main", MAIN_CAST[first]
    return None


def friends_block(speakers):
    """Relationship sentences for the characters in `speakers` (window + current speaker), or '' if nothing is known."""
    seen, side, main = set(), [], set()
    for s in speakers:
        r = recognise(s)
        if not r or r[1] in seen:
            continue
        seen.add(r[1])
        if r[0] == "side":
            side.append(r[1])
        else:
            main.add(r[1])
    lines = [f"{d} is {FRIENDS_KB[d][1]}." for d in side]
    if all(x in main for x in SIBLINGS):
        lines.append(f"{SIBLINGS[0]} and {SIBLINGS[1]} are siblings.")
    return " ".join(lines)


def iemocap_block(scene_id):
    return IEMOCAP_REL.get(scene_id, "")
