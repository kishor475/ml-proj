"""Character relationship map built by hand (files in this folder) -> 'Character Relationships' lines for a dialogue window.

Files per dataset (names are spelled as they are on disk):
  *_chars.json            the kept side characters (waiters, extras ... were discarded)
  *_char_relations.json   per character: core_friend (primary), relation (category), detail, description, all_core_relations
  *_relation_tuples.json  {"(Character, CoreFriend)": category} - a cross-check only: it lacks some pairs that all_core_relations has
Source of truth = all_core_relations. 'core_group' rows (the main cast about each other, with late-series facts such as "Husband to Monica")
are skipped by default because they would be anachronistic in early-season dialogues.

Styles (what each line says about a character):
  detail    "Alan - relation to Monica Geller: Boyfriend (adored by all six friends)"        (default)
  category  "Alan - relation to Monica Geller: romantic connection"
  desc      detail + the hand-written description sentence
"""
import json
import os
import re

HERE = os.path.dirname(os.path.realpath(__file__))
FILES = {"emorynlp": dict(chars="emory_chars.json", rel="emory_char_realtions.json", tuples="emory_relation_tuples.json"),
         "meld": dict(chars="meld_chars.json", rel="meld_char_relations.json", tuples="meld_realtion_tuples.json")}
CATEGORY_PHRASE = {"family": "family member", "romantic": "romantic connection", "friend": "friend", "colleague": "colleague",
                   "service_neighbor": "acquaintance (service provider or neighbor)", "celebrity_guest": "celebrity guest",
                   "alter_ego": "acting role / alter ego", "core_group": "member of the core friend group"}
STYLES = ("detail", "category", "desc")


def _load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


class RelMap:
    def __init__(self, dataset, include_core_group=False):
        fl = FILES[dataset]
        self.kept = set(_load(fl["chars"]))
        self.entries = {}
        for e in _load(fl["rel"]):
            if e["character"] not in self.kept or (e["relation"] == "core_group" and not include_core_group):
                continue
            rels = [(c["core_friend"], c["relation"], c["detail"]) for c in e["all_core_relations"]
                    if include_core_group or c["relation"] != "core_group"]
            primary = next((r for r in rels if r[0] == e["core_friend"]), rels[0] if rels else None)
            if primary:
                self.entries[e["character"]] = dict(primary=primary, rels=rels, desc=e["description"])
        self.tuples = {}
        for k, v in _load(fl["tuples"])["tuples"].items():
            m = re.match(r"\((.*), ([^,]*)\)$", k)
            self.tuples[(m.group(1), m.group(2))] = v

    def check(self):
        """Cross-check of all_core_relations against the tuple file: (#pairs, #only in relations, #category disagreements)."""
        pairs = {(c, f): r for c, e in self.entries.items() for f, r, _ in e["rels"]}
        only = [k for k in pairs if k not in self.tuples]
        dis = [k for k in pairs if k in self.tuples and self.tuples[k] != pairs[k]]
        return len(pairs), len(only), len(dis)

    def lines(self, speakers, style="detail", max_lines=4, max_per_char=2):
        """speakers: names in the window, MOST RELEVANT FIRST (current speaker, then history speakers newest first)."""
        present = set(speakers)
        out = []
        for s in dict.fromkeys(speakers):
            e = self.entries.get(s)
            if not e:
                continue
            use = [r for r in e["rels"] if r[0] in present] or [e["primary"]]      # relations to people in the scene first
            for k, (core, cat, detail) in enumerate(use[:max_per_char]):
                text = CATEGORY_PHRASE.get(cat, cat) if style == "category" else detail
                line = f"{s} - relation to {core}: {text}"
                if style == "desc" and k == 0:           # the description is about the character: once, not under every line
                    line += f". {e['desc']}"
                out.append(line + ("" if line.endswith(".") else "."))
                if len(out) >= max_lines:
                    return out
        return out
