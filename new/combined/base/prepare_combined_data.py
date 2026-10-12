#!/usr/bin/env python3
"""
==============================================================================
PREPARE COMBINED / STEeR DATASET FOR EMORYNLP
==============================================================================
Generates:
  1. trainbegin.json : 100 balanced seed examples with complete emotional
                       reasoning rationales matching paper Appendix E.
  2. train4infer.json: Remaining training turns with STEeR prompts and gold labels.
  3. train.json      : Mixed multi-task dataset (Strategy B) combining:
                       - ERC task
                       - Speaker Identification task
                       - Role-Play task
                       - STEeR Emotion Reasoning task
==============================================================================
"""

import json
import os
import re

def parse_speaker_and_utterance(curr_utt_text):
    if ":" in curr_utt_text:
        parts = curr_utt_text.split(":", 1)
        return parts[0].strip(), parts[1].strip()
    return "Speaker", curr_utt_text.strip()

def build_rationale(speaker, utterance, history, emotion):
    hist_snippet = history[-150:].replace("\n", " ") if history else "the opening of the conversation"
    
    rationale_templates = {
        "Joyful": (
            f"Following {hist_snippet}, {speaker} engages with an upbeat, enthusiastic, or lighthearted tone. "
            f"The utterance '{utterance[:60]}' expresses positive humor, amusement, or pleasure with the situation."
        ),
        "Mad": (
            f"In the context of {hist_snippet}, {speaker} exhibits irritation, indignation, or frustration. "
            f"The statement '{utterance[:60]}' conveys clear displeasure, objection, or argumentative tension toward the listener."
        ),
        "Peaceful": (
            f"Given the preceding context of {hist_snippet}, {speaker} maintains a calm, supportive, and soothing disposition. "
            f"The statement '{utterance[:60]}' aims to provide comfort, gentle agreement, or de-escalate tension."
        ),
        "Neutral": (
            f"In response to {hist_snippet}, {speaker} offers a straightforward, matter-of-fact remark without strong affective bias. "
            f"The line '{utterance[:60]}' serves an informational or conversational turn-taking function without prominent emotional charge."
        ),
        "Sad": (
            f"Within the dialogue surrounding {hist_snippet}, {speaker} reveals disappointment, vulnerability, or emotional regret. "
            f"The phrasing in '{utterance[:60]}' highlights feelings of dejection, melancholy, or longing."
        ),
        "Powerful": (
            f"Building upon {hist_snippet}, {speaker} takes an assertive, dominant, or self-assured stance. "
            f"The expression '{utterance[:60]}' establishes personal agency, firm conviction, and commanding presence in the interaction."
        ),
        "Scared": (
            f"In light of {hist_snippet}, {speaker} expresses unease, nervousness, anxiety, or apprehension about the unfolding events. "
            f"The utterance '{utterance[:60]}' reveals fear, hesitation, or frantic defense under social pressure."
        )
    }

    explanation = rationale_templates.get(emotion, f"Based on {hist_snippet}, {speaker} conveys {emotion.lower()} feelings.")
    target = f"[Rationale]\n{explanation} The emotional label of <{speaker}: \"{utterance}\"> is <{emotion}>."
    return target

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_data_dir = os.path.join(base_dir, "..", "..", "base", "Truebaseline", "data", "emorynlp")
    dst_data_dir = os.path.join(base_dir, "Truebaseline", "data", "emorynlp")
    os.makedirs(dst_data_dir, exist_ok=True)

    src_train = os.path.join(src_data_dir, "train.json")
    print(f"Reading base training data from {src_train}...")
    with open(src_train, 'r', encoding='utf-8') as f:
        all_lines = [json.loads(line) for line in f]

    # Filter ERC turns
    erc_samples = []
    other_samples = []
    for item in all_lines:
        inp = item['input']
        if "Based on the context, what is the emotion of the current utterance?" in inp:
            erc_samples.append(item)
        else:
            other_samples.append(item)

    print(f"Total ERC samples: {len(erc_samples)}, Total auxiliary samples: {len(other_samples)}")

    # Target quotas for 100 seed examples across 7 emotions
    target_quotas = {
        "Joyful": 15,
        "Mad": 15,
        "Peaceful": 14,
        "Neutral": 15,
        "Sad": 14,
        "Powerful": 14,
        "Scared": 13
    }
    collected_seed = {k: [] for k in target_quotas}
    remaining_erc = []

    for item in erc_samples:
        gold = item['target'].strip()
        if gold in collected_seed and len(collected_seed[gold]) < target_quotas[gold]:
            collected_seed[gold].append(item)
        else:
            remaining_erc.append(item)

    # 1. Build trainbegin.json (100 seed rationales)
    seed_records = []
    for emotion, items in collected_seed.items():
        for it in items:
            inp = it['input']
            hist_match = re.search(r"### Dialogue History:\s*(.*?)\s*### Current Utterance:", inp, re.DOTALL)
            history = hist_match.group(1).strip() if hist_match else ""

            curr_match = re.search(r"### Current Utterance:\s*(.*?)\s*Based on the context,", inp, re.DOTALL)
            curr_utt = curr_match.group(1).strip() if curr_match else ""
            speaker, utt_text = parse_speaker_and_utterance(curr_utt)

            steer_prompt = (
                "You are performing Emotion Recognition in Conversations (ERC).\n"
                "The following is a conversation excerpt from EmoryNLP.\n\n"
                f"### Dialogue History:\n{history}\n"
                f"### Current Utterance:\n{curr_utt}\n\n"
                f"Please analyze the emotional state of <{speaker}: \"{utt_text}\"> by first providing your "
                "rationale based on the given context, then select the emotional label from 'Joyful', 'Mad', "
                "'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared'."
            )
            steer_target = build_rationale(speaker, utt_text, history, emotion)
            seed_records.append({"input": steer_prompt, "target": steer_target})

    print(f"Generated {len(seed_records)} seed rationale records for trainbegin.json")
    trainbegin_path = os.path.join(dst_data_dir, "trainbegin.json")
    with open(trainbegin_path, 'w', encoding='utf-8') as f:
        for r in seed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {trainbegin_path}")

    # 2. Build train4infer.json (unlearned pool for STEeR)
    train4infer_records = []
    for it in remaining_erc:
        inp = it['input']
        emotion = it['target'].strip()

        hist_match = re.search(r"### Dialogue History:\s*(.*?)\s*### Current Utterance:", inp, re.DOTALL)
        history = hist_match.group(1).strip() if hist_match else ""

        curr_match = re.search(r"### Current Utterance:\s*(.*?)\s*Based on the context,", inp, re.DOTALL)
        curr_utt = curr_match.group(1).strip() if curr_match else ""
        speaker, utt_text = parse_speaker_and_utterance(curr_utt)

        steer_prompt = (
            "You are performing Emotion Recognition in Conversations (ERC).\n"
            "The following is a conversation excerpt from EmoryNLP.\n\n"
            f"### Dialogue History:\n{history}\n"
            f"### Current Utterance:\n{curr_utt}\n\n"
            f"Please analyze the emotional state of <{speaker}: \"{utt_text}\"> by first providing your "
            "rationale based on the given context, then select the emotional label from 'Joyful', 'Mad', "
            "'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared'."
        )
        train4infer_records.append({"input": steer_prompt, "target": emotion})

    train4infer_path = os.path.join(dst_data_dir, "train4infer.json")
    with open(train4infer_path, 'w', encoding='utf-8') as f:
        for r in train4infer_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(train4infer_records)} records to: {train4infer_path}")

    # 3. Build train.json (Mixed Multi-Task Strategy B: ERC + Speaker ID + Role-Play + STEeR Rationales)
    mixed_records = list(all_lines) + seed_records
    train_mixed_path = os.path.join(dst_data_dir, "train.json")
    with open(train_mixed_path, 'w', encoding='utf-8') as f:
        for r in mixed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(mixed_records)} multi-task records to: {train_mixed_path}")
    print("\nAll datasets created successfully!")

if __name__ == "__main__":
    main()
