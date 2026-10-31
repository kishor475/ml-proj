"""
prepare_datasets.py
-------------------
Converts raw MELD, EmoryNLP, and IEMOCAP datasets into the JSONL format required by the CoE
training pipeline, now injecting the acoustic clues from SpeechCueLLM!

    {"input": "<prompt string>", "target": "<emotion label>"}

Each line in the output file is one JSON object (JSONL format).
"""

import json
import os
import csv

# ─── Paths ────────────────────────────────────────────────────────────────────

CURRENT_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR     = os.path.dirname(CURRENT_DIR)

MELD_DIR       = os.path.join(PARENT_DIR, "MELD", "data", "MELD")
EMORYNLP_DIR   = os.path.join(PARENT_DIR, "emotion-detection")
IEMOCAP_DIR    = os.path.join(PARENT_DIR, "SpeechCueLLM", "data", "IEMOCAP_full_release")
SPEECHCUE_DIR  = os.path.join(PARENT_DIR, "SpeechCueLLM", "data")
OUT_DIR        = os.path.join(CURRENT_DIR, "data")

# ─── Label normalization ──────────────────────────────────────────────────────

# MELD: neutral, surprise, fear, sad, joyful, disgust, angry
MELD_EMOTION_MAP = {
    "neutral":  "neutral",
    "surprise": "surprise",
    "fear":     "fear",
    "sadness":  "sad",
    "joy":      "joyful",
    "disgust":  "disgust",
    "anger":    "angry",
    "sad":      "sad",
    "joyful":   "joyful",
    "angry":    "angry",
}

# IEMOCAP: happy, sad, neutral, angry, excited, frustrated
IEMOCAP_EMOTION_MAP = {
    "neu": "neutral",
    "fru": "frustrated",
    "ang": "angry",
    "sad": "sad",
    "exc": "excited",
    "hap": "happy",
}

IEMOCAP_RELATIONSHIPS = {
    "impro01": "A frustrated customer and a clerk at a counter.",
    "impro02": "Friends dealing with bad news.",
    "impro03": "Friends catching up and sharing surprising news.",
    "impro04": "Friends discussing looking for something.",
    "impro05": "An angry customer and a worker.",
    "impro06": "A friend comforting another friend experiencing loss or sadness.",
    "impro07": "Friends celebrating good news.",
    "impro08": "A frustrated customer and a customer service agent.",
    "script01_1": "Family members discussing a difficult situation.",
    "script01_2": "Family members arguing about a guest.",
    "script01_3": "Friends having an intimate conversation.",
    "script02_1": "A couple on a romantic date or outing.",
    "script02_2": "A couple having a romantic moment.",
    "script03_1": "A couple in hiding or a secretive affair.",
    "script03_2": "An older couple reminiscing about their past."
}

FRIENDS_SIDE_CHARACTERS = {
    "carol": "Ross's ex-wife.",
    "susan": "Carol's partner.",
    "barry": "Rachel's ex-fiancé.",
    "richard": "Monica's older ex-boyfriend.",
    "mike": "Phoebe's romantic partner.",
    "gunther": "The coffee shop manager who has a crush on Rachel.",
    "janice": "Chandler's on-and-off annoying ex-girlfriend.",
    "paul": "Monica's bad date.",
    "estelle": "Joey's eccentric talent agent.",
    "judy": "Ross and Monica's mother.",
    "jack": "Ross and Monica's father.",
    "frank": "Phoebe's half-brother.",
    "ursula": "Phoebe's twin sister.",
    "tag": "Rachel's young assistant and boyfriend.",
    "emily": "Ross's British ex-wife.",
    "stephen": "Emily's father.",
    "andrea": "Emily's mother.",
    "heckles": "The downstairs neighbor.",
    "treeger": "The building superintendent.",
    "joshua": "Rachel's client and boyfriend.",
    "pete": "Monica's millionaire boyfriend.",
    "charlie": "Ross and Joey's academic girlfriend.",
    "mona": "Ross's girlfriend.",
    "david": "Phoebe's scientist boyfriend.",
    "amy": "Rachel's sister.",
    "jill": "Rachel's sister.",
    "ben": "Ross's son."
}

FRIENDS_MAIN_PERSONAS = {
    "chandler": "Known for wit and sarcasm, works in advertising, group's humorist with logical insights.",
    "phoebe": "A masseuse and musician, eccentric with unique wisdom, offering empathy to the group.",
    "ross": "A paleontologist and professor, adds intellectual depth, connections through family ties with Monica and past relationship with Rachel.",
    "rachel": "Progresses from a waitress to fashion, influences group dynamics, evolving relationship with Ross.",
    "monica": "A chef, known for organization, group's anchor with hospitality, Ross's sister.",
    "joey": "An aspiring actor, the group's innocent and charming member, deepening bond with Chandler through loyalty and simplicity."
}

EMORYNLP_VALID_LABELS = {"Joyful", "Mad", "Peaceful", "Neutral", "Sad", "Powerful", "Scared"}

# ─── Prompt builders ──────────────────────────────────────────────────────────

def build_prompts(dataset_name, history, current_utterance, current_speaker, emotion_label_str, norm_emotion, acoustic_clue=None, relationship_clue=None):
    history_str = ""
    history_speakers = set()
    for spk, utt, emo in history:
        history_str += f"{spk}: {utt}\n"
        history_speakers.add(spk)
    history_speakers.add(current_speaker)
    
    # =========================================================================
    # CLUE INJECTION SECTION
    # Here we embed the novel multimodal and social clues directly into the context.
    # =========================================================================

    acoustic_section = ""
    rel_section = ""

    # Persona Injection for MELD and EmoryNLP (Friends characters)
    persona_lines = []
    for spk in history_speakers:
        spk_lower = spk.lower().strip()
        if spk_lower in FRIENDS_MAIN_PERSONAS:
            persona_lines.append(f"{spk.capitalize()}: {FRIENDS_MAIN_PERSONAS[spk_lower]}")
            
    persona_section = ""
    if persona_lines:
        persona_section = "### Speaker Persona:\n" + "\n".join(persona_lines) + "\n\n"

    # =========================================================================
    # MULTI-STAGE AUXILIARY TASK GENERATION (Strategy B Implementation)
    # Generates 3 separate task prompts per utterance to create the mixed dataset.
    # =========================================================================

    # 1. Main ERC Task
    # [ERC Task]: Standard Emotion Recognition in Conversations task.
    erc_prompt = (
        f"You are performing Emotion Recognition in Conversations (ERC).\n"
        f"The following is a conversation excerpt from {dataset_name}.\n\n"
        f"{persona_section}"
        f"### Dialogue History:\n{history_str}"
        f"{acoustic_section}"
        f"{rel_section}"
        f"### Current Utterance:\n{current_speaker}: {current_utterance}\n\n"
        f"Based on the context, what is the emotion of the current utterance?\n"
        f"Choose from: {emotion_label_str}\n"
        f"Answer with just the emotion label."
    )
    
    # [Chain of Emotion / Emotion Transition Tracking]: The target sequence over the window.
    # [STEeR Task]: By forcing this Chain-of-Thought into the ERC target, we implicitly satisfy the STEeR rationale reasoning requirement!
    spk_flow = [emo for spk, u, emo in history if spk == current_speaker and emo != 'Unknown']
    flow_str = " -> ".join(spk_flow) if spk_flow else "None"
    erc_target = f"Emotion Flow: {flow_str}. \nCurrent Emotion: {norm_emotion}"
    
    # 2. Speaker Recognition Task
    # [Speaker Identification Task]: A CoE baseline auxiliary task to learn character traits.
    spk_list_str = "<" + ", ".join(list(history_speakers)) + ">"
    speaker_prompt = (
        f"You are now an expert in Conversational Emotion Recognition. Your task is to predict the speaker by analyzing sentences.\n"
        f"{persona_section}"
        f"### Dialogue History:\n{history_str}"
        f"{acoustic_section}"
        f"{rel_section}"
        f"Please select the Speaker of <\"{current_utterance}\", Emotion: {norm_emotion} > from {spk_list_str}."
    )
    speaker_target = current_speaker
    
    # 3. Role-play Task
    # [Role-Playing Task]: A CoE baseline auxiliary task to generate dialogue in-character.
    role_prompt = (
        f"You are now an expert role-play actor. Analyze the character's Scene and Dialogue History to predict their most likely next line.\n"
        f"{persona_section}"
        f"### Dialogue History:\n{history_str}"
        f"{acoustic_section}"
        f"{rel_section}"
        f"You will stay in character whenever possible, and generate responses in <{norm_emotion}> emotion as if you were <{current_speaker}>."
    )
    role_target = current_utterance
    
    return [
        {"input": erc_prompt.strip(), "target": erc_target},
        {"input": speaker_prompt.strip(), "target": speaker_target},
        {"input": role_prompt.strip(), "target": role_target}
    ]


# ─── Converters ───────────────────────────────────────────────────────────────

def load_acoustic_features(csv_path, key_cols):
    """Loads acoustic impressions into a dict keyed by key_cols."""
    mapping = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Acoustic CSV not found at {csv_path}")
        return mapping
        
    def safe_parse(val):
        val = str(val).strip()
        try:
            return str(int(float(val)))
        except ValueError:
            return val

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(safe_parse(row[k]) for k in key_cols)
            impression = row.get('impression', '').strip()
            if impression:
                mapping[key] = impression
    return mapping


def convert_iemocap(out_path, window=5):
    """Reads iemocap_full_dataset and mapped acoustic features."""
    emotion_label_str = "'happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'"
    
    # Load acoustic mappings: (video_id, segment_id) -> impression
    acoustic_csv = os.path.join(SPEECHCUE_DIR, "processed_iemocap_audio_features_5.csv")
    acoustic_map = load_acoustic_features(acoustic_csv, ['video_id', 'segment_id'])
    
    main_csv = os.path.join(IEMOCAP_DIR, "iemocap_full_dataset.csv")
    
    # Group by video_id
    dialogues = {}
    with open(main_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['video_id']
            seg_id = int(row['segment_id'])
            emotion = row['emotion']
            speaker = row['gender']  # F or M
            utt = row['text']
            mode = row['mode'] # train, test
            
            if vid not in dialogues:
                dialogues[vid] = []
            dialogues[vid].append({
                'seg_id': seg_id,
                'speaker': speaker,
                'utt': utt,
                'emotion': emotion,
                'mode': mode,
                'vid': vid
            })
            
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # IEMOCAP splits are usually determined by cross-validation sessions or train/test splits.
    # Here we just write train and test based on the 'mode' column.
    out_train = os.path.join(os.path.dirname(out_path), "train.json")
    out_test = os.path.join(os.path.dirname(out_path), "test.json")
    
    written_train, written_test, skipped = 0, 0, 0
    
    with open(out_train, 'w', encoding='utf-8') as f_train, open(out_test, 'w', encoding='utf-8') as f_test:
        for vid, utterances in dialogues.items():
            utterances.sort(key=lambda x: x['seg_id'])
            history = []
            
            for u in utterances:
                raw_emo = u['emotion'].lower()
                norm_emo = IEMOCAP_EMOTION_MAP.get(raw_emo)
                
                # Drop xxx, oth, dis, sur, fea
                if not norm_emo:
                    history.append((u['speaker'], u['utt'], 'Unknown'))
                    skipped += 1
                    continue
                    
                impression = acoustic_map.get((u['vid'], str(u['seg_id'])), "No acoustic data available.")
                
                # Extract scenario ID for IEMOCAP Relationship (e.g. Ses01F_impro01 -> impro01)
                scenario_id = "_".join(u['vid'].split("_")[1:])
                rel_clue = IEMOCAP_RELATIONSHIPS.get(scenario_id, "Dyadic conversational partners.")

                recent_history = history[-window:]
                records = build_prompts("IEMOCAP", recent_history, u['utt'], u['speaker'], emotion_label_str, norm_emo, impression, rel_clue)
                
                session_id = u['vid'][:5]
                out_f = f_test if session_id == 'Ses05' else f_train
                if session_id == 'Ses05':
                    records = [records[0]] # Only keep ERC task for evaluation
                    
                for record in records:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                if session_id == 'Ses05': written_test += 1
                else: written_train += 1
                
                history.append((u['speaker'], u['utt'], norm_emo))
                
    # CoE uses test.json as eval, so we will also duplicate test.json to dev.json just in case.
    import shutil
    shutil.copy(out_test, os.path.join(os.path.dirname(out_path), "dev.json"))
                
    print(f"[IEMOCAP] Written: {written_train} train, {written_test} test, Skipped: {skipped}")


def convert_meld(split_name, csv_path, acoustic_csv_path, out_path, window=5):
    emotion_label_str = "'neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'"
    acoustic_map = load_acoustic_features(acoustic_csv_path, ['Dialogue_ID', 'Utterance_ID'])

    dialogues = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dlg_id  = int(row['Dialogue_ID'])
            utt_id  = int(row['Utterance_ID'])
            emotion = row['Emotion'].strip().lower()
            speaker = row['Speaker'].strip()
            utt     = row['Utterance'].strip()

            if dlg_id not in dialogues:
                dialogues[dlg_id] = []
            dialogues[dlg_id].append((utt_id, speaker, utt, emotion, str(dlg_id), str(utt_id)))

    for dlg_id in dialogues:
        dialogues[dlg_id].sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    skipped, written = 0, 0

    with open(out_path, 'w', encoding='utf-8') as out_f:
        for dlg_id, utterances in dialogues.items():
            history = []
            for utt_id, speaker, utt, emotion, d_id_str, u_id_str in utterances:
                norm_emotion = MELD_EMOTION_MAP.get(emotion)
                if norm_emotion is None:
                    skipped += 1
                    history.append((speaker, utt, 'Unknown'))
                    continue

                impression = acoustic_map.get((d_id_str, u_id_str), "No acoustic data available.")
                recent_history = history[-window:]
                
                # Check for friends
                friends_core = {"rachel", "ross", "monica", "joey", "chandler", "phoebe", "all"}
                current_speakers = {spk.lower().replace('#', '').split()[0] for spk, _, _ in recent_history}
                current_speakers.add(speaker.lower().replace('#', '').split()[0])
                
                if current_speakers.issubset(friends_core):
                    rel_clue = "The speakers are a group of close friends."
                else:
                    friends_present = [spk.capitalize() for spk in current_speakers if spk in friends_core and spk != 'all']
                    strangers_present = [spk.capitalize() for spk in current_speakers if spk not in friends_core and spk not in FRIENDS_SIDE_CHARACTERS]
                    
                    rel_parts = []
                    if friends_present:
                        rel_parts.append(f"close friends ({', '.join(friends_present)})")
                    
                    side_chars_found = [FRIENDS_SIDE_CHARACTERS[spk] for spk in current_speakers if spk in FRIENDS_SIDE_CHARACTERS]
                    if side_chars_found:
                        rel_parts.append("known associates (" + ", ".join(list(set(side_chars_found))) + ")")
                        
                    if strangers_present:
                        rel_parts.append(f"acquaintances/strangers ({', '.join(strangers_present)})")
                        
                    rel_clue = "The speakers involve interacting groups: " + " and ".join(rel_parts) + "."

                records = build_prompts("MELD", recent_history, utt, speaker, emotion_label_str, norm_emotion, impression, rel_clue)
                
                if split_name != 'train':
                    records = [records[0]]

                for record in records:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                history.append((speaker, utt, norm_emotion))

    print(f"[MELD {split_name}] Written: {written}, Skipped: {skipped}")


def load_emory_acoustic_features():
    csv_path = r"C:\Users\kishor\Pictures\ML_proj\emorynlp_processed_audio_features.csv"
    features_dict = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                features_dict[row['filename']] = row['impression']
    return features_dict

def json_id_to_filename(utt_id):
    # Converts 's01_e02_c01_u001' -> 'sea1_ep2_sc1_utt1.mp4'
    parts = utt_id.split('_')
    if len(parts) != 4: return None
    try:
        season = int(parts[0][1:])
        ep = int(parts[1][1:])
        sc = int(parts[2][1:])
        utt = int(parts[3][1:])
        return f'sea{season}_ep{ep}_sc{sc}_utt{utt}.mp4'
    except ValueError:
        return None

def convert_emorynlp(split_name, json_path, out_path, emory_audio_dict, window=5):
    emotion_label_str = "'Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared'"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    skipped, written = 0, 0

    with open(out_path, 'w', encoding='utf-8') as out_f:
        for episode in data['episodes']:
            for scene in episode['scenes']:
                history = []
                for utt in scene['utterances']:
                    speakers   = utt.get('speakers', ['Unknown'])
                    speaker    = speakers[0] if speakers else 'Unknown'
                    transcript = utt.get('transcript', '').strip()
                    emotion    = utt.get('emotion', '').strip()
                    utt_id     = utt.get('utterance_id', '')

                    if not emotion or emotion not in EMORYNLP_VALID_LABELS:
                        history.append((speaker, transcript, 'Unknown'))
                        skipped += 1
                        continue

                    # Mathematically reconstruct filename to query the CSV
                    vid_filename = json_id_to_filename(utt_id)
                    impression = emory_audio_dict.get(vid_filename, "No distinct acoustic variation observed in transcript.")
                    
                    recent_history = history[-window:]

                    # Check for friends
                    friends_core = {"rachel", "ross", "monica", "joey", "chandler", "phoebe", "all"}
                    current_speakers = {spk.lower().replace('#', '').split()[0] for spk, _, _ in recent_history}
                    current_speakers.add(speaker.lower().replace('#', '').split()[0])
                    
                    if current_speakers.issubset(friends_core):
                        rel_clue = "The speakers are a group of close friends."
                    else:
                        friends_present = [spk.capitalize() for spk in current_speakers if spk in friends_core and spk != 'all']
                        strangers_present = [spk.capitalize() for spk in current_speakers if spk not in friends_core and spk not in FRIENDS_SIDE_CHARACTERS]
                        
                        rel_parts = []
                        if friends_present:
                            rel_parts.append(f"close friends ({', '.join(friends_present)})")
                        
                        side_chars_found = [FRIENDS_SIDE_CHARACTERS[spk] for spk in current_speakers if spk in FRIENDS_SIDE_CHARACTERS]
                        if side_chars_found:
                            rel_parts.append("known associates (" + ", ".join(list(set(side_chars_found))) + ")")
                            
                        if strangers_present:
                            rel_parts.append(f"acquaintances/strangers ({', '.join(strangers_present)})")
                            
                        rel_clue = "The speakers involve interacting groups: " + " and ".join(rel_parts) + "."

                    records = build_prompts("EmoryNLP", recent_history, transcript, speaker, emotion_label_str, emotion, impression, rel_clue)
                    
                    if split_name != 'train':
                        records = [records[0]]
                        
                    for record in records:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    history.append((speaker, transcript, emotion))

    print(f"[EmoryNLP {split_name}] Written: {written}, Skipped: {skipped}")


def main():
    print("=" * 60)
    print("  CoE + SpeechCueLLM Dataset Converter")
    print("=" * 60)

    # ── IEMOCAP ────────────────────────────────────────────────────
    print("\n[1/3] Converting IEMOCAP with Acoustic Clues...")
    convert_iemocap(
        out_path = os.path.join(OUT_DIR, "iemocap", "dataset.json")
    )

    # ── MELD ──────────────────────────────────────────────────────
    print("\n[2/3] Converting MELD with Acoustic Clues...")
    convert_meld("train", 
                 os.path.join(MELD_DIR, "train_sent_emo.csv"),
                 os.path.join(SPEECHCUE_DIR, "meld_processed_5class_train_audio_features.csv"),
                 os.path.join(OUT_DIR, "meld", "train.json"))
    convert_meld("dev", 
                 os.path.join(MELD_DIR, "dev_sent_emo.csv"),
                 os.path.join(SPEECHCUE_DIR, "meld_processed_5class_dev_audio_features.csv"),
                 os.path.join(OUT_DIR, "meld", "dev.json"))
    convert_meld("test", 
                 os.path.join(MELD_DIR, "test_sent_emo.csv"),
                 os.path.join(SPEECHCUE_DIR, "meld_processed_5class_test_audio_features.csv"),
                 os.path.join(OUT_DIR, "meld", "test.json"))

    # ── EmoryNLP ──────────────────────────────────────────────────
    print("\n[3/3] Converting EmoryNLP with Acoustic Clues...")
    emory_audio_dict = load_emory_acoustic_features()
    convert_emorynlp("train", os.path.join(EMORYNLP_DIR, "emotion-detection-trn.json"), os.path.join(OUT_DIR, "emorynlp", "train.json"), emory_audio_dict)
    convert_emorynlp("dev",   os.path.join(EMORYNLP_DIR, "emotion-detection-dev.json"), os.path.join(OUT_DIR, "emorynlp", "dev.json"), emory_audio_dict)
    convert_emorynlp("test",  os.path.join(EMORYNLP_DIR, "emotion-detection-tst.json"), os.path.join(OUT_DIR, "emorynlp", "test.json"), emory_audio_dict)

    print("\nDone! Files written to:", OUT_DIR)

if __name__ == "__main__":
    main()
