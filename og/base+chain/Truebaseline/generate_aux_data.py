import os
import json
import random

def generate_steer_files(dataset_dir):
    train_file = os.path.join(dataset_dir, "train.json")
    if not os.path.exists(train_file):
        return

    print(f"Generating STEeR data for {dataset_dir}...")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    all_data = [json.loads(l) for l in lines]
    
    # Filter ONLY the ERC tasks (not roleplay or speaker ID)
    erc_data = [d for d in all_data if "what is the emotion of the current utterance?" in d["input"]]
    
    steer_data = []
    
    for d in erc_data:
        input_str = d["input"]
        target_emo = d["target"] # e.g. "Emotion Flow: None. \nCurrent Emotion: neutral"
        
        # Make it ask for rationale
        steer_input = input_str.replace(
            "Answer with just the emotion label.",
            "First provide a rationale based on the context, then answer with the emotion label."
        )
        
        # Add generic rationale to target for bootstrapping
        rationale_text = f"[Rationale]\nBased on the dialogue history and the current utterance, the speaker's language and tone suggest this emotional state.\n{target_emo}"
        
        steer_data.append({"input": steer_input, "target": rationale_text})

    # Shuffle
    random.seed(42)
    random.shuffle(steer_data)
    
    # trainbegin (10 seed examples)
    trainbegin = steer_data[:10]
    
    # train4infer (take 5%)
    infer_size = int(len(steer_data) * 0.05)
    if infer_size < 10: infer_size = len(steer_data)
    train4infer = steer_data[10:10+infer_size]
    
    # Save files
    with open(os.path.join(dataset_dir, "train4infer.json"), 'w', encoding='utf-8') as f:
        for d in train4infer: f.write(json.dumps(d, ensure_ascii=False) + "\n")
        
    with open(os.path.join(dataset_dir, "trainbegin.json"), 'w', encoding='utf-8') as f:
        for d in trainbegin: f.write(json.dumps(d, ensure_ascii=False) + "\n")
        
    print(f"Generated {len(train4infer)} infer tasks, and 10 trainbegin seed rationales for {dataset_dir}!")

if __name__ == "__main__":
    base_dir = r"C:\Users\kishor\Pictures\ML_proj\Truebaseline\data"
    datasets = ["meld", "iemocap", "emorynlp"]
    for d in datasets:
        generate_steer_files(os.path.join(base_dir, d))
