import os
import csv
import sys
import subprocess
import imageio_ffmpeg
from tqdm import tqdm  # Gives a nice progress bar

# Point this to SpeechCueLLM's extractor
sys.path.append(r'C:\Users\kishor\Pictures\ML_proj\SpeechCueLLM\feature_utils')
from extract_audio_feature import extract_audio_features_with_praat

def process_emorynlp_audio():
    base_dir = r"C:\Users\kishor\Pictures\ML_proj"
    input_base = os.path.join(base_dir, "emorynlp_video_splits")
    wav_dir = os.path.join(base_dir, "emorynlp_wavs")
    out_csv = os.path.join(base_dir, "emorynlp_audio_features.csv")
    corrupted_log = os.path.join(base_dir, "corrupted_emory_videos.txt")
    
    os.makedirs(wav_dir, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # ─── FAILSAFE / RESUME MECHANISM ───
    # Read the CSV to find out which files we've already successfully processed
    already_processed = set()
    if os.path.exists(out_csv):
        with open(out_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_processed.add(row['filename'])
                
    # Open CSV in append mode ('a') so we resume safely instead of overwriting
    mode = 'a' if len(already_processed) > 0 else 'w'
    csv_file = open(out_csv, mode, newline='', encoding='utf-8')
    writer = None
    
    splits = ["emorynlp_train_splits", "emorynlp_dev_splits", "emorynlp_test_splits"]
    
    # Gather all videos first to setup the progress bar
    all_videos_to_process = []
    for split in splits:
        split_dir = os.path.join(input_base, split)
        if not os.path.exists(split_dir): continue
        videos = [f for f in os.listdir(split_dir) if f.endswith('.mp4')]
        for vid in videos:
            if vid not in already_processed:
                all_videos_to_process.append((split_dir, vid))
                
    print(f"Resuming... {len(already_processed)} files already done.")
    print(f"Total files left to process: {len(all_videos_to_process)}")

    # Use tqdm to show a real-time progress bar in your terminal
    for split_dir, vid in tqdm(all_videos_to_process, desc="Extracting Audio Features", unit="file"):
        vid_path = os.path.join(split_dir, vid)
        wav_name = vid.replace('.mp4', '.wav')
        wav_path = os.path.join(wav_dir, wav_name)
        
        try:
            # 1. Convert MP4 -> WAV (safely overwrites corrupted partial wavs if any)
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", vid_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 2. Extract Praat Features
            features = extract_audio_features_with_praat(wav_path)
            features['filename'] = vid
            
            # Write header on the very first row of a completely new file
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=features.keys())
                if mode == 'w':
                    writer.writeheader()
                
            writer.writerow(features)
            csv_file.flush() # Forces save to disk immediately per file in case of crash!
            
        except Exception as e:
            # If a video is corrupted or empty, log it and keep going!
            with open(corrupted_log, 'a', encoding='utf-8') as err_f:
                err_f.write(f"Failed {vid}: {str(e)}\n")

    csv_file.close()
    print(f"\nAll Done! Features saved to: {out_csv}")
    if os.path.exists(corrupted_log):
        print(f"Some files were corrupted. Check {corrupted_log} for details.")

if __name__ == '__main__':
    process_emorynlp_audio()
