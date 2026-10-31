import pandas as pd
import numpy as np
import os

def extract_thresholds_and_stats(df, num_classes):
    features = ["duration", "avg_intensity", "intensity_variation", "avg_pitch", 
                "pitch_std", "pitch_range", "articulation_rate", "mean_hnr"]
    
    thresholds = {}
    stats = {}
    
    # Calculate overall thresholds and stats
    overall_thresholds = {}
    overall_stats = {}
    for feature in features:
        if feature not in df.columns:
            continue
        feature_data = df[feature].replace(0, np.nan).dropna()
        if num_classes == 3:
            q1, q3 = feature_data.quantile([0.25, 0.75])
            overall_thresholds[feature] = {'low': q1, 'medium_high': q3}
        elif num_classes == 4:
            q1, q2, q3 = feature_data.quantile([0.25, 0.5, 0.75])
            overall_thresholds[feature] = {'low': q1, 'medium': q2, 'high': q3}
        elif num_classes == 5:
            q1, q2, q3, q4 = feature_data.quantile([0.1, 0.25, 0.75, 0.9])
            overall_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium': q3, 'high': q4}
        else:  # 6 classes
            q1, q2, q3, q4, q5 = feature_data.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            overall_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium_low': q3, 'medium_high': q4, 'high': q5}
        overall_stats[feature] = {'mean': feature_data.mean(), 'std': feature_data.std()}
    
    thresholds['overall'] = overall_thresholds
    stats['overall'] = overall_stats
    
    return thresholds, stats

def categorize(value, thresholds, num_classes):
    if pd.isna(value) or value == 0:
        return 'none'
    if num_classes == 5:
        if value <= thresholds['very_low']: return 'very_low'
        elif value <= thresholds['low']: return 'low'
        elif value <= thresholds['medium']: return 'medium'
        elif value <= thresholds['high']: return 'high'
        else: return 'very_high'
    return 'medium' # Fallback

def standardize_and_process_df(df, thresholds, stats, num_classes):
    features = list(thresholds['overall'].keys())
    
    # Standardize features
    for feature in features:
        df[f'{feature}_standardized'] = df.apply(lambda row: 
            (row[feature] - stats['overall'][feature]['mean']) / 
            stats['overall'][feature]['std']
        if not pd.isna(row[feature]) and row[feature] != 0 else np.nan, axis=1)
    
    # Categorize original features
    for feature in features:
        df[f'{feature}_category'] = df.apply(lambda row: categorize(
            row[feature], 
            thresholds['overall'][feature],
            num_classes
        ), axis=1)
    
    return df

def generate_impression(row, num_classes):
    def get_level(category):
        if category in ['very_low']: return ('high', 'very low')
        elif category in ['low']: return ('medium', 'low')
        elif category in ['medium_low', 'medium', 'medium_high']: return ('low', 'medium')
        elif category in ['high']: return ('medium', 'high')
        else: return ('high', 'very high')

    pitch_certainty, pitch = get_level(row.get('avg_pitch_category', 'medium'))
    pitch_var_certainty, pitch_var = get_level(row.get('pitch_std_category', 'medium'))
    volume_certainty, volume = get_level(row.get('avg_intensity_category', 'medium'))
    volume_var_certainty, volume_var = get_level(row.get('intensity_variation_category', 'medium'))
    rate_certainty, rate = get_level(row.get('articulation_rate_category', 'medium'))

    def get_certainty_phrase(certainty):
        if certainty == 'high': return ""
        elif certainty == 'medium': return "likely "
        else: return "may "

    if pitch in ['high', 'very high']: pitch_impression = f"{get_certainty_phrase(pitch_certainty)}uses a higher pitch"
    elif pitch in ['low', 'very low']: pitch_impression = f"{get_certainty_phrase(pitch_certainty)}uses a lower pitch"
    else: pitch_impression = "has a moderate pitch"

    if pitch_var in ['high', 'very high']: pitch_impression += f" with {get_certainty_phrase(pitch_var_certainty)}noticeable variation, suggesting expressiveness"
    elif pitch_var in ['low', 'very low']: pitch_impression += f" that {get_certainty_phrase(pitch_var_certainty)}remains steady, potentially indicating calmness or seriousness"
    else: pitch_impression += " with typical variation"

    if volume in ['high', 'very high']: volume_impression = f"{get_certainty_phrase(volume_certainty)}speaking loudly, which might indicate excitement, confidence, or urgency"
    elif volume in ['low', 'very low']: volume_impression = f"{get_certainty_phrase(volume_certainty)}speaking softly, possibly suggesting calmness, shyness, or caution"
    else: volume_impression = "using a moderate volume"

    if volume_var in ['high', 'very high']: volume_impression += f", with {get_certainty_phrase(volume_var_certainty)}significant volume changes"
    elif volume_var in ['low', 'very low']: volume_impression += f", with {get_certainty_phrase(volume_var_certainty)}little volume variation"
    else: volume_impression += ", with normal volume variation"

    if rate in ['high', 'very high']: rate_impression = f"{get_certainty_phrase(rate_certainty)}talking quickly, which could indicate excitement, urgency, or nervousness"
    elif rate in ['low', 'very low']: rate_impression = f"{get_certainty_phrase(rate_certainty)}talking slowly, possibly suggesting thoughtfulness, hesitation, or calmness"
    else: rate_impression = "speaking at a moderate pace"

    return f"The target speaker {pitch_impression}, while {volume_impression}, and is {rate_impression}."

def main():
    base_dir = r"C:\Users\kishor\Pictures\ML_proj"
    input_file = os.path.join(base_dir, "emorynlp_audio_features.csv")
    output_file = os.path.join(base_dir, "emorynlp_processed_audio_features.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    print("Loading raw features...")
    df = pd.read_csv(input_file)
    num_classes = 5
    
    print("Calculating thresholds and quantiles...")
    thresholds, stats = extract_thresholds_and_stats(df, num_classes)
    
    print("Standardizing and categorizing data...")
    processed_df = standardize_and_process_df(df, thresholds, stats, num_classes)
    
    print("Generating natural language impressions...")
    processed_df['impression'] = processed_df.apply(lambda row: generate_impression(row, num_classes), axis=1)
    
    original_features = ["duration", "avg_intensity", "intensity_variation", "avg_pitch", 
                         "pitch_std", "pitch_range", "articulation_rate", "mean_hnr"]
    
    columns_to_keep = [col for col in processed_df.columns if col not in original_features]
    output_df = processed_df[columns_to_keep]
    
    output_df.to_csv(output_file, index=False)
    print(f"Success! Final verbalized dataset saved to {output_file}")

if __name__ == "__main__":
    main()
