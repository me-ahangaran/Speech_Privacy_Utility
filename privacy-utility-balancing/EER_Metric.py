""" The Equal Error Rate (EER) for speaker verification was calculated using the Librosa package with MFCC features.
The input file, 'Input.csv', contains three columns: 'id' (the unique identifier for each speaker), 
'speech_path' (the path to the original speech file), and 'speech_anonymized_path' (the path to the corresponding anonymized speech file).
The EER is computed based on the MFCC features extracted from both the original and anonymized speech files.
"""
import pandas as pd
import librosa
import numpy as np
from sklearn.metrics import roc_curve, auc

# Load datasets
main_df = pd.read_csv('input.csv')

# Function to extract MFCC features from audio files
def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path)  # Set sampling rate to 16kHz
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    
    # Calculate mean MFCC coefficients
    mean_mfccs = np.mean(mfccs, axis=1)
    
    return mean_mfccs

# Function to extract embeddings from audio files
def extract_embeddings(df, file_path):
    embeddings = {}
    for _, row in df.iterrows():
        speaker_id = row['id']
        speech_path = row[file_path]
        
        # Extract features
        features = extract_features(speech_path)
        
        if speaker_id not in embeddings:
            embeddings[speaker_id] = []
        embeddings[speaker_id].append(features)
    
    return embeddings

# Extract embeddings for original and altered datasets
original_embeddings = extract_embeddings(main_df, 'speech_path')
altered_embeddings = extract_embeddings(main_df, 'speech_anonymized_path')

# Function to compute EER
def compute_eer(original_embeddings, altered_embeddings):
    similarities = []
    labels = []

    # Compare each embedding in original_embeddings with all embeddings in altered_embeddings
    for speaker_id, orig_emb_list in original_embeddings.items():
        for orig_emb in orig_emb_list:
            for speaker_id_alt, alt_emb_list in altered_embeddings.items():
                for alt_emb in alt_emb_list:
                    similarity = np.dot(orig_emb, alt_emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(alt_emb))
                    similarities.append(similarity)
                    labels.append(1 if speaker_id == speaker_id_alt else 0)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, similarities, pos_label=1)
    fnr = 1 - tpr

    # Find the threshold where FPR equals FNR
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    # Calculate AUC score
    roc_auc = auc(fpr, tpr)

    return eer, roc_auc

# Calculate EER
eer, roc_auc = compute_eer(original_embeddings, altered_embeddings)

print(f'Equal Error Rate (EER) after anonymization: {eer:.4f}')
print(f'Area Under the ROC Curve (AUC): {roc_auc:.4f}')