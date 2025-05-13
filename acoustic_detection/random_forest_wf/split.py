import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy import signal
import librosa
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from extr_func import extract_signal, extract_gtcc_paper_version, extract_mfcc, extract_time_domain_features, extract_chroma_features
import pickle

# Load dataset
dataset_name = "DroneAudioDataset/Multiclass_Done_Audio"
bebop_1 = glob.glob("../DroneAudioDataset/Multiclass_Drone_Audio/bebop_1/*.wav")
membo_1 = glob.glob("../DroneAudioDataset/Multiclass_Drone_Audio/membo_1/*.wav")
unknown = glob.glob("../DroneAudioDataset/Multiclass_Drone_Audio/unknown/*.wav")[::10]

all_files = bebop_1 + membo_1 + unknown
all_labels = (["bebop"] * len(bebop_1)) + (["membo"] * len(membo_1)) + (["unknown"] * len(unknown))

X_train, X_test, y_train, y_test = train_test_split(all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

X_train = extract_signal(X_train)
X_test = extract_signal(X_test)

feature_sets_train = {
    'gtcc': [],
    'mfcc': [],
    'time_domain': [],
    'chroma': [],
}
feature_sets_test = {
    'gtcc': [],
    'mfcc': [],
    'time_domain': [],
    'chroma': [],
}

# Process training data
for i, sample in enumerate(X_train):
    if i % 100 == 0 or i == len(X_train) - 1:
        print(f"Processing training sample {i+1}/{len(X_train)} ({(i+1)/len(X_train)*100:.1f}%)")
    # Extract individual feature types
    signal = sample["time_series"]
    sr = sample["sampling_rate"]
    # Extract all feature types
    feature_sets_train['gtcc'].append(extract_gtcc_paper_version(signal, sr))
    feature_sets_train['mfcc'].append(extract_mfcc(signal, sr))
    feature_sets_train['time_domain'].append(extract_time_domain_features(signal))
    feature_sets_train['chroma'].append(extract_chroma_features(signal, sr))

# Process test data
for i, sample in enumerate(X_test):
    # Fix: Update the progress message to use X_test instead of X_train
    if i % 100 == 0 or i == len(X_test) - 1:
        print(f"Processing test sample {i+1}/{len(X_test)} ({(i+1)/len(X_test)*100:.1f}%)")
    # Extract individual feature types
    signal = sample["time_series"]
    sr = sample["sampling_rate"]
    # Extract all feature types
    feature_sets_test['gtcc'].append(extract_gtcc_paper_version(signal, sr))
    feature_sets_test['mfcc'].append(extract_mfcc(signal, sr))
    feature_sets_test['time_domain'].append(extract_time_domain_features(signal))
    feature_sets_test['chroma'].append(extract_chroma_features(signal, sr))

# Convert lists to numpy arrays for better handling
for feature_type in feature_sets_train:
    feature_sets_train[feature_type] = np.array(feature_sets_train[feature_type])
    feature_sets_test[feature_type] = np.array(feature_sets_test[feature_type])

# Save processed features and labels to files
output_dir = "processed_features/"
import os
os.makedirs(output_dir, exist_ok=True)

# Save train features
for feature_type, features in feature_sets_train.items():
    np.save(f"{output_dir}{feature_type}_train.npy", features)

# Save test features
for feature_type, features in feature_sets_test.items():
    np.save(f"{output_dir}{feature_type}_test.npy", features)

# Save labels
np.save(f"{output_dir}y_train.npy", np.array(y_train))
np.save(f"{output_dir}y_test.npy", np.array(y_test))

# Save mapping of files to their index in the feature arrays (for reference)
with open(f"{output_dir}file_mapping_train.pkl", "wb") as f:
    pickle.dump({i: file for i, file in enumerate(X_train)}, f)

with open(f"{output_dir}file_mapping_test.pkl", "wb") as f:
    pickle.dump({i: file for i, file in enumerate(X_test)}, f)

print("All features and labels saved successfully to:", output_dir)
print(f"Train set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")