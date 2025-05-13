import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy import signal
import librosa
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset_name = "DroneAudioDataset/Multiclass_Done_Audio"
bebop_1 = glob.glob("DroneAudioDataset/Multiclass_Drone_Audio/bebop_1/*.wav")
membo_1 = glob.glob("DroneAudioDataset/Multiclass_Drone_Audio/membo_1/*.wav")
unknown = glob.glob("DroneAudioDataset/Multiclass_Drone_Audio/unknown/*.wav")

# Data structures to hold samples
bepop_wave_samples = []
membo_wave_samples = []
unknown_wave_samples = []

# Load audio files
for b in bebop_1:
    y, sr = librosa.load(b, sr=None)
    sample = {"time_series": y, "sampling_rate": sr, "label": "bebop"}
    bepop_wave_samples.append(sample)

for m in membo_1:
    y, sr = librosa.load(m, sr=None)
    sample = {"time_series": y, "sampling_rate": sr, "label": "membo"}
    membo_wave_samples.append(sample)

for i, u in enumerate(unknown):
    if i % 10 == 0:
        y, sr = librosa.load(u, sr=None)
        sample = {"time_series": y, "sampling_rate": sr, "label": "unknown"}
        unknown_wave_samples.append(sample)

# Combine all samples
all_samples = bepop_wave_samples + membo_wave_samples + unknown_wave_samples

# Print dataset statistics
print(f"Total samples: {len(all_samples)}")
print(f"Bebop samples: {len(bepop_wave_samples)}")
print(f"Membo samples: {len(membo_wave_samples)}")
print(f"Unknown samples: {len(unknown_wave_samples)}")

# === Feature Extraction Functions ===

# Keep your existing GTCC extraction code
def create_gammatone_filterbank(num_filters, fs, low_freq=50, high_freq=None):
    if high_freq is None:
        high_freq = fs / 2
        
    # Filter order (typically 4 for gammatone)
    n = 4
    
    # Calculate ERB space center frequencies
    ears_constant = 9.26449
    min_erb = 24.7 * (4.37 * low_freq / 1000 + 1)
    max_erb = 24.7 * (4.37 * high_freq / 1000 + 1)
    
    # Space center frequencies on an ERB scale
    center_freqs_erb = np.linspace(min_erb, max_erb, num_filters)
    center_freqs = (center_freqs_erb / ears_constant - 1) * 1000 / 4.37
    
    # FFT parameters
    fft_size = 512
    fftfreqs = np.linspace(0, fs/2, fft_size//2+1)
    
    # Initialize filter bank
    filters = np.zeros((num_filters, fft_size//2+1))
    
    # Create each filter in the bank
    for i, cf in enumerate(center_freqs):
        # Calculate ERB bandwidth for this center frequency
        erb_bandwidth = 24.7 * (4.37 * cf / 1000 + 1)
        b = 1.019 * erb_bandwidth  # Bandwidth parameter
        
        # Generate the gammatone filter in the frequency domain
        # Using approximation of the gammatone magnitude response
        for j, f in enumerate(fftfreqs):
            if f > 0:  # Avoid division by zero at DC
                # Simplified gammatone magnitude response
                # |H(f)| = (1 + (f/cf - 1)^2 * (b/cf)^2)^(-n/2)
                filters[i, j] = (1 + (f/cf - 1)**2 * (b/cf)**2)**(-n/2)
        
        # Normalize filter to have unit peak response
        if filters[i].max() > 0:
            filters[i] = filters[i] / filters[i].max()
    
    return filters, center_freqs

def extract_gtcc_paper_version(signal, sample_rate, num_ceps=13, num_filters=40):
    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    # Framing parameters
    frame_size = 0.025  # 25ms
    frame_stride = 0.010  # 10ms
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    
    # Adjust frame length if needed
    frame_length = min(frame_length, len(emphasized_signal) // 4)
    frame_step = frame_length // 2  # 50% overlap

    # Frame the signal
    frames = librosa.util.frame(emphasized_signal, frame_length=frame_length, hop_length=frame_step)
    
    frames_shape = frames.shape
    
    # Apply windowing
    window = np.hamming(frames_shape[1])
    window = window.reshape(1, -1)  # Shape: (1, frame_length)
    frames = frames * window
    
    # Power spectrum
    fft_size = 512
    mag_frames = np.abs(np.fft.rfft(frames, fft_size))
    power_frames = (1.0/fft_size) * np.square(mag_frames)
    
    # Create gammatone filterbank
    filter_bank, center_freqs = create_gammatone_filterbank(
        num_filters=num_filters, 
        fs=sample_rate,
        low_freq=50
    )
    
    # Apply filterbank
    filter_banks = np.dot(power_frames, filter_bank.T)
    
    # Log compression
    epsilon = 1e-10
    log_filter_banks = np.log(filter_banks + epsilon)
    
    # DCT
    gtcc = dct(log_filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]
    
    # Delta and delta-delta features
    # Ensure width is appropriate for the number of frames
    safe_width = min(9, gtcc.shape[0]-1) if gtcc.shape[0] > 1 else 1
    
    if gtcc.shape[0] > 1:
        delta_gtcc = librosa.feature.delta(gtcc, width=safe_width)
        delta2_gtcc = librosa.feature.delta(gtcc, width=safe_width, order=2)
    else:
        # Handle edge case with just one frame
        delta_gtcc = np.zeros_like(gtcc)
        delta2_gtcc = np.zeros_like(gtcc)
    
    # Combine features - mean across frames for fixed-length feature vector
    gtcc_features = np.hstack((
        np.mean(gtcc, axis=0),
        np.mean(delta_gtcc, axis=0),
        np.mean(delta2_gtcc, axis=0)
    ))
    
    return gtcc_features

# 1. MFCC (Mel-Frequency Cepstral Coefficients)
def extract_mfcc(signal, sample_rate, num_ceps=13, num_filters=40):
    """
    Extract MFCC features from audio signal.
    Implementation based on your GTCC function for consistent comparison.
    """
    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    # Framing parameters
    frame_size = 0.025  # 25ms
    frame_stride = 0.010  # 10ms
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    
    # Adjust frame length if needed
    frame_length = min(frame_length, len(emphasized_signal) // 4)
    frame_step = frame_length // 2  # 50% overlap

    # Frame the signal
    frames = librosa.util.frame(emphasized_signal, frame_length=frame_length, hop_length=frame_step)
    
    # Apply windowing
    window = np.hamming(frames.shape[1])
    window = window.reshape(1, -1)
    frames = frames * window
    
    # Power spectrum
    fft_size = 512
    mag_frames = np.abs(np.fft.rfft(frames, fft_size))
    power_frames = (1.0/fft_size) * np.square(mag_frames)
    
    # Apply Mel filterbank
    mel_filters = librosa.filters.mel(sr=sample_rate, 
                                     n_fft=fft_size, 
                                     n_mels=num_filters, 
                                     fmin=50, 
                                     fmax=sample_rate/2)
    
    mel_energy = np.dot(power_frames, mel_filters.T)
    
    # Log compression
    epsilon = 1e-10
    log_mel_energy = np.log(mel_energy + epsilon)
    
    # DCT (to get cepstral coefficients)
    mfcc = dct(log_mel_energy, type=2, axis=1, norm='ortho')[:, :num_ceps]
    
    # Compute delta and delta-delta features
    safe_width = min(9, mfcc.shape[0]-1) if mfcc.shape[0] > 1 else 1
    
    if mfcc.shape[0] > 1:
        delta_mfcc = librosa.feature.delta(mfcc, width=safe_width)
        delta2_mfcc = librosa.feature.delta(mfcc, width=safe_width, order=2)
    else:
        delta_mfcc = np.zeros_like(mfcc)
        delta2_mfcc = np.zeros_like(mfcc)
    
    # Combine features
    mfcc_features = np.hstack((
        np.mean(mfcc, axis=0),
        np.mean(delta_mfcc, axis=0),
        np.mean(delta2_mfcc, axis=0)
    ))
    
    return mfcc_features

# 2. LPC (Linear Prediction Coefficients)
def extract_lpc(signal, sample_rate, order=13):
    """
    Extract LPC features from audio signal.
    
    Parameters:
    signal : ndarray
        Input audio signal
    sample_rate : int
        Sampling rate
    order : int
        Order of the LPC analysis (typically 13 for speech)
        
    Returns:
    lpc_features : ndarray
        LPC features
    """
    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    # Framing parameters
    frame_size = 0.025  # 25ms
    frame_stride = 0.010  # 10ms
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    
    # Adjust frame length if needed
    frame_length = min(frame_length, len(emphasized_signal) // 4)
    frame_step = frame_length // 2  # 50% overlap

    # Frame the signal
    frames = librosa.util.frame(emphasized_signal, frame_length=frame_length, hop_length=frame_step)
    
    # Apply windowing
    window = np.hamming(frames.shape[1])
    window = window.reshape(1, -1)
    frames = frames * window
    
    # Calculate LPC coefficients for each frame
    lpc_frames = np.zeros((frames.shape[0], order))
    for i in range(frames.shape[0]):
        a = librosa.lpc(frames[i], order=order)
        lpc_frames[i] = a[1:]  # Skip the first coefficient (a[0] = 1)
    
    # Take mean of coefficients across frames
    lpc_features = np.mean(lpc_frames, axis=0)
    
    return lpc_features

# 3. Spectral features (spectral centroid, bandwidth, flatness, rolloff)
def extract_spectral_features(signal, sample_rate):
    """
    Extract various spectral features from audio signal.
    
    Returns:
    spectral_features : ndarray
        Combined spectral features
    """
    # Calculate spectral features using librosa
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=signal)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)[0]
    
    # Combine features by taking statistics
    features = np.hstack((
        [np.mean(spectral_centroid), np.std(spectral_centroid)],
        [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
        [np.mean(spectral_flatness), np.std(spectral_flatness)],
        [np.mean(spectral_rolloff), np.std(spectral_rolloff)]
    ))
    
    return features

# 4. Zero Crossing Rate and RMS Energy
def extract_time_domain_features(signal):
    """
    Extract time domain features from audio signal.
    
    Returns:
    time_features : ndarray
        Combined time domain features
    """
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    
    # RMS energy
    rms = librosa.feature.rms(y=signal)[0]
    
    # Combine features
    features = np.hstack((
        [np.mean(zcr), np.std(zcr)],
        [np.mean(rms), np.std(rms)]
    ))
    
    return features

# 5. Chroma features (relates to the 12 different pitch classes)
def extract_chroma_features(signal, sample_rate):
    """
    Extract chroma features from audio signal.
    
    Returns:
    chroma_features : ndarray
        Combined chroma features
    """
    # Compute chroma features
    chroma = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
    
    # Take mean and std for each chroma bin
    chroma_means = np.mean(chroma, axis=1)
    chroma_stds = np.std(chroma, axis=1)
    
    # Combine features
    features = np.hstack((chroma_means, chroma_stds))
    
    return features

# Function to extract all features
def extract_all_features(signal, sample_rate):
    """
    Extract all features from audio signal and combine them.
    
    Returns:
    all_features : ndarray
        Combined features
    """
    # Extract individual feature sets
    gtcc_features = extract_gtcc_paper_version(signal, sample_rate)
    mfcc_features = extract_mfcc(signal, sample_rate)
    lpc_features = extract_lpc(signal, sample_rate)
    spectral_features = extract_spectral_features(signal, sample_rate)
    time_features = extract_time_domain_features(signal)
    chroma_features = extract_chroma_features(signal, sample_rate)
    
    # Combine all features
    all_features = np.hstack((
        gtcc_features,  # Your original GTCC features
        mfcc_features,  # MFCC features
        lpc_features,   # LPC features
        spectral_features,  # Spectral features
        time_features,  # Time domain features
        chroma_features  # Chroma features
    ))
    
    return all_features

# === Extract features and save them ===
# Dictionary to hold different feature sets
feature_sets = {
    'gtcc': [],
    'mfcc': [],
    'lpc': [],
    'spectral': [],
    'time_domain': [],
    'chroma': [],
    'combined': []
}

# Extract features from all samples with a progress indicator
total_samples = len(all_samples)
print(f"Extracting features from {total_samples} samples...")

for i, sample in enumerate(all_samples):
    if i % 100 == 0 or i == total_samples - 1:
        print(f"Processing sample {i+1}/{total_samples} ({(i+1)/total_samples*100:.1f}%)")
    
    # Extract individual feature types
    signal = sample["time_series"]
    sr = sample["sampling_rate"]
    
    # Extract all feature types
    feature_sets['gtcc'].append(extract_gtcc_paper_version(signal, sr))
    feature_sets['mfcc'].append(extract_mfcc(signal, sr))
    feature_sets['lpc'].append(extract_lpc(signal, sr))
    feature_sets['spectral'].append(extract_spectral_features(signal, sr))
    feature_sets['time_domain'].append(extract_time_domain_features(signal))
    feature_sets['chroma'].append(extract_chroma_features(signal, sr))
    feature_sets['combined'].append(extract_all_features(signal, sr))

# Collect labels
labels = [sample["label"] for sample in all_samples]

# Save each feature set
for feature_type, features in feature_sets.items():
    features_array = np.array(features)
    np.save(f'X_{feature_type}.npy', features_array)
    print(f"Saved {feature_type} features with shape {features_array.shape}")

# Save labels once
labels_array = np.array(labels)
np.save('y.npy', labels_array)
print(f"Saved labels with shape {labels_array.shape}")

# Display feature dimensions
print("\nFeature set dimensions:")
for feature_type, features in feature_sets.items():
    print(f"{feature_type}: {np.array(features).shape}")