import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import os
import librosa
import librosa.display
from scipy import stats
from collections import defaultdict
import pickle

# Set the aesthetics for all the plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Function to load the raw audio data
def load_audio_data(file_paths):
    data = []
    for file_path in file_paths:
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            data.append({
                'file_path': file_path,
                'signal': y,
                'sr': sr,
                'duration': len(y) / sr,
                'class': os.path.basename(os.path.dirname(file_path))
            })
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return pd.DataFrame(data)

# Load the dataset
def load_dataset():
    bebop_1 = glob.glob("../DroneAudioDataset/Multiclass_Drone_Audio/bebop_1/*.wav")
    membo_1 = glob.glob("../DroneAudioDataset/Multiclass_Drone_Audio/membo_1/*.wav")
    unknown = glob.glob("../DroneAudioDataset/Multiclass_Drone_Audio/unknown/*.wav")[::10]  # Using subset as in original script
    
    print(f"Found {len(bebop_1)} bebop files, {len(membo_1)} membo files, and {len(unknown)} unknown files")
    
    # Load audio data (limit to 100 files per class for initial analysis if dataset is large)
    bebop_df = load_audio_data(bebop_1[:100] if len(bebop_1) > 100 else bebop_1)
    membo_df = load_audio_data(membo_1[:100] if len(membo_1) > 100 else membo_1)
    unknown_df = load_audio_data(unknown[:100] if len(unknown) > 100 else unknown)
    
    # Combine datasets and set class explicitly
    bebop_df['class'] = 'bebop'
    membo_df['class'] = 'membo'
    unknown_df['class'] = 'unknown'
    
    # Combine all data
    all_data = pd.concat([bebop_df, membo_df, unknown_df], ignore_index=True)
    
    return all_data

# Analyze basic audio properties
def analyze_basic_properties(df):
    # Create output directory for plots
    os.makedirs("audio_analysis", exist_ok=True)
    
    # 1. Duration statistics
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='class', y='duration', data=df)
    plt.title('Audio Duration by Class')
    plt.ylabel('Duration (seconds)')
    plt.savefig('audio_analysis/duration_boxplot.png')
    
    # 2. Duration distribution
    plt.figure(figsize=(15, 5))
    for i, cls in enumerate(df['class'].unique()):
        plt.subplot(1, 3, i+1)
        sns.histplot(df[df['class'] == cls]['duration'], kde=True)
        plt.title(f'{cls} Duration Distribution')
        plt.xlabel('Duration (seconds)')
    plt.tight_layout()
    plt.savefig('audio_analysis/duration_distribution.png')
    
    # 3. Calculate statistical metrics
    stats_by_class = df.groupby('class')['duration'].agg(['count', 'mean', 'std', 'min', 'max'])
    
    # 4. Statistical tests
    classes = df['class'].unique()
    p_values = []
    
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            cls1, cls2 = classes[i], classes[j]
            stat, p = stats.ttest_ind(
                df[df['class'] == cls1]['duration'],
                df[df['class'] == cls2]['duration'],
                equal_var=False
            )
            p_values.append({
                'class1': cls1,
                'class2': cls2,
                't_statistic': stat,
                'p_value': p
            })
    
    p_values_df = pd.DataFrame(p_values)
    
    return stats_by_class, p_values_df

# Analyze time domain features
def analyze_time_domain(df):
    # Create a dictionary to store statistical features by class
    time_stats = defaultdict(list)
    
    # Calculate time domain features for each audio file
    for i, row in df.iterrows():
        signal = row['signal']
        
        # Calculate statistical features
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        crest_factor = peak / rms if rms > 0 else 0
        zero_crossings = librosa.feature.zero_crossing_rate(signal)[0].mean()
        
        # Calculate temporal envelope
        envelope = np.abs(librosa.stft(signal))
        envelope_mean = np.mean(envelope)
        envelope_std = np.std(envelope)
        
        # Store features
        time_stats['class'].append(row['class'])
        time_stats['rms'].append(rms)
        time_stats['peak'].append(peak)
        time_stats['crest_factor'].append(crest_factor)
        time_stats['zero_crossing_rate'].append(zero_crossings)
        time_stats['envelope_mean'].append(envelope_mean)
        time_stats['envelope_std'].append(envelope_std)
    
    # Convert to DataFrame
    time_df = pd.DataFrame(time_stats)
    
    # Create plots for time domain features
    plt.figure(figsize=(15, 12))
    
    features = ['rms', 'peak', 'crest_factor', 'zero_crossing_rate', 'envelope_mean', 'envelope_std']
    
    for i, feature in enumerate(features):
        plt.subplot(3, 2, i+1)
        sns.boxplot(x='class', y=feature, data=time_df)
        plt.title(f'{feature} by Class')
    
    plt.tight_layout()
    plt.savefig('audio_analysis/time_domain_features.png')
    
    return time_df

# Analyze frequency domain features
def analyze_frequency_domain(df, n_samples=10):
    # Sample a subset of files for spectral analysis
    sampled_df = df.groupby('class').apply(lambda x: x.sample(min(n_samples, len(x)))).reset_index(drop=True)
    
    # Calculate spectral features
    spectral_features = defaultdict(list)
    
    for i, row in sampled_df.iterrows():
        signal = row['signal']
        sr = row['sr']
        
        # Compute spectrogram
        S = np.abs(librosa.stft(signal))
        
        # Calculate spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=S)[0].mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S)[0].mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(S=S)[0].mean()
        spectral_contrast = librosa.feature.spectral_contrast(S=S).mean()
        
        # Store features
        spectral_features['class'].append(row['class'])
        spectral_features['spectral_centroid'].append(spectral_centroid)
        spectral_features['spectral_bandwidth'].append(spectral_bandwidth)
        spectral_features['spectral_rolloff'].append(spectral_rolloff)
        spectral_features['spectral_contrast'].append(spectral_contrast)
        
        # Plot spectrograms for the first few samples of each class
        if i < 3:
            plt.figure(figsize=(12, 6))
            
            # Plot waveform
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(signal, sr=sr)
            plt.title(f"Waveform: {row['class']}")
            
            # Plot spectrogram
            plt.subplot(2, 1, 2)
            librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), 
                                     y_axis='log', x_axis='time', sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Spectrogram: {row['class']}")
            
            plt.tight_layout()
            plt.savefig(f"audio_analysis/spectrogram_{row['class']}_{i}.png")
            plt.close()
    
    # Convert to DataFrame
    spectral_df = pd.DataFrame(spectral_features)
    
    # Create feature plots
    plt.figure(figsize=(15, 10))
    features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_contrast']
    
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='class', y=feature, data=spectral_df)
        plt.title(f'{feature} by Class')
    
    plt.tight_layout()
    plt.savefig('audio_analysis/spectral_features.png')
    
    return spectral_df

# Analyze harmonic vs percussive components
def analyze_harmonic_percussive(df, n_samples=5):
    plt.figure(figsize=(15, 10))
    sample_idx = 1
    
    harmonic_energy = []
    percussive_energy = []
    classes = []
    
    for cls in df['class'].unique():
        class_samples = df[df['class'] == cls].head(n_samples)
        
        for i, row in class_samples.iterrows():
            signal = row['signal']
            sr = row['sr']
            
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(signal)
            
            # Calculate energy
            harmonic_e = np.sum(y_harmonic**2)
            percussive_e = np.sum(y_percussive**2)
            
            harmonic_energy.append(harmonic_e)
            percussive_energy.append(percussive_e)
            classes.append(cls)
            
            # Plot the first few samples
            if sample_idx <= 9:
                plt.subplot(3, 3, sample_idx)
                plt.plot(y_harmonic, alpha=0.6, label='Harmonic')
                plt.plot(y_percussive, alpha=0.6, label='Percussive')
                plt.title(f"{cls} - H/P Separation")
                plt.legend()
                sample_idx += 1
    
    plt.tight_layout()
    plt.savefig('audio_analysis/harmonic_percussive_separation.png')
    
    # Create dataframe for harmonic/percussive ratio
    hp_df = pd.DataFrame({
        'class': classes,
        'harmonic_energy': harmonic_energy,
        'percussive_energy': percussive_energy,
        'harmonic_percussive_ratio': np.array(harmonic_energy) / np.array(percussive_energy)
    })
    
    # Plot harmonic/percussive ratio
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='class', y='harmonic_percussive_ratio', data=hp_df)
    plt.title('Harmonic/Percussive Ratio by Class')
    plt.savefig('audio_analysis/harmonic_percussive_ratio.png')
    
    return hp_df

# Compute mel-spectrograms and analyze them
def analyze_melspectrograms(df, n_samples=3):
    for cls in df['class'].unique():
        class_samples = df[df['class'] == cls].head(n_samples)
        
        for i, row in class_samples.iterrows():
            signal = row['signal']
            sr = row['sr']
            
            # Compute mel-spectrograms
            S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Plot mel-spectrogram
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel-Spectrogram - {cls}')
            plt.tight_layout()
            plt.savefig(f'audio_analysis/melspectrogram_{cls}_{i}.png')
            plt.close()
    
    # Extract mel-spectrogram statistics
    mel_stats = defaultdict(list)
    
    for i, row in df.iterrows():
        signal = row['signal']
        sr = row['sr']
        
        # Compute mel-spectrogram
        S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Calculate statistics
        mel_stats['class'].append(row['class'])
        mel_stats['mean'].append(np.mean(S_db))
        mel_stats['std'].append(np.std(S_db))
        mel_stats['min'].append(np.min(S_db))
        mel_stats['max'].append(np.max(S_db))
        mel_stats['range'].append(np.max(S_db) - np.min(S_db))
    
    mel_df = pd.DataFrame(mel_stats)
    
    # Plot mel-spectrogram statistics
    plt.figure(figsize=(15, 10))
    features = ['mean', 'std', 'min', 'max', 'range']
    
    for i, feature in enumerate(features):
        plt.subplot(3, 2, i+1)
        sns.boxplot(x='class', y=feature, data=mel_df)
        plt.title(f'Mel-Spectrogram {feature} by Class')
    
    plt.tight_layout()
    plt.savefig('audio_analysis/melspectrogram_stats.png')
    
    return mel_df

# Main function to run all analyses
def main():
    print("Loading audio dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} audio files")
    
    print("\nAnalyzing basic audio properties...")
    duration_stats, duration_tests = analyze_basic_properties(df)
    
    print("\nDuration statistics by class:")
    print(duration_stats)
    
    print("\nDuration statistical tests:")
    print(duration_tests)
    
    print("\nAnalyzing time domain features...")
    time_df = analyze_time_domain(df)
    
    print("\nTime domain feature statistics:")
    print(time_df.groupby('class').agg(['mean', 'std']).T)
    
    print("\nAnalyzing frequency domain features...")
    spectral_df = analyze_frequency_domain(df)
    
    print("\nSpectral feature statistics:")
    print(spectral_df.groupby('class').agg(['mean', 'std']).T)
    
    print("\nAnalyzing harmonic/percussive components...")
    hp_df = analyze_harmonic_percussive(df)
    
    print("\nHarmonic/Percussive statistics:")
    print(hp_df.groupby('class').agg(['mean', 'std']).T)
    
    print("\nAnalyzing mel-spectrograms...")
    mel_df = analyze_melspectrograms(df)
    
    print("\nMel-spectrogram statistics:")
    print(mel_df.groupby('class').agg(['mean', 'std']).T)
    
    # Save all the statistical results
    results = {
        'duration_stats': duration_stats,
        'duration_tests': duration_tests,
        'time_domain_stats': time_df.groupby('class').agg(['mean', 'std', 'min', 'max']),
        'spectral_stats': spectral_df.groupby('class').agg(['mean', 'std', 'min', 'max']),
        'harmonic_percussive_stats': hp_df.groupby('class').agg(['mean', 'std', 'min', 'max']),
        'mel_stats': mel_df.groupby('class').agg(['mean', 'std', 'min', 'max'])
    }
    
    # Save results to a pickle file
    with open('audio_analysis/statistical_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Generate a summary report
    with open('audio_analysis/summary_report.txt', 'w') as f:
        f.write("Drone Audio Dataset Statistical Analysis\n")
        f.write("======================================\n\n")
        
        f.write(f"Total samples analyzed: {len(df)}\n")
        f.write(f"Classes: {', '.join(df['class'].unique())}\n\n")
        
        f.write("Class distribution:\n")
        for cls in df['class'].unique():
            f.write(f"  {cls}: {len(df[df['class'] == cls])} samples\n")
        
        f.write("\nDuration Statistics (seconds):\n")
        f.write(duration_stats.to_string())
        
        f.write("\n\nKey Findings:\n")
        
        # Check for significant differences in duration
        for _, row in duration_tests.iterrows():
            if row['p_value'] < 0.05:
                f.write(f"  - Significant difference in duration between {row['class1']} and {row['class2']} (p={row['p_value']:.4f})\n")
        
        # Get most distinctive time domain features
        time_means = time_df.groupby('class').mean()
        for feature in time_means.columns:
            min_class = time_means[feature].idxmin()
            max_class = time_means[feature].idxmax()
            f.write(f"  - {feature}: Highest in {max_class}, lowest in {min_class}\n")
        
        f.write("\nGenerated visualizations can be found in the 'audio_analysis' directory.\n")
    
    print("\nAnalysis complete! Results saved to 'audio_analysis' directory.")
    print("See 'audio_analysis/summary_report.txt' for a summary of the findings.")

if __name__ == "__main__":
    main()