import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

# Define feature sets to evaluate
feature_types = ['gtcc', 'mfcc', 'lpc', 'spectral', 'time_domain', 'chroma', 'combined']

# Load labels once
y = np.load('y.npy')
y = np.array(y)

# Convert string labels to numeric values for plotting
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

scaler = StandardScaler()

# Function to evaluate a feature set
def evaluate_feature_set(feature_type, y):
    print(f"\n=== Evaluating {feature_type} features ===")
    
    # Load feature set
    X = np.load(f'X_{feature_type}.npy')
    X = np.array(X)
    
    # Count samples per class
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"{cls}: {count} samples")
    
    # Split data with 20% holdout
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Get training set class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print("\nTraining set class distribution:")
    for cls, count in zip(unique_train, counts_train):
        print(f"{cls}: {count} samples")
    
    # For undersampling, we need to ensure we don't create more samples than we have
    # Find the size of the smallest class
    min_class_idx = np.argmin(counts_train)
    min_class = unique_train[min_class_idx]
    min_samples = counts_train[min_class_idx]
    
    # Calculate target samples - make sure it's not larger than the smallest majority class
    # We'll use the same number of samples for each class to balance perfectly
    target_samples = min_samples  # Use the same number of samples for all classes
    
    # Configure the undersampler - downsample all classes to match the smallest class
    sampling_strategy = {cls: min(count, target_samples) for cls, count in zip(unique_train, counts_train)}
    
    print("\nUndersampling strategy:")
    for cls, count in sampling_strategy.items():
        print(f"{cls}: {count} samples")
    
    # Create and apply the undersampler
    undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
    
    # Print resampled class distribution
    unique_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
    print("\nResampled training set class distribution:")
    for cls, count in zip(unique_resampled, counts_resampled):
        print(f"{cls}: {count} samples")
    
    # Standardize features
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Set gamma value for RBF kernel
    gamma = 1.0 / (X.shape[1] * X.var())
    print(f"\nUsing gamma value: {gamma}")
    
    # Try different C values for regularization
    c_values = [0.1, 1, 10, 100]
    best_f1 = 0
    best_model = None
    best_c = None
    best_predictions = None
    
    print("\nTuning SVM hyperparameter C:")
    for c in c_values:
        clf = SVC(kernel='rbf', gamma=gamma, C=c)
        clf.fit(X_train_scaled, y_train_resampled)
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Use macro F1 score for imbalanced data evaluation
        f1_macro = report['macro avg']['f1-score']
        print(f"C={c}, Macro F1: {f1_macro:.4f}")
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model = clf
            best_c = c
            best_predictions = y_pred
    
    print(f"\nBest SVM hyperparameter: C={best_c}")
    print("\nClassification Report:")
    print(classification_report(y_test, best_predictions))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, best_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {feature_type.upper()} Features')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{feature_type}.png')
    plt.show()
    
    return best_f1, best_model, best_c

# Evaluate each feature set
results = {}
for feature_type in feature_types:
    best_f1, best_model, best_c = evaluate_feature_set(feature_type, y)
    results[feature_type] = {
        'f1_score': best_f1,
        'best_c': best_c
    }

# Compare results
print("\n=== Feature Set Comparison ===")
print("Feature Type | F1 Score | Best C")
print("-------------|----------|-------")
for feature_type, metrics in results.items():
    print(f"{feature_type.ljust(12)} | {metrics['f1_score']:.4f} | {metrics['best_c']}")

# Plot feature comparison
plt.figure(figsize=(12, 6))
feature_names = list(results.keys())
f1_scores = [results[feature]['f1_score'] for feature in feature_names]

# Bar plot
bars = plt.bar(feature_names, f1_scores, color='skyblue')
plt.ylim(0, 1.0)
plt.xlabel('Feature Type')
plt.ylabel('Macro F1 Score')
plt.title('Performance Comparison of Different Feature Types')

# Add value labels above bars
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('feature_comparison.png')
plt.show()

# Identify the best performing feature type
best_feature_type = max(results, key=lambda k: results[k]['f1_score'])
print(f"\nThe best performing feature type is: {best_feature_type}")

