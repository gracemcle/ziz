import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Directory containing saved features
output_dir = "processed_features/"

# Load all features and combine them
def load_combined_features():
    # Load individual feature sets[1, 10],
    feature_types = ['gtcc', 'mfcc', 'time_domain', 'chroma']
    
    # Load train features and combine
    X_train_parts = []
    for feature_type in feature_types:
        features = np.load(f"{output_dir}{feature_type}_train.npy")
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        X_train_parts.append(features)
    
    # Load test features and combine
    X_test_parts = []
    for feature_type in feature_types:
        features = np.load(f"{output_dir}{feature_type}_test.npy")
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        X_test_parts.append(features)
    
    # Concatenate all features
    X_train = np.hstack(X_train_parts)
    X_test = np.hstack(X_test_parts)
    
    # Load labels
    y_train = np.load(f"{output_dir}y_train.npy")
    y_test = np.load(f"{output_dir}y_test.npy")
    
    print(f"Combined feature shape - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# Plot confusion matrix function
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{output_dir}combined_features_confusion_matrix.png")
    plt.show()

print("Loading and combining all features...")
X_train, X_test, y_train, y_test = load_combined_features()

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create SVM classifier
svm = SVC(random_state=42)

# Perform 5-fold cross-validation
print("Performing 5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=kf, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Perform randomized search with reduced parameters
print("Performing randomized parameter search (this may take a few minutes)...")
param_dist = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.1],
    'kernel': ['rbf']
}

random_search = RandomizedSearchCV(
    SVC(random_state=42), 
    param_distributions=param_dist,
    n_iter=4,  # Try only 4 combinations since grid is already small
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {random_search.best_params_}")
best_svm = random_search.best_estimator_

# Train model with best parameters and evaluate
print("Training final model with optimal parameters...")
best_svm.fit(X_train_scaled, y_train)
y_pred = best_svm.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Test accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Plot confusion matrix
classes = np.unique(y_train)
plot_confusion_matrix(cm, classes, title='Confusion Matrix - Combined Features')

# Save model and scaler
print("Saving model and results...")
with open(f"{output_dir}svm_model_combined_features.pkl", "wb") as f:
    pickle.dump(best_svm, f)

with open(f"{output_dir}scaler_combined_features.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save results
results = {
    'cv_scores': cv_scores,
    'mean_cv_accuracy': cv_scores.mean(),
    'test_accuracy': accuracy,
    'best_params': random_search.best_params_,
    'classification_report': report
}

with open(f"{output_dir}svm_results_combined.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"All models and results saved to {output_dir}")
print(f"Combined features test accuracy: {accuracy:.4f}")