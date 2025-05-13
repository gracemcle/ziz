import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

# Directory containing saved features and to save model to
output_dir = "processed_features/"

# Load all features and combine them
def load_combined_features():
    # Load individual feature sets
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
    plt.savefig(f"{output_dir}random_forest_confusion_matrix.png")
    plt.show()

print("Loading and combining all features...")
X_train, X_test, y_train, y_test = load_combined_features()

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optional: Feature selection
print("Performing feature selection...")
selector = SelectKBest(f_classif, k=min(300, X_train.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Create stratified k-fold where k = 5
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Perform 5-fold cross-validation
print("Performing 5-fold cross-validation...")
cv_scores = cross_val_score(rf, X_train_selected, y_train, cv=cv, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Perform grid search to find best hyperparameters
print("Performing grid search for optimal hyperparameters...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1), 
    param_grid, 
    cv=cv, 
    scoring='accuracy', 
    n_jobs=-1
)
grid_search.fit(X_train_selected, y_train)

print(f"Best parameters: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_

# Train model with best parameters and evaluate
print("Training final model with optimal parameters...")
best_rf.fit(X_train_selected, y_train)
y_pred = best_rf.predict(X_test_selected)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Test accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Plot confusion matrix
classes = np.unique(y_train)
plot_confusion_matrix(cm, classes, title='Random Forest Confusion Matrix')

# Save model, scaler, and selector
# print("Saving model and results...")
# with open(f"{output_dir}random_forest_model.pkl", "wb") as f:
#     pickle.dump(best_rf, f)

# with open(f"{output_dir}scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# with open(f"{output_dir}feature_selector.pkl", "wb") as f:
#     pickle.dump(selector, f)

# Save results
# results = {
#     'cv_scores': cv_scores,
#     'mean_cv_accuracy': cv_scores.mean(),
#     'test_accuracy': accuracy,
#     'best_params': grid_search.best_params_,
#     'classification_report': report
# }

# with open(f"{output_dir}random_forest_results.pkl", "wb") as f:
#     pickle.dump(results, f)

# print(f"All models and results saved to {output_dir}")
print(f"Random Forest test accuracy: {accuracy:.4f}")