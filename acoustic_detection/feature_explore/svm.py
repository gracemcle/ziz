import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

# Load the individual feature sets
print("Loading features...")
X_time = np.load('X_time_domain.npy')
X_chroma = np.load('X_chroma.npy')
X_gtcc = np.load('X_gtcc.npy')
X_mfcc = np.load('X_mfcc.npy')

# Combine the features
X_best = np.hstack((X_time, X_chroma, X_gtcc, X_mfcc))
print(f"Combined features shape: {X_best.shape}")

# Load labels
y = np.load('y.npy')
print(f"Labels shape: {y.shape}")

# Convert string labels to numeric for visualization
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
print(f"Classes: {class_names}")

# IMPORTANT: Split data with a fixed random state for consistency
X_train, X_test, y_train, y_test = train_test_split(
    X_best, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale the features properly (fit on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use cross-validation within the training set to find the best parameters
print("\nPerforming grid search with cross-validation...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
}

# Use KFold for cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid_search = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'),
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1
)

# Fit the grid search to the training data
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters
print(f"Best parameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Evaluate on the test set
y_pred = best_model.predict(X_test_scaled)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Time + Chroma + GTCC Features')
plt.colorbar()

# Set tick marks and labels
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('confusion_matrix_best.png')
plt.show()

# Visualize with PCA - using scaled data from the entire dataset for visualization only
print("\nApplying PCA for visualization...")
pca = PCA(n_components=2)
X_all_scaled = scaler.transform(X_best)  # Use the scaler fit on training data
X_pca = pca.fit_transform(X_all_scaled)

# Plot PCA with train/test markers
plt.figure(figsize=(12, 10))

# Create mask for training and test samples
train_mask = np.zeros(len(y), dtype=bool)
train_mask[np.arange(len(X_train))] = True  # This is a simplification - adjust if your split is different

# Training points
scatter_train = plt.scatter(
    X_pca[train_mask, 0], X_pca[train_mask, 1], 
    c=y_numeric[train_mask], 
    marker='o', s=40, alpha=0.7, edgecolor='k'
)

# Test points
scatter_test = plt.scatter(
    X_pca[~train_mask, 0], X_pca[~train_mask, 1], 
    c=y_numeric[~train_mask], 
    marker='x', s=50, alpha=0.9, edgecolor='k'
)

plt.title('PCA of Time + Chroma + GTCC Features')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')

# Add class legend
class_legend = plt.legend(
    handles=scatter_train.legend_elements()[0], 
    labels=class_names,
    title="Drone Classes", 
    loc="upper right"
)
plt.gca().add_artist(class_legend)

# Add train/test legend
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', markersize=10)
]
plt.legend(custom_lines, ['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.savefig('pca_best_features.png')
plt.show()

# Add a note about the potential limitation
print("\nNOTE: Features were extracted from the entire dataset before train/test splitting.")
print("For a more rigorous evaluation, consider re-extracting features separately for train and test sets.")