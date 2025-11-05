"""
This script trains and evaluates a classifier to distinguish between two types of press wine
(Merlot and Cabernet Sauvignon) based on GC-MS (Gas Chromatography-Mass Spectrometry) data
using Total Ion Chromatograms (TICs) as features.

Workflow summary:
1. Load GC-MS data from CSV files for Merlot (training) and Cabernet (test) samples.
2. Align sample lengths and decimate the data to reduce dimensionality.
3. Compute Total Ion Chromatograms (TICs) for each sample.
4. Normalize the TIC data (optional).
5. Assign class labels based on file identifiers.
6. Train a Ridge Classifier on the Merlot TICs.
7. Evaluate performance on Cabernet samples using standard classification metrics:
   - Accuracy, Balanced Accuracy, Weighted Accuracy
   - Precision, Recall, F1 Score
   - Normalized Confusion Matrix
8. Print all metrics and compare against chance-level performance based on class priors.

Dependencies:
- NumPy
- Scikit-learn
- Custom modules: `utils`, `wine_analysis`, and `classification`

Note:
- Designed for comparing generalization to a different varietal (domain shift).
- Data format must match expected structure in `utils.load_ms_csv_data_from_directories()`.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
import utils
from sklearn.linear_model import RidgeClassifier
from classification import assign_category_to_press_wine
from wine_analysis import GCMSDataProcessor
from utils import normalize_data, calculate_chance_accuracy_with_priors


DATA_DIRECTORY = "/Users/juliaroquette/Work/oenology/data/3D_data/PRESS_WINES/Esters22/MERLOT/"
CHEMICAL_NAME = 'PRESS_WINES_ESTERS_2022_M'
DATA_DIRECTORY_2 = "/Users/juliaroquette/Work/oenology/data/3D_data/PRESS_WINES/Esters22/CABERNET/"
CHEMICAL_NAME_2 = 'PRESS_WINES_ESTERS_2022_CS'
NORMALIZE=True
PRINT_RESULTS=True
N_DECIMATION= 5

# Load training data
row_start = 1
row_end_1, fc_idx_1, lc_idx_1 = utils.find_data_margins_in_csv(DATA_DIRECTORY)
column_indices = list(range(fc_idx_1, lc_idx_1 + 1))

data_train = utils.load_ms_csv_data_from_directories(DATA_DIRECTORY, column_indices, row_start, row_end_1)
data_test = utils.load_ms_csv_data_from_directories(DATA_DIRECTORY_2, column_indices, row_start, row_end_1)

# Ensure both datasets are trimmed to the shortest sample length
min_length = min(min(array.shape[0] for array in data_train.values()),
                 min(array.shape[0] for array in data_test.values()))
#
data_train = {key: array[:min_length, :] for key, array in data_train.items()}
data_test = {key: array[:min_length, :] for key, array in data_test.items()}

data_train = {key: matrix[::N_DECIMATION, :] for key, matrix in data_train.items()}
data_test = {key: matrix[::N_DECIMATION, :] for key, matrix in data_test.items()}


gcms_train = GCMSDataProcessor(data_train)
gcms_test = GCMSDataProcessor(data_test)

data_train = gcms_train.compute_tics()
data_test = gcms_test.compute_tics()
chromatograms_train = {key: signal for key, signal in data_train.items()}
chromatograms_test = {key: signal for key, signal in data_test.items()}

# Convert data to NumPy arrays
X_train_full = np.array(list(data_train.values()))
X_test = np.array(list(data_test.values()))

# Labels
labels_train = assign_category_to_press_wine(data_train.keys())
labels_test = assign_category_to_press_wine(data_test.keys())
y_train_full = np.array(labels_train)
y_test = np.array(labels_test)

# Normalize Data
if NORMALIZE:
    X_train_full, _= normalize_data(X_train_full)
    # X_test = scaler_train.transform(X_test)
    X_test, _= normalize_data(X_test)

# Train classifier
classifier = RidgeClassifier(alpha=1)
classifier.fit(X_train_full, y_train_full)

# Predict on test data
y_pred = classifier.predict(X_test)

# Compute Metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
weighted_acc = np.average(y_pred == y_test, weights=compute_sample_weight('balanced', y_test))
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_normalized = conf_matrix.astype(np.float64) / conf_matrix.sum(axis=1, keepdims=True)

# Print Results
if PRINT_RESULTS:
    print("Test on separate dataset metrics:")
    print(f"  Chance Accuracy: {calculate_chance_accuracy_with_priors(y_test):.3f}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Balanced Accuracy: {balanced_acc:.3f}")
    print(f"  Weighted Accuracy: {weighted_acc:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    # print(f"  Confusion Matrix:\n{conf_matrix}")
    print("  Normalized Confusion Matrix (rows sum to 1):")
    print(conf_matrix_normalized)
