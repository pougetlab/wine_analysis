import os
import numpy as np
from config import MULTICHANNEL, WINDOW, STRIDE, NCONV, SYNC_STATE
from classification import Classifier
import utils
from wine_analysis import GCMSDataProcessor
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from wine_analysis import WineAnalysis, ChromatogramAnalysis, GCMSDataProcessor


# Define paths for the two datasets
DATASET_1_DIR = "/Users/juliaroquette/Work/oenology/data/3D_data/220322_Pinot_Noir_Tom_CDF/"
DATASET_2_DIR = "/Users/juliaroquette/Work/oenology/data/3D_data/PINOT_NOIR/LLE_SCAN/"
N_DECIMATION = 5
DATA_TYPE = "TIS"
SYNC_STATE = True
CHROM_CAP = 29000

cl = ChromatogramAnalysis()


def pad_or_truncate_tics(tics, target_length):
    """
    Ensures that all TICs have the same length by:
    - Truncating longer TICs
    - Padding shorter TICs with zeros

    Parameters:
    ----------
    tics : dict
        Dictionary of TICs where keys are sample names and values are NumPy arrays.
    target_length : int
        The desired length of all TICs.

    Returns:
    -------
    np.ndarray
        A 2D NumPy array where all TICs have the same length.
    """
    processed_tics = []

    for tic in tics.values():
        if len(tic) > target_length:
            processed_tics.append(tic[:target_length])  # Truncate
        else:
            padded_tic = np.pad(tic, (0, target_length - len(tic)), mode='constant')  # Pad with zeros
            processed_tics.append(padded_tic)

    return np.array(processed_tics)

# Function to load and preprocess data
def load_and_compute_chromatograms(data_dir):
    """
    Loads data from a given directory and computes TICs.
    """
    row_start = 1
    row_end, fc_idx, lc_idx = utils.find_data_margins_in_csv(data_dir)
    column_indices = list(range(fc_idx, lc_idx + 1))

    # Load data
    data_dict = utils.load_ms_data_from_directories(data_dir, column_indices, row_start, row_end)

    # Trim all data to the same length
    min_length = min(array.shape[0] for array in data_dict.values())
    data_dict = {key: array[:min_length, :] for key, array in data_dict.items()}
    data_dict = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict.items()}

    # Compute TICs
    gcms = GCMSDataProcessor(data_dict)

    if DATA_TYPE == "TIC":
        if SYNC_STATE:
            tics, _ = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)
        else:
            # norm_tics = utils.normalize_dict(gcms.compute_tics(), scaler='standard')
            tics = gcms.compute_tics()
        chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in tics.items()}
    elif DATA_TYPE == "TIS":
        chromatograms = gcms.compute_tiss()


    # Normalize TICs
    chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in chromatograms.items()}

    return chromatograms



# Load datasets
signal_train = load_and_compute_chromatograms(DATASET_1_DIR)
signal_test = load_and_compute_chromatograms(DATASET_2_DIR)

# Determine the target length for all TICs (choosing the longest)
target_length = max(
    max(len(tic) for tic in signal_train.values()),
    max(len(tic) for tic in signal_test.values())
)

if DATA_TYPE == "TIC":
    # Pad or truncate TICs
    X_train = pad_or_truncate_tics(signal_train, target_length)
    X_test = pad_or_truncate_tics(signal_test, target_length)

# Convert labels
y_train = np.array(list(signal_train.keys()))
y_test = np.array(list(signal_test.keys()))

## Normalize data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train Ridge Classifier
classifier = RidgeClassifier(alpha=1.0)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Ridge Classifier Accuracy on the second dataset: {accuracy:.4f}")