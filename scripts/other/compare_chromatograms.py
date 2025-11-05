#!/usr/bin/env python
"""
compare_chromatograms.py

This script loads TIC data (chromatograms) from two different directories,
combines them into a single dataset, and processes them to produce normalized chromatograms.
It uses the wine_analysis framework and associated utilities.

Usage:
    python compare_chromatograms.py
"""

import os
import numpy as np
from wine_analysis import ChromatogramAnalysis, GCMSDataProcessor
import utils

# --- Configuration ---
# Define your two data directories (adjust these paths as needed)
DATA_DIRECTORY1 = "/Users/juliaroquette/Work/oenology/data/3D_data/PINOT_NOIR/LLE_SCAN/"
DATA_DIRECTORY2 = "/Users/juliaroquette/Work/oenology/data/3D_data/220322_Pinot_Noir_Tom_CDF/"

# Set the starting row and determine the columns to load from CSV (assuming similar structure in both)
ROW_START = 1
row_end1, fc_idx1, lc_idx1 = utils.find_data_margins_in_csv(DATA_DIRECTORY1)
column_indices1 = list(range(fc_idx1, lc_idx1 + 1))
row_end2, fc_idx2, lc_idx2 = utils.find_data_margins_in_csv(DATA_DIRECTORY2)
column_indices2 = list(range(fc_idx2, lc_idx2 + 1))

# Decimation factor and chromatogram capacity (adjust as needed)
N_DECIMATION = 5  # set to the desired decimation factor
CHROM_CAP = 29000  # example value for chromatogram capacity

# Data type and synchronization state for processing TIC data
DATA_TYPE = "TIC"  # We're processing TIC data in this script
SYNC_STATE = False  # Set True if you want to align TICs

# Custom labels for your datasets and plot title
DATASET1_LABEL = "ISVV Dataset"
DATASET2_LABEL = "Changins Dataset"
CUSTOM_TITLE = "Comparison of Chromatograms for Sample"


# ----------------------

def load_two_datasets():
    """
    Loads data from two directories and combines them.

    Returns:
        combined_data_dict (dict): Combined dictionary of data arrays.
    """
    # Load data from the first directory
    data_dict1 = utils.load_ms_data_from_directories(DATA_DIRECTORY1, column_indices1, ROW_START, row_end1)
    # Load data from the second directory
    data_dict2 = utils.load_ms_data_from_directories(DATA_DIRECTORY2, column_indices2, ROW_START, row_end2)

    # Ensure that each dictionary's arrays have consistent lengths by truncating to the minimum length
    min_length1 = min(array.shape[0] for array in data_dict1.values())
    min_length2 = min(array.shape[0] for array in data_dict2.values())
    data_dict1 = {key: array[:min_length1, :] for key, array in data_dict1.items()}
    data_dict2 = {key: array[:min_length2, :] for key, array in data_dict2.items()}

    # Apply decimation if needed
    data_dict1 = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict1.items()}
    data_dict2 = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict2.items()}

    return data_dict1, data_dict2


def process_ms_data(dataset1_data, dataset2_data):
    """
    Processes MS data to produce signals (TICs or TIS) for two separate datasets.

    Args:
        dataset1_data (dict): Dictionary of raw data arrays for dataset 1.
        dataset2_data (dict): Dictionary of raw data arrays for dataset 2.

    Returns:
        dataset1_signals (dict): Processed (and normalized) signals for dataset 1.
        dataset2_signals (dict): Processed (and normalized) signals for dataset 2.
    """
    cl = ChromatogramAnalysis()

    # Process dataset 1
    gcms1 = GCMSDataProcessor(dataset1_data)
    if DATA_TYPE == "TIC":
        if SYNC_STATE:
            tics1, _ = cl.align_tics(dataset1_data, gcms1, chrom_cap=CHROM_CAP)
        else:
            tics1 = gcms1.compute_tics()
        dataset1_chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in tics1.items()}
    elif DATA_TYPE == "TIS":
        tiss1 = gcms1.compute_tiss()
        dataset1_chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in tiss1.items()}
    else:
        raise ValueError("This script is set up for TIC data only.")

    # Process dataset 2
    gcms2 = GCMSDataProcessor(dataset2_data)
    if DATA_TYPE == "TIC":
        if SYNC_STATE:
            tics2, _ = cl.align_tics(dataset2_data, gcms2, chrom_cap=CHROM_CAP)
        else:
            tics2 = gcms2.compute_tics()
        dataset2_chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in tics2.items()}
    elif DATA_TYPE == "TIS":
        tiss2 = gcms2.compute_tiss()
        dataset2_chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in tiss2.items()}
    else:
        raise ValueError("This script is set up for TIC data only.")

    return dataset1_chromatograms, dataset2_chromatograms


def main():
    print("Loading data from directories:")
    print(f"  Directory 1: {DATA_DIRECTORY1}")
    print(f"  Directory 2: {DATA_DIRECTORY2}")
    data1, data2 = load_two_datasets()
    print(f"Loaded data 1 from {len(data1)} samples.")
    print(f"Loaded data 2 from {len(data2)} samples.")

    print("Processing MS data...")
    signal1, signal2 = process_ms_data(data1, data2)
    print("Chromatograms processed.")

    # Find common sample names (assuming keys represent sample names)
    common_samples = set(signal1.keys()).intersection(set(signal2.keys()))
    print(f"Found {len(common_samples)} common samples between the two datasets.")

    # Plot common chromatograms side by side for comparison
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    # Define a function to emphasize peaks
    def emphasize_peaks(signal, exponent=2):
        return np.power(signal, exponent)

    # Set a maximum allowed shift
    MAX_SHIFT = 5000  # adjust as needed

    # Set smoothing parameters (sigma controls the amount of smoothing)
    SMOOTH_SIGMA = 25  # adjust this value to smooth more or less

    # Get a colormap (here we use 'tab10' which provides 10 distinct colors)
    colors = plt.get_cmap('tab10').colors

    for sample in sorted(common_samples):
        # Retrieve the chromatograms from both datasets (original signals)
        chrom1 = signal1[sample]
        chrom2 = signal2[sample]

        if DATA_TYPE == "TIC":
            # Compute heavily smoothed versions of both signals
            chrom1_smoothed = gaussian_filter1d(chrom1, sigma=SMOOTH_SIGMA)
            chrom2_smoothed = gaussian_filter1d(chrom2, sigma=SMOOTH_SIGMA)


            # Use only the first third of the weighted signals for correlation estimation
            n_total = len(chrom1_smoothed)
            n_third = n_total // 3
            chrom1_part = chrom1_smoothed[:n_third]
            chrom2_part = chrom2_smoothed[:n_third]

            # Compute cross-correlation on the smoothed signals
            corr = np.correlate(chrom1_part - np.mean(chrom1_part),
                                chrom2_part - np.mean(chrom2_part), mode='full')
            # For signals of length N, zero lag is at index (N-1)
            zero_index = len(chrom1_smoothed) - 1
            shift_index = np.argmax(corr)
            optimal_shift = shift_index - zero_index

            # Limit the shift so that it does not exceed MAX_SHIFT
            if abs(optimal_shift) > MAX_SHIFT:
                optimal_shift = MAX_SHIFT * np.sign(optimal_shift)

            # Create shift-corrected versions for dataset 2
            chrom2_shifted = np.roll(chrom2, -optimal_shift)
            chrom2_smoothed_shifted = np.roll(chrom2_smoothed, -optimal_shift)

            # Plot all signals in one plot
            plt.figure(figsize=(12, 6))

            # Plot Dataset 1 original and smoothed
            plt.plot(chrom1, label=f"{DATASET1_LABEL} Original", color=colors[0])
            # plt.plot(chrom1_smoothed, linestyle='--', label=f"{DATASET1_LABEL} Smoothed", color=colors[0])

            # Plot Dataset 2 original and smoothed
            plt.plot(chrom2, label=f"{DATASET2_LABEL} Original", color=colors[1])
            # plt.plot(chrom2_smoothed, linestyle='--', label=f"{DATASET2_LABEL} Smoothed", color=colors[1])

            # # Plot the shift-corrected signals for Dataset 2
            # plt.plot(chrom2_shifted, linestyle=':', linewidth=2,
            #          label=f"{DATASET2_LABEL} Shifted by {optimal_shift} pts", color=colors[1])
            # plt.plot(chrom2_smoothed_shifted, linestyle='-.',
            #          label=f"{DATASET2_LABEL} Shifted Smoothed", color=colors[1])

            # Set title and labels
            plt.title(f"{CUSTOM_TITLE}: {sample}")
            plt.xlabel("Time")
            plt.ylabel("Normalized Intensity")
            plt.legend()
            plt.tight_layout()
        elif DATA_TYPE == "TIS":
            # For TIS data, no smoothing, correlation, or shift is applied.
            # We simply plot the signals versus m/z channel.
            # Here we assume that the signal's index corresponds to the m/z channel.
            plt.figure(figsize=(12, 6))
            plt.plot(chrom1, label=DATASET1_LABEL, color=colors[0])
            plt.plot(chrom2, label=DATASET2_LABEL, color=colors[1])
            plt.title(f"{CUSTOM_TITLE}: {sample}")
            plt.xlabel("m/z channel")
            plt.ylabel("Normalized Intensity")
            plt.legend()
            plt.tight_layout()
        else:
            raise ValueError("Invalid DATA_TYPE. Must be either 'TIC' or 'TIS'.")

        # # Maximize the figure window (for TkAgg backend)
        # mng = plt.get_current_fig_manager()
        # try:
        #     mng.window.showMaximized()  # For TkAgg backend
        # except AttributeError:
        #     mng.resize(*mng.window.maxsize())

        plt.show()

if __name__ == "__main__":
    main()