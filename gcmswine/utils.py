import numpy as np
import torch
import re
import os
import subprocess
import pandas as pd
from gcmswine import utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from matplotlib import pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
import shutil
from scipy.stats import pearsonr
from itertools import combinations


def collapse_lists(d):
    """
    Collapse nested lists into a single list for each dictionary entry.

    This function is useful when the values in the dictionary are lists of lists, and you want to flatten them into a
    single list.

    Parameters
    ----------
    d : dict
        The dictionary with list or list of lists as values.

    Returns
    -------
    dict
        The dictionary with the lists collapsed into single lists.

    Raises
    ------
    ValueError
        If a value in the dictionary is not a list or a list of lists.
    """
    for key, value in d.items():
        # Check if the value is a list of lists
        if isinstance(value, list) and all(isinstance(sublist, list) for sublist in value):
            # Flatten the list of lists into a single list
            d[key] = [item for sublist in value for item in sublist]
        elif not isinstance(value, list):
            raise ValueError(f"Value for key '{key}' is not a list or a list of lists.")

    return d


def concatenate_dict_values(d1, d2):
    """
    Concatenate the values of two dictionaries.

    If a key exists in both dictionaries, their values are concatenated. If a key exists in only one dictionary, its
    value is included as is.

    Parameters
    ----------
    d1 : dict
        The first dictionary.
    d2 : dict
        The second dictionary.

    Returns
    -------
    dict
        A new dictionary with concatenated values.
    """
    result = {}

    # Union of keys from both dictionaries
    all_keys = set(d1.keys()).union(set(d2.keys()))

    for key in all_keys:
        # Concatenate values for each key
        result[key] = d1.get(key, []) + d2.get(key, [])

    return result


def find_first_and_last_position(s):
    """
    Find the positions of the first and last numbers in a string.

    This function searches for numbers in the string and returns the starting positions of the first and last occurrences.

    Parameters
    ----------
    s : str
        The input string to search for numbers.

    Returns
    -------
    tuple
        A tuple containing the start positions of the first and last numbers. If no numbers are found, returns (-1, -1).
    """
    # Define the regex pattern to find numbers
    pattern = r'\d+'

    # Find the first occurrence of the pattern
    first_match = re.search(pattern, s)
    first_pos = first_match.start() if first_match else -1

    # Find the last occurrence of the pattern
    last_match = None
    for match in re.finditer(pattern, s):
        last_match = match
    last_pos = last_match.start() if last_match else -1

    return first_pos, last_pos


def load_chromatograms(file_path, normalize=False):
    """
    Load chromatogram data from a given file path.

    Parameters
    ----------
    file_path : str
        The path to the file containing the chromatogram data.
    normalize : bool, optional
        Whether to normalize the chromatogram data. Default is False.

    Returns
    -------
    dict
        A dictionary containing the loaded chromatogram data.
    """
    from wine_analysis import WineAnalysis

    # Expand the user path to the full file path
    analysis = WineAnalysis(os.path.expanduser(file_path))

    # Return the data loaded by the WineAnalysis instance
    return analysis.data_loader.data


def min_max_normalize(data, min_range=0, max_range=1):
    """
    Perform Min-Max normalization on data.

    This function scales the input data to a specified range [min_range, max_range] using Min-Max normalization.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be normalized.
    min_range : float, optional
        The minimum value of the desired range. Default is 0.
    max_range : float, optional
        The maximum value of the desired range. Default is 1.

    Returns
    -------
    numpy.ndarray
        The normalized data within the specified range.

    Notes
    -----
    - Min-Max normalization rescales the data to a fixed range, typically [0, 1], based on the minimum and maximum values in the data.
    - This function normalizes the data across all values (i.e., sample-wise normalization) in the input array.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = min_range + ((data - min_val) * (max_range - min_range) / (max_val - min_val))

    return normalized_data


def normalize_mz_profiles_amplitude(data_dict, method="z-score"):
    """
    Normalize the amplitude of each m/z profile (column-wise normalization
    across retention times) using z-score or min-max normalization.

    Parameters:
    -----------
    data_dict : dict
        Dictionary where each key is a sample and each value is a 2D matrix
        (rows: retention times, columns: m/z values).
    method : str
        Normalization method, either "z-score" or "min-max".

    Returns:
    --------
    dict
        Dictionary with normalized matrices.
    """
    if method not in ["z-score", "min-max"]:
        raise ValueError("Invalid method. Choose 'z-score' or 'min-max'.")

    normalized_dict = {}
    for key, matrix in data_dict.items():
        # Convert matrix to NumPy array if not already
        matrix = np.array(matrix)

        if method == "z-score":
            # Z-score normalization
            col_mean = matrix.mean(axis=0, keepdims=True)  # Mean for each column (m/z profile)
            col_std = matrix.std(axis=0, keepdims=True)    # Standard deviation for each column (m/z profile)
            normalized_matrix = (matrix - col_mean) / col_std

            # Handle columns with zero standard deviation (avoid NaN)
            normalized_matrix[np.isnan(normalized_matrix)] = 0  # Replace NaN with 0

        elif method == "min-max":
            # Min-max normalization
            col_min = matrix.min(axis=0, keepdims=True)  # Min for each column (m/z profile)
            col_max = matrix.max(axis=0, keepdims=True)  # Max for each column (m/z profile)
            normalized_matrix = (matrix - col_min) / (col_max - col_min)

            # Handle columns with zero range (avoid division by zero)
            normalized_matrix[np.isnan(normalized_matrix)] = 0  # Replace NaN with 0

        # Store the normalized matrix in the dictionary
        normalized_dict[key] = normalized_matrix

    return normalized_dict


def smooth_mz_profiles(data_dict, sigma=1):
    """
    Smooth the m/z profiles using a Gaussian filter.

    Parameters:
    -----------
    data_dict : dict
        Dictionary where each key is a sample and each value is a 2D matrix
        (rows: retention times, columns: m/z values).
    sigma : float
        The standard deviation for the Gaussian kernel. Larger values result in smoother profiles.

    Returns:
    --------
    dict
        Dictionary with smoothed matrices.
    """
    smoothed_dict = {}
    for key, matrix in data_dict.items():
        # Convert to NumPy array if not already
        matrix = np.array(matrix)

        # Apply Gaussian filter along each m/z profile (column)
        smoothed_matrix = gaussian_filter1d(matrix, sigma=sigma, axis=0)

        # Store in the new dictionary
        smoothed_dict[key] = smoothed_matrix

    return smoothed_dict


def normalize_amplitude_minmax(signal):
    """
    Normalize the signal using min-max normalization.

    This method scales the input signal so that its values range between 0 and 1. The minimum value
    of the signal becomes 0, and the maximum value becomes 1.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be normalized.

    Returns
    -------
    numpy.ndarray
        The normalized signal, where the values are scaled to the range [0, 1].
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val)

    return normalized_signal


def normalize_amplitude_zscore(signal):
    """
    Normalize the signal using standard normalization (z-score normalization).

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be normalized.

    Returns
    -------
    numpy.ndarray
        The normalized signal.
    """
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    normalized_signal = (signal - mean_val) / std_val

    return normalized_signal


def normalize_data(data, scaler='standard', feature_range=(0,1)):
    """
    Normalize data which can be either a dictionary or an array.

    This function normalizes each feature (column) across all samples (rows) using the specified scaler.
    It ensures that the normalization is consistent for each feature across the dataset.

    Parameters
    ----------
    data : dict or np.ndarray
        The data to be normalized. If dict, values should be arrays. If np.ndarray, it should be a 2D array.
    scaler : str, optional
        The type of scaler to use for normalization. Options are 'standard', 'minmax', 'robust'. Default is 'standard'.

    Returns
    -------
    dict or np.ndarray
        The normalized data. If input was a dict, returns a dict with the same keys and normalized values.
        If input was an array, returns a normalized array.
    """
    values = None

    if isinstance(data, dict):
        keys = list(data.keys())
        values = np.array(list(data.values()))
    elif isinstance(data, np.ndarray):
        values = data
    else:
        raise ValueError("Input data must be either a dictionary or a numpy array.")

    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif scaler == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Unsupported scaler type. Choose from 'standard', 'minmax', or 'robust'.")

    values_scaled = scaler.fit_transform(values)

    if isinstance(data, dict):
        norm_data = {key: values_scaled[idx, :].tolist() for idx, key in enumerate(keys)}
        return norm_data, scaler
    elif isinstance(data, np.ndarray):
        return values_scaled, scaler


def normalize_dict(data, scaler='standard'):
    """
    Normalize the values in a dictionary across samples.

    This function normalizes each feature (column) in the dictionary across all samples (rows) using the specified scaler.
    The dictionary keys remain the same, while the values are normalized.

    Parameters
    ----------
    data : dict
        The input dictionary to be normalized. The values should be lists or numpy arrays representing the features.
    scaler : str, optional
        The type of scaler to use for normalization. Options are 'standard' (Z-score normalization),
        'minmax' (Min-Max scaling), and 'robust' (RobustScaler). Default is 'standard'.

    Returns
    -------
    dict
        A dictionary with the same keys and normalized values.

    Notes
    -----
    - Normalization is performed feature-wise across all samples in the dictionary, meaning each feature (value associated with a key)
      is normalized across all samples (rows) in the dataset.
    """
    keys = list(data.keys())
    values = np.array(list(data.values()))
    values_scaled = None

    if scaler == 'standard':
        scaler = StandardScaler()
        values_scaled = scaler.fit_transform(values)
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(values)
    elif scaler == 'robust':
        scaler = RobustScaler()
        values_scaled = scaler.fit_transform(values)

    norm_data = {key: values_scaled[idx, :].tolist() for idx, key in enumerate(keys)}
    return norm_data


import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def normalize_dict_multichannel(data, scaler='standard'):
    """
    Normalize the values in a dictionary across samples for each channel separately.
    The output for each sample will be a matrix where columns represent channels and rows represent values.

    Parameters
    ----------
    data : dict
        The input dictionary to be normalized. The values should be lists of numpy arrays,
        where each inner array represents a channel.
    scaler : str, optional
        The type of scaler to use for normalization. Options are 'standard' (Z-score normalization),
        'minmax' (Min-Max scaling), and 'robust' (RobustScaler). Default is 'standard'.

    Returns
    -------
    dict
        A dictionary with the same keys and normalized values, where each sample is a NumPy matrix.
        Columns represent channels, and rows represent normalized values.
    """
    keys = list(data.keys())

    # Get the number of channels from the second dimension of any sample
    first_sample = next(iter(data.values()))  # Retrieve the first sample
    if not isinstance(first_sample, np.ndarray):
        raise ValueError("Each sample in the dictionary should be a NumPy array.")
    num_channels = first_sample.shape[1]  # Number of columns (channels)

    norm_data = {}

    # Normalize each channel independently
    for channel_idx in range(num_channels):
        # Extract data for the current channel across all samples
        channel_data = np.array([sample[:, channel_idx] for sample in data.values()])

        # Choose and apply the scaler
        if scaler == 'standard':
            scaler_instance = StandardScaler()
        elif scaler == 'minmax':
            scaler_instance = MinMaxScaler()
        elif scaler == 'robust':
            scaler_instance = RobustScaler()
        else:
            raise ValueError("Unsupported scaler type. Choose 'standard', 'minmax', or 'robust'.")

        # Normalize the current channel with all samples
        channel_data_scaled = scaler_instance.fit_transform(channel_data)

        # Store normalized data back into the dictionary, grouped by sample
        for idx, key in enumerate(keys):
            if key not in norm_data:
                norm_data[key] = []
            # Append the normalized channel for the corresponding sample. Also normalize the amplitude
            # norm_data[key].append(utils.normalize_amplitude_zscore(channel_data_scaled[idx, :]))  # Ensure correct slicing
            norm_data[key].append(channel_data_scaled[idx, :])  # Ensure correct slicing

    # Convert lists of channels into matrices (columns are channels, rows are values)
    for key in norm_data:
        norm_data[key] = np.column_stack(norm_data[key])  # Stack as columns for all cases

    return norm_data


def normalize_multichannel(data, scaler='standard'):
    """
    Normalize the values across samples for each channel separately.
    The output for each sample will be a matrix where columns represent channels and rows represent values.

    Parameters
    ----------
    data : dict or np.ndarray
        If a dictionary, the values should be NumPy arrays where rows are timepoints and columns are channels.
        If an array, it should have shape (samples, timepoints, channels).
    scaler : str, optional
        The type of scaler to use for normalization. Options are 'standard' (Z-score normalization),
        'minmax' (Min-Max scaling), and 'robust' (RobustScaler). Default is 'standard'.

    Returns
    -------
    dict or np.ndarray
        If input is a dictionary, returns a dictionary with the same keys and normalized values.
        If input is an array, returns an array of the same shape with normalized values.
        Columns represent channels, and rows represent normalized values.
    """
    # Determine if input is a dictionary or array
    if isinstance(data, dict):
        keys = list(data.keys())

        # Ensure the first sample is a NumPy array
        first_sample = next(iter(data.values()))
        if not isinstance(first_sample, np.ndarray):
            raise ValueError("Each sample in the dictionary should be a NumPy array.")

        num_channels = first_sample.shape[1]  # Number of channels
        norm_data = {}

        # Normalize each channel independently
        for channel_idx in range(num_channels):
            # Extract data for the current channel across all samples
            channel_data = np.array([sample[:, channel_idx] for sample in data.values()])

            # Choose and apply the scaler
            if scaler == 'standard':
                scaler_instance = StandardScaler()
            elif scaler == 'minmax':
                scaler_instance = MinMaxScaler()
            elif scaler == 'robust':
                scaler_instance = RobustScaler()
            else:
                raise ValueError("Unsupported scaler type. Choose 'standard', 'minmax', or 'robust'.")

            # Normalize the current channel with all samples
            channel_data_scaled = scaler_instance.fit_transform(channel_data)

            # Store normalized data back into the dictionary, grouped by sample
            for idx, key in enumerate(keys):
                if key not in norm_data:
                    norm_data[key] = []
                norm_data[key].append(channel_data_scaled[idx, :])  # Ensure correct slicing

        # Convert lists of channels into matrices (columns are channels, rows are values)
        for key in norm_data:
            norm_data[key] = np.column_stack(norm_data[key])  # Stack as columns for all cases

        return norm_data

    elif isinstance(data, np.ndarray):
        if data.ndim != 3:
            raise ValueError("Input array must have shape (samples, timepoints, channels).")

        samples, timepoints, channels = data.shape
        norm_data = np.zeros_like(data)

        # Normalize each channel independently
        for channel_idx in range(channels):
            # Extract data for the current channel across all samples
            channel_data = data[:, :, channel_idx]

            # Choose and apply the scaler
            if scaler == 'standard':
                scaler_instance = StandardScaler()
            elif scaler == 'minmax':
                scaler_instance = MinMaxScaler()
            elif scaler == 'robust':
                scaler_instance = RobustScaler()
            else:
                raise ValueError("Unsupported scaler type. Choose 'standard', 'minmax', or 'robust'.")

            # Normalize the current channel
            channel_data_scaled = scaler_instance.fit_transform(channel_data.T).T  # Transpose for sample-wise normalization
            norm_data[:, :, channel_idx] = channel_data_scaled

        return norm_data

    else:
        raise TypeError("Input must be a dictionary or a 3D NumPy array.")


def filter_dict_by_keys(original_dict, keys_to_keep):
    return {key: original_dict[key] for key in keys_to_keep if key in original_dict}


def filter_dict_by_first_letter(original_dict, letters):
    return {key: original_dict[key] for key in original_dict if key[0] in letters}


def plot_data_from_dict(data_dict, title, legend=False):
    """
    Plots all lists of data contained in a dictionary.

    Parameters
    ----------
    data_dict : dict
        A dictionary with labels as keys and lists of values as values.
    title : str
        The title of the plot.
    legend : bool, optional
        Whether to display the legend. Default is False.

    Raises
    ------
    ValueError
        If the input dictionary is empty.
    """
    from itertools import cycle

    if not data_dict:
        raise ValueError("The input data dictionary is empty.")

    plt.figure(figsize=(10, 6))

    # Create a color cycle to avoid repeating colors
    colors = plt.cm.get_cmap('tab10', len(data_dict)).colors
    color_cycle = cycle(colors)

    for label, data in data_dict.items():
        plt.plot(data, label=label, color=next(color_cycle))

    plt.xlabel('Retention time')
    # plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title(title)
    if legend:
        plt.legend()
    plt.grid(True)
    plt.show()


def plot_lag(lags_loc, lags, title='Lag as a function of retention time'):
    """
    Plot the lag as a function of retention time.

    Parameters
    ----------
    lags_loc : numpy.ndarray
        Locations of the lags.
    lags : numpy.ndarray
        Lags for each datapoint.
    title : str, optional
        Title of the plot. Default is 'Lag as a function of retention time'.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags_loc, lags, label='Lag')
    plt.xlabel('Retention time')
    plt.ylabel('Lag')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def remove_peak(signal, peak_idx, window_size=5):
    """
    Smoothly removes a peak from a signal using interpolation.

    Parameters
    ----------
    signal : np.ndarray
        The input signal.
    peak_idx : int
        The index of the peak to remove.
    window_size : int, optional
        The size of the window around the peak to use for interpolation. Default is 5.

    Returns
    -------
    np.ndarray
        The signal with the peak removed.
    """
    left_idx = max(0, peak_idx - window_size)
    right_idx = min(len(signal), peak_idx + window_size + 1)

    # Interpolate over the region including the peak
    x = np.array([left_idx, right_idx])
    y = signal[x]
    interpolated_values = np.interp(np.arange(left_idx, right_idx), x, y)

    # Replace the values in the signal with the interpolated values
    signal_smooth = np.copy(signal)
    signal_smooth[left_idx:right_idx] = interpolated_values

    return signal_smooth


def calculate_chance_accuracy_with_priors(labels):
    """
    Calculate the chance accuracy with priors based on class distribution.

    This function computes the chance accuracy by using the class priors.
    The chance accuracy is the sum of squared class proportions, where the
    proportion of each class is based on its occurrence in the dataset.

    Parameters
    ----------
    labels : list of str
        A list of labels representing the classes of the samples.

    Returns
    -------
    float
        The calculated chance accuracy with priors.

    Examples
    --------
    >>> labels = ['E', 'L', 'Q', 'D', 'P', 'E', 'E', 'Q', 'D', 'P', 'L', 'E', 'R', 'X', 'M']
    >>> calculate_chance_accuracy_with_priors(labels)
    0.06842374493702534
    """
    # Create a Counter object to count occurrences of each class
    class_counts = Counter(labels)

    # Calculate the total number of samples
    total_samples = sum(class_counts.values())

    # Calculate chance accuracy by summing the square of the class proportions
    chance_accuracy = sum((count / total_samples) ** 2 for count in class_counts.values())

    return chance_accuracy


# def load_ms_csv_data_from_directories(directory, columns, row_start, row_end):
#     """
#     Reads CSV files from all .D directories in the specified directory and extracts specific columns and row ranges.
#
#     Args:
#         directory (str): The path to the main directory containing .D directories.
#         columns (list of int): A list of column indices to extract from each CSV file.
#         row_start (int): The starting row index to extract (inclusive).
#         row_end (int): The ending row index to extract (exclusive).
#
#     Returns:
#         dict of numpy arrays: A dictionary where each key is a .D directory name (without the .D suffix),
#                               and each value is a numpy array containing the extracted data from each CSV file.
#     """
#     data_dict = {}
#
#     # Loop through all .D directories in the specified directory
#     for subdir in sorted(os.listdir(directory)):
#         if subdir.endswith('.D'):
#             # Remove the '.D' part for use as the dictionary key
#             dir_name = subdir[:-2]
#
#             # Construct the expected CSV file path
#             # csv_file_path = os.path.join(directory, subdir, f"{dir_name}_MS.csv")
#             csv_file_path = os.path.join(directory, subdir, f"{dir_name}.csv")
#
#             # Check if the expected CSV file exists
#             if os.path.isfile(csv_file_path):
#                 try:
#                     # Read the CSV file with pandas
#                     df = pd.read_csv(csv_file_path)
#
#                     # Select the specified columns and row range
#                     extracted_data = df.iloc[row_start:row_end, columns].to_numpy()
#
#                     # Store the extracted data in the dictionary with the directory name as the key
#                     data_dict[dir_name] = extracted_data
#
#                 except Exception as e:
#                     print(f"Error processing file {csv_file_path}: {e}")
#             else:
#                 print(f"No matching CSV file found in {subdir}.")
#
#     return data_dict

def load_ms_csv_data_from_directories(directory, columns, row_start, row_end):
    """
    Reads CSV files from all .D directories in the specified directory and extracts specific columns and row ranges.
    Uses .npy caching to speed up repeated loading.

    Args:
        directory (str): The path to the main directory containing .D directories.
        columns (list of int): A list of column indices to extract from each CSV file.
        row_start (int): The starting row index to extract (inclusive).
        row_end (int): The ending row index to extract (exclusive).

    Returns:
        dict of numpy arrays: A dictionary where each key is a .D directory name (without the .D suffix),
                              and each value is a numpy array containing the extracted data from each CSV file.
    """
    import os
    import numpy as np
    import pandas as pd

    data_dict = {}

    # Loop through all .D directories in the specified directory
    for subdir in sorted(os.listdir(directory)):
        if subdir.endswith('.D'):
            dir_name = subdir[:-2]  # without '.D'
            csv_file_path = os.path.join(directory, subdir, f"{dir_name}.csv")
            cache_file_path = os.path.join(directory, subdir, f"{dir_name}_cached.npy")

            if os.path.isfile(cache_file_path):
                try:
                    extracted_data = np.load(cache_file_path)
                    data_dict[dir_name] = extracted_data
                    print(f"Loaded cached data from {cache_file_path}")
                except Exception as e:
                    print(f"Error loading cache for {dir_name}: {e}")
            elif os.path.isfile(csv_file_path):
                try:
                    df = pd.read_csv(csv_file_path)
                    extracted_data = df.iloc[row_start:row_end, columns].to_numpy()
                    np.save(cache_file_path, extracted_data)
                    data_dict[dir_name] = extracted_data
                    print(f"Processed and cached data from {csv_file_path}")
                except Exception as e:
                    print(f"Error processing file {csv_file_path}: {e}")
            else:
                print(f"No matching CSV file found in {subdir}.")

    return data_dict

def find_data_margins_in_csv(directory):
    """
    Finds the first CSV file in a .D directory within the specified directory,
    returns the number of rows, and identifies the indices of the first and last columns
    with integer headers, or handles the special case of 2-column files.

    Args:
        directory (str): Path to the main directory containing .D directories.

    Returns:
        tuple: (int, int, int) where:
            - The first element is the row count in the CSV,
            - The second element is the index of the first relevant column,
            - The third element is the index of the last relevant column.
        Returns (None, None, None) if no valid CSV file is found.
    """
    for subdir in sorted(os.listdir(directory)):
        if subdir.endswith('.D'):
            dir_name = subdir[:-2]
            csv_file_path = os.path.join(directory, subdir, f"{dir_name}.csv")

            if os.path.isfile(csv_file_path):
                try:
                    # Load the CSV to get row count
                    df_full = pd.read_csv(csv_file_path)
                    row_count = len(df_full) + 1  # +1 includes header row

                    # Load a few rows to inspect the columns
                    df = pd.read_csv(csv_file_path, nrows=5)
                    col_names = df.columns

                    if len(col_names) == 2:
                        # Special case: only 2 columns (e.g., Ret. Time, TIC)
                        return row_count, 1, 1

                    # General case: look for integer-named columns
                    integer_columns = [i for i, col in enumerate(col_names) if col.isdigit()]

                    if integer_columns:
                        return row_count, integer_columns[0], integer_columns[-1]

                except Exception as e:
                    print(f"Error processing file {csv_file_path}: {e}")
                    return None, None, None

    print("No CSV file found in any .D directory.")
    return None, None, None


def sum_data_in_data_dict(data_dict, axis=1):
    """
    Takes a dictionary with arrays as values and returns a new dictionary with the same keys.
    Depending on the axis parameter, each value is a list of either row sums or column sums.

    Args:
        data_dict (dict): A dictionary where keys are directory names, and values are 2D numpy arrays.
        axis (int): Axis along which to sum:
                    - 1 for summing all columns in a row (default)
                    - 0 for summing all rows in a column
    Returns:
        dict: A dictionary with the same keys as data_dict. Each value is a list of sums along the specified axis.
    """
    sum_dict = {}

    # Loop through each key and array in the input dictionary
    for key, array in data_dict.items():
        # Calculate the sum along the specified axis and convert to a list
        sums = array.sum(axis=axis).tolist()
        # Store the list of sums in the new dictionary
        sum_dict[key] = sums

    return sum_dict


def string_to_latex_confusion_matrix(data_str, headers):
    # Convert string to numpy array
    data_str = re.sub(r'\s+', ' ', data_str.replace('\n', ' '))  # Clean up whitespace
    data = np.array([list(map(float, row.split())) for row in data_str[2:-2].split('] [')])

    # Multiply by 100 and convert to integer
    # data = np.round(data * 100).astype(int)
    data = np.ceil(data * 100).astype(int)

    # Begin LaTeX table string
    latex_string = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|c|" + "c|" * len(headers) + "}\n    \\hline\n"

    # Add column headers with rotated labels
    latex_string += "    & " + " & ".join(f"\\rotatebox{{90}}{{{header}}}" for header in headers) + " \\\\\\hline\n"

    # Populate rows with cell color and no display value
    for i, row in enumerate(data):
        row_name = headers[i]
        row_cells = " & ".join(f"\\cellcolorval{{{value}}}" for value in row)
        latex_string += f"    {row_name} & {row_cells} \\\\\\hline\n"

    # Complete LaTeX table
    latex_string += "\\end{tabular}\n\\caption{Confusion Matrix in LaTeX}\n\\end{table}"

    # Print without escape characters
    print(latex_string)


def string_to_latex_confusion_matrix_modified(data_str, headers):
    """Header labels at the bottom"""

    def round_row_to_100_percent(row):
        """Round a row of probabilities to integers summing to 100."""
        scaled = row * 100
        floored = np.floor(scaled).astype(int)
        remainder = int(100 - floored.sum())

        # Distribute the remaining +1s to the highest decimals
        decimal_parts = scaled - floored
        top_indices = np.argsort(-decimal_parts)[:remainder]
        floored[top_indices] += 1
        return floored

    # Clean up and convert string to numpy array
    data_str = re.sub(r'\s+', ' ', data_str.replace('\n', ' '))  # Clean up whitespace
    data = np.array([list(map(float, row.split())) for row in data_str[2:-2].split('] [')])
    # data = np.round(data * 100).astype(int)  # Scale to percentages
    data = np.array([round_row_to_100_percent(row) for row in data])
    n = len(headers)

    latex = []
    latex.append("\\begin{table}[h!]")
    latex.append("    \\centering")
    latex.append("    \\setlength{\\tabcolsep}{3pt}")
    latex.append(f"    \\begin{{tabular}}{{l{'|c' * n}|}}")
    latex.append("        \\hline")

    # Add data rows
    for i, row in enumerate(data):
        row_cells = " & ".join(f"\\cellcolorval{{{val}}}" for val in row)
        latex.append(f"        {headers[i]} & {row_cells} \\\\\\hline")

    # Space before bottom headers
    latex.append("        \\\\[-3.8ex]")
    latex.append(f"        \\multicolumn{{{n + 1}}}{{c}}{{}} \\\\[-1.9ex]")

    # Bottom column headers
    header_row = " & " + " & ".join(f"\\rotatebox{{90}}{{{label}}}" for label in headers) + " \\\\"
    latex.append(f"        {header_row}")

    # Close tabular and table
    latex.append("    \\end{tabular}")
    latex.append(
        "    \\caption{Normalized confusion matrix with RGC classifier and TIC+TIS feature vectors for the Changins dataset. Rows represent the true labels, columns the predicted labels. Cell values are percentage of samples assigned to each predicted label. Darker blue = stronger prediction confidence.}")
    latex.append("\\end{table}")

    print("\n".join(latex))


def string_to_latex_correlation_matrix(data_str, headers):
    """
    Converts a correlation matrix string to a LaTeX table with \cellcolorval highlighting.

    Parameters:
    ----------
    data_str : str
        The correlation matrix as a string (e.g., from np.array2string()).
    headers : list of str
        List of column and row headers.

    Returns:
    -------
    None
        Prints the LaTeX-formatted table with \cellcolorval for values in [0, 1].
    """
    # Cleanup and parse matrix from string to numpy array
    data_str = re.sub(r'\s+', ' ', data_str.replace('\n', ' '))
    data = np.array([list(map(float, row.split())) for row in data_str[2:-2].split('] [')])

    # Begin LaTeX table string
    latex = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|c|" + "c|" * len(headers) + "}\n\\hline\n"

    # Add headers
    latex += " & " + " & ".join(f"\\rotatebox{{90}}{{{h}}}" for h in headers) + " \\\\\n\\hline\n"

    # Populate rows with \cellcolorval and actual values
    for i, row in enumerate(data):
        row_label = headers[i]
        formatted_row = " & ".join(f"\\cellcolorval{{{v:.3f}}} {v:.3f}" for v in row)
        latex += f"{row_label} & {formatted_row} \\\\\n\\hline\n"

    latex += "\\end{tabular}\n\\caption{Correlation Matrix with Heatmap Coloring}\n\\end{table}"
    print(latex)


def string_to_latex_colored_matrix(data_str, headers, round_decimals=3):
    """
    Converts a correlation matrix string to a LaTeX table with dynamic cell coloring and font color for readability.

    Parameters:
    -----------
    data_str : str
        Correlation matrix as a string from np.array2string().
    headers : list of str
        Labels for rows/columns.
    round_decimals : int
        Number of decimals to display in the cell.
    """
    import re
    import numpy as np

    # Parse and clean string matrix
    data_str = re.sub(r'\s+', ' ', data_str.replace('\n', ' '))
    data = np.array([list(map(float, row.split())) for row in data_str[2:-2].split('] [')])

    # Begin LaTeX table
    latex = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|c|" + "c|" * len(headers) + "}\n\\hline\n"
    latex += " & " + " & ".join(f"\\rotatebox{{90}}{{{h}}}" for h in headers) + " \\\\\n\\hline\n"

    for i, row in enumerate(data):
        label = headers[i]
        cells = []
        for val in row:
            colorval = int(round(val * 100))
            text_color = ""
            # text_color = "\\textcolor{white}" if colorval > 70 else ""
            cell = f"\\cellcolor{{lightblue!{colorval}!white}} {text_color}{{{val:.{round_decimals}f}}}"
            cells.append(cell)
        latex += f"{label} & {' & '.join(cells)} \\\\\n\\hline\n"

    latex += "\\end{tabular}\n\\caption{Correlation Matrix with Heatmap Coloring}\n\\end{table}"

    print(latex)




def plot_image(image):
    """
    Plot a 1-channel or 3-channel image (either a torch.Tensor or a numpy.ndarray) using Matplotlib.

    Parameters:
    -----------
    image : torch.Tensor or numpy.ndarray
        A 1-channel or 3-channel image of shape (1, H, W), (3, H, W), (H, W), or (H, W, 3).
    """
    # Handle PyTorch tensor input
    if isinstance(image, torch.Tensor):
        # Ensure the image is on the CPU and not batched
        if len(image.shape) == 4:  # Handle batch dimension
            image = image[0]
        image = image.cpu()  # Move to CPU if needed

        if image.shape[0] == 3:  # Convert 3-channel tensor to (H, W, 3)
            image = image.permute(1, 2, 0).numpy()
        elif image.shape[0] == 1:  # Convert 1-channel tensor to (H, W)
            image = image.squeeze(0).numpy()
        else:
            raise ValueError("Input tensor should have shape (1, H, W), (3, H, W), or (N, 3/1, H, W).")

    # Handle NumPy array input
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[-1] == 3:  # (H, W, 3)
            pass  # Already in the correct format
        elif len(image.shape) == 2:  # (H, W)
            pass  # Already single-channel
        else:
            raise ValueError("Input numpy array should have shape (H, W) or (H, W, 3).")
    else:
        raise TypeError("Input should be a torch.Tensor or numpy.ndarray.")

    # De-normalize if needed (for 3-channel images, using ImageNet stats)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if len(image.shape) == 3:  # (H, W, 3)
        if image.max() > 1.0:  # Assume it's in the [0, 255] range
            image = image / 255.0
        image = (image * std) + mean  # De-normalize
        image = np.clip(image, 0, 1)  # Clip values to [0, 1] for display

    # Plot the image
    if len(image.shape) == 3:  # RGB Image
        plt.imshow(image)
    elif len(image.shape) == 2:  # Grayscale Image
        plt.imshow(image, cmap="gray")  # Use grayscale colormap
    else:
        raise ValueError("Unexpected image format for plotting.")

    plt.axis('off')  # Turn off axes
    plt.show()


def aggregate_retention_times(data, window_size, method="mean"):
    """
    Aggregate GCMS data along the retention time dimension.

    Parameters:
    ----------
    data : numpy.ndarray
        A 2D array with shape (retention time, m/z).
    window_size : int
        The number of retention times to aggregate into one.
    method : str
        The aggregation method: "mean", "max", or "median".

    Returns:
    -------
    aggregated_data : numpy.ndarray
        A 2D array with reduced retention time dimension.
    """
    # Ensure the number of retention times is divisible by the window size
    n_retention_times = data.shape[0]
    truncated_data = data[:n_retention_times // window_size * window_size, :]

    # Reshape for aggregation
    reshaped_data = truncated_data.reshape(-1, window_size, data.shape[1])  # (num_windows, window_size, m/z)

    # Perform aggregation
    if method == "mean":
        aggregated_data = reshaped_data.mean(axis=1)  # Aggregate along the window size dimension
    elif method == "max":
        aggregated_data = reshaped_data.max(axis=1)
    elif method == "median":
        aggregated_data = np.median(reshaped_data, axis=1)
    else:
        raise ValueError("Invalid method. Choose 'mean', 'max', or 'median'.")

    return aggregated_data


def split_tensor_into_overlapping_windows(tensor, window_size, stride):
    """
    Split the last dimension of a tensor into overlapping windows.

    Parameters:
    ----------
    tensor : torch.Tensor
        The input tensor of shape (batch_size, channels, length, 1).
    window_size : int
        The size of each window along the last dimension.
    stride : int
        The stride (step size) for overlapping windows.

    Returns:
    -------
    torch.Tensor
        A tensor of shape (batch_size * num_overlaps, channels, window_size).
    """
    # Ensure the input tensor is 4D
    if tensor.ndim != 4:
        raise ValueError("Input tensor must be 4D (batch_size, channels, length, 1)")

    # Calculate the number of overlaps
    num_overlaps = (tensor.shape[2] - window_size) // stride + 1

    # Apply unfolding to create windows
    windows = torch.nn.functional.unfold(
        tensor.permute(0, 3, 1, 2),  # Permute to (batch_size, 1, channels, length)
        kernel_size=(tensor.shape[1], window_size),  # Window size for channels and length
        stride=(1, stride)  # Stride for channels and length
    )

    # Reshape and permute back to the desired format
    windows = windows.permute(0, 2, 1).reshape(-1, tensor.shape[1], window_size)

    return windows, num_overlaps

# def reduce_columns_in_dict(matrices_dict, n):
#     """
#     Reduces the column dimension of matrices in a dictionary by summing n contiguous columns.
#     Excess columns are added to the last column of the reduced matrix if the total is not divisible by n.
#
#     Parameters:
#         matrices_dict (dict): Dictionary of matrices (key: name, value: numpy.ndarray).
#                               Each matrix should have the same number of columns.
#         n (int): Number of contiguous columns to sum.
#
#     Returns:
#         dict: A new dictionary with reduced-dimension matrices.
#     """
#     reduced_dict = {}
#     for key, matrix in matrices_dict.items():
#         rows, cols = matrix.shape
#         full_groups = cols // n
#
#         # Handle the main contiguous columns
#         reshaped_data = matrix[:, :full_groups * n].reshape(rows, full_groups, n)
#         reduced_matrix = reshaped_data.sum(axis=2)
#
#         # Handle excess columns
#         if cols % n != 0:
#             excess_columns = matrix[:, full_groups * n:].sum(axis=1, keepdims=True)
#             # Add excess to the last column of the reduced matrix
#             reduced_matrix[:, -1] += excess_columns.flatten()
#
#         reduced_matrix = np.apply_along_axis(utils.normalize_amplitude_zscore, axis=0, arr=reduced_matrix)
#
#         # Store in the result dictionary
#         reduced_dict[key] = reduced_matrix
#
#     return reduced_dict


def reduce_columns_in_dict(matrices_dict, n, normalize=False):
    """
    Reduces the column dimension of matrices in a dictionary by summing n contiguous columns.
    Excess columns are added to the last column of the reduced matrix if the total is not divisible by n.

    Parameters:
        matrices_dict (dict): Dictionary of matrices (key: name, value: numpy.ndarray).
                              Each matrix should have the same number of columns.
        n (int): Number of contiguous columns to sum.

    Returns:
        dict: A new dictionary with reduced-dimension matrices.
    """
    reduced_dict = {}
    for key, matrix in matrices_dict.items():
        rows, cols = matrix.shape
        full_groups = cols // n

        # Handle the main contiguous columns
        reshaped_data = matrix[:, :full_groups * n].reshape(rows, full_groups, n)
        reduced_matrix = reshaped_data.sum(axis=2)

        # Handle excess columns
        if cols % n != 0:
            excess_columns = matrix[:, full_groups * n:].sum(axis=1, keepdims=True)
            # Add excess to the last column of the reduced matrix
            reduced_matrix[:, -1] += excess_columns.flatten()

        if normalize:
            reduced_matrix = np.apply_along_axis(utils.normalize_amplitude_zscore, axis=0, arr=reduced_matrix)

        # Store in the result dictionary
        reduced_dict[key] = reduced_matrix

    return reduced_dict

def reduce_columns_to_final_channels(matrix, final_channels):
    """
    Reduces the column dimension of a 3D array to a specified number of final channels by averaging
    contiguous elements across the last dimension. Excess elements are added to the last group's average.

    Parameters:
        matrix (numpy.ndarray): A 3D array to process (shape: samples x features x channels).
        final_channels (int): The desired number of channels after reduction.

    Returns:
        numpy.ndarray: A reduced-dimension 3D array.
    """
    samples, features, channels = matrix.shape

    # Calculate the number of channels to group (n) and handle any remaining channels
    n = channels // final_channels  # Number of contiguous channels to average per group
    remaining = channels % final_channels  # Remaining channels to add to the last group

    # Initialize the reduced matrix
    reduced_matrix = np.zeros((samples, features, final_channels))

    # Aggregate channels into groups
    for i in range(final_channels):
        start_idx = i * n
        if i == final_channels - 1:  # Include the remaining channels in the last group
            end_idx = channels
        else:
            end_idx = (i + 1) * n
        reduced_matrix[:, :, i] = np.mean(matrix[:, :, start_idx:end_idx], axis=-1)

    return reduced_matrix


import netCDF4
import csv
import os
import subprocess


def convert_ms_files_to_cdf_in_place(root_folder):
    """
    Recursively converts all .ms files in the root_folder (including subdirectories)
    to .cdf format using ProteoWizard's msconvert and saves the output in the same folder.

    Parameters:
        root_folder (str): Path to the directory containing .ms files.

    Returns:
        None
    """
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith("data.ms"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, file.replace(".ms", ".cdf"))

                # Run msconvert command
                command = [
                    "msconvert", input_path,
                    "--outdir", root,  # Save in the same folder
                    "--mzML"
                ]

                try:
                    subprocess.run(command, check=True)
                    print(f"Converted: {input_path} -> {output_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error converting {input_path}: {e}")


def convert_cdf_to_csv(cdf_file, csv_file, mz_min=40, mz_max=220):
    """
    Converts a CDF file to a CSV file with retention time and m/z intensity channels.

    Parameters:
    cdf_file (str): Path to the input CDF file.
    csv_file (str): Path to the output CSV file.
    mz_min (int): Minimum m/z value for the channels.
    mz_max (int): Maximum m/z value for the channels.
    """
    try:
        # Open the NetCDF file
        dataset = netCDF4.Dataset(cdf_file, mode='r')

        # Retrieve necessary variables
        retention_times = dataset.variables['scan_acquisition_time'][:] * 1000  # Convert to milliseconds
        mass_values = dataset.variables['mass_values'][:]
        intensity_values = dataset.variables['intensity_values'][:]
        point_counts = dataset.variables['point_count'][:]

        # Define m/z range (integer values between mz_min and mz_max)
        mz_range = np.arange(mz_min, mz_max + 1)  # Include mz_max

        # Prepare the output matrix
        # Rows: Retention times, Columns: Retention time + m/z channels
        output_matrix = np.zeros((len(retention_times), len(mz_range) + 1), dtype=float)
        output_matrix[:, 0] = retention_times  # First column is retention time

        # Precompute all rounded m/z values and corresponding intensity values
        mz_rounded_all = np.round(mass_values).astype(int)

        # Precompute valid m/z indices (those in range [mz_min, mz_max])
        valid_mask = (mz_rounded_all >= mz_min) & (mz_rounded_all <= mz_max)
        mz_indices = mz_rounded_all[valid_mask] - mz_min  # Map valid m/z to column indices
        valid_intensities = intensity_values[valid_mask]

        # Create a pointer for the current position in point_counts
        current_index = 0

        # Iterate over scans and aggregate intensities
        for i, point_count in enumerate(point_counts):
            # Get the range for this scan
            scan_range = slice(current_index, current_index + point_count)

            # Apply the valid mask for this scan
            scan_mz_values = mz_rounded_all[scan_range]
            scan_intensities = intensity_values[scan_range]

            # Apply filter to get valid m/z and intensities
            valid_scan_mask = (scan_mz_values >= mz_min) & (scan_mz_values <= mz_max)
            mz_indices_scan = scan_mz_values[valid_scan_mask] - mz_min
            intensities_scan = scan_intensities[valid_scan_mask]

            # Use NumPy's add.at for fast accumulation
            np.add.at(output_matrix[i, 1:], mz_indices_scan, intensities_scan)

            # Move to the next scan range
            current_index += point_count

        # Write the output matrix to a CSV file
        with open(csv_file, mode='w', newline='') as csv_output:
            writer = csv.writer(csv_output)

            # Write the header
            header = ['Retention Time (ms)'] + [f"{mz}" for mz in mz_range]
            writer.writerow(header)

            # Write the rows of data
            writer.writerows(output_matrix)

        print(f"Successfully converted {cdf_file} to {csv_file}")

    except KeyError as ke:
        print(f"Key Error: {ke}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the dataset
        dataset.close()



def convert_cdf_directory_to_csv(input_dir, mz_min=40, mz_max=220):
    """
    Converts all CDF files in *.D directories to CSV files in the same directories.

    Parameters:
    input_dir (str): Path to the root directory containing *.D subdirectories with CDF files.
    mz_min (int): Minimum m/z value for the channels.
    mz_max (int): Maximum m/z value for the channels.
    """
    # Walk through all subdirectories in the input directory
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        # Check if the subdirectory is a *.D directory
        if os.path.isdir(subdir_path) and subdir.endswith('.D'):
            cdf_file = os.path.join(subdir_path, 'data.cdf')  # Path to the data.cdf file
            if os.path.isfile(cdf_file):
                # Generate the output CSV file name
                csv_file = os.path.join(subdir_path, f"{os.path.splitext(subdir)[0]}.csv")

                # Convert the CDF file to CSV
                try:
                    print(f"Converting {cdf_file} to {csv_file}...")
                    convert_cdf_to_csv(cdf_file, csv_file, mz_min=mz_min, mz_max=mz_max)
                except Exception as e:
                    print(f"Failed to convert {cdf_file}: {e}")



def plot_aggregated_weights(weights, bin_size=10000, class_index=None):
    """
    Aggregates feature weights into bins and plots the average weight per bin.

    Parameters
    ----------
    weights : np.ndarray
        Array of feature weights (1D array for binary classification or row for a specific class in multi-class classification).
    bin_size : int
        Number of features per bin for aggregation.
    class_index : int or None
        Index of the class for labeling (used only for multi-class classification).
    """
    # Ensure weights is a 1D array
    weights = np.asarray(weights)

    # Calculate number of bins
    num_bins = len(weights) // bin_size
    remainder = len(weights) % bin_size

    # Aggregate weights by bins
    binned_weights = np.mean(weights[:num_bins * bin_size].reshape(num_bins, bin_size), axis=1)

    # Handle remainder features if they exist
    if remainder > 0:
        remainder_avg = np.mean(weights[num_bins * bin_size:])
        binned_weights = np.append(binned_weights, remainder_avg)

    # Plot aggregated weights
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(binned_weights)), binned_weights, width=0.8)
    plt.xlabel("Feature Bins")
    plt.ylabel("Average Weight Value")
    title = "Aggregated Feature Weights by Bin"
    if class_index is not None:
        title += f" (Class {class_index})"
    plt.title(title)
    plt.show()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def compute_channel_correlation_single_sample(data, sample_index, figsize=(8, 6), cmap="coolwarm"):
    """
    Computes and visualizes the correlation matrix for m/z channels within a single sample.

    Parameters
    ----------
    data : np.ndarray
        Input 3D array with shape (samples, features, channels).
    sample_index : int
        Index of the sample to analyze.
    figsize : tuple
        Size of the heatmap figure (default: (8, 6)).
    cmap : str
        Colormap for the heatmap (default: "coolwarm").

    Returns
    -------
    np.ndarray
        Correlation matrix of shape (channels, channels) for the selected sample.
    """
    # Step 1: Extract the data for the specific sample
    sample_data = data[sample_index]  # Shape: (features, channels)
    print(f"Shape of sample_data: {sample_data.shape}")  # Should be (features, channels)

    # Step 2: Compute the correlation matrix for the channels within this sample
    try:
        correlation_matrix = np.corrcoef(sample_data, rowvar=False)  # Shape: (channels, channels)
    except Exception as e:
        raise ValueError(f"Error computing correlation matrix: {e}")

    # Step 3: Plot the correlation matrix
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, cmap=cmap, center=0, square=True, annot=False, fmt=".2f")
    plt.title(f"Correlation Between m/z Channels (Sample {sample_index})")
    plt.xlabel("Channels")
    plt.ylabel("Channels")
    plt.show()

    return correlation_matrix


def split_train_val_test(data, labels, test_size=0.2, validation_size=0.2, random_seed=42):
    """
    Splits the dataset into train, validation, and test sets.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix.
    labels : np.ndarray
        Corresponding labels.
    test_size : float
        Proportion of data to reserve for the test set.
    validation_size : float
        Proportion of the training data to reserve for the validation set.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_train : np.ndarray
        Training features.
    X_val : np.ndarray
        Validation features.
    X_test : np.ndarray
        Test features.
    y_train : np.ndarray
        Training labels.
    y_val : np.ndarray
        Validation labels.
    y_test : np.ndarray
        Test labels.
    """
    # First, split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data, labels, test_size=test_size, stratify=labels, random_state=random_seed
    )

    # Then, split the remaining data into train and validation sets
    val_size_adjusted = validation_size / (1 - test_size)  # Adjust validation size proportion
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, stratify=y_train_val, random_state=random_seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def copy_files_to_matching_directories(source_dir, mother_dir):
    """
    Reads files from source_dir and copies each file into a directory inside
    mother_dir (or its subdirectories) that matches the file's name (without extension, ignoring '.D').

    If no matching directory is found, a new directory named 'file.D' is created at the root.

    Args:
        source_dir (str): Path to the directory containing files to be copied.
        mother_dir (str): Path to the mother directory where files will be organized.
    """

    def find_matching_directory(folder_name, mother_dir):
        """
        Recursively search for a directory named 'folder_name.D' within 'mother_dir'.
        Returns the path if found, otherwise None.
        """
        for root, dirs, _ in os.walk(mother_dir):
            for directory in dirs:
                # Ignore case and check if directory ends with .D and matches folder_name
                if directory.lower().endswith(".d") and directory[:-2] == folder_name:
                    return os.path.join(root, directory)
        return None

    # Ensure the mother directory exists
    os.makedirs(mother_dir, exist_ok=True)

    # Loop through each file in the source directory
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)

        # Only process files
        if os.path.isfile(file_path):
            # Extract filename without extension
            folder_name = os.path.splitext(file_name)[0]

            # Search for a matching directory recursively (ignoring '.D')
            destination_folder = find_matching_directory(folder_name, mother_dir)

            # If no directory is found, create one at the root of mother_dir with .D suffix
            if destination_folder is None:
                destination_folder = os.path.join(mother_dir, folder_name + ".D")
                os.makedirs(destination_folder, exist_ok=True)

            # Copy the file to the destination folder
            destination_path = os.path.join(destination_folder, file_name)
            shutil.copy(file_path, destination_path)
            print(f"Copied: {file_name} -> {destination_folder}")


# def remove_zero_variance_channels(data_dict):
#     """
#     Removes channels with zero variance across all samples in the dataset.
#
#     Parameters:
#         data_dict (dict): Dictionary where keys are sample IDs and values are NumPy arrays
#                           of shape (timepoints, num_channels).
#
#     Returns:
#         filtered_data_dict (dict): Dictionary with zero-variance channels removed.
#         valid_channels (list): Indices of retained channels.
#     """
#     # Stack all sample arrays along a new axis: shape (num_samples, timepoints, num_channels)
#     all_samples = np.array(list(data_dict.values()))  # Shape: (num_samples, timepoints, num_channels)
#
#     # Compute variance for each channel across all timepoints and samples
#     channel_variances = np.var(all_samples, axis=(0, 1))  # Shape: (num_channels,)
#
#     # Identify valid channels (non-zero variance)
#     valid_channels = np.where(channel_variances > 1e-8)[0]  # Indices of channels to keep
#
#     # Remove zero-variance channels from each sample
#     filtered_data_dict = {
#         sample_id: data[:, valid_channels] for sample_id, data in data_dict.items()
#     }
#
#     return filtered_data_dict, valid_channels

def remove_zero_variance_channels(data_dict):
    """
    Removes channels with zero variance across all samples in the dataset.

    Parameters:
        data_dict (dict): Dictionary where keys are sample IDs and values are NumPy arrays
                          of shape (timepoints, num_channels).

    Returns:
        filtered_data_dict (dict): Dictionary with zero-variance channels removed.
        valid_channels (list): Indices of retained channels.
    """
    sample_shapes = [v.shape for v in data_dict.values()]
    num_channels = sample_shapes[0][1] if sample_shapes else 0

    if num_channels <= 1:
        print("⚠️ Only one channel found (or data is not multi-channel). Skipping variance filtering.")
        return data_dict, list(range(num_channels))

    # Stack all samples: (num_samples, timepoints, num_channels)
    all_samples = np.array(list(data_dict.values()))

    # Compute variance for each channel
    channel_variances = np.var(all_samples, axis=(0, 1))

    # Keep channels with variance > threshold
    valid_channels = np.where(channel_variances > 1e-8)[0]

    filtered_data_dict = {
        sample_id: data[:, valid_channels] for sample_id, data in data_dict.items()
    }

    return filtered_data_dict, valid_channels


import numpy as np
import matplotlib.pyplot as plt


def plot_snr_per_channel(data_dict):
    """
    Plots the Signal-to-Noise Ratio (SNR) for each channel as a bar plot.

    Parameters:
    - data_dict: dict
        Dictionary where keys are sample identifiers and values are 2D NumPy arrays
        representing chromatographic data (retention times as rows, m/z channels as columns).
    """
    import matplotlib
    matplotlib.use('TkAgg')

    # Convert data dictionary to an array (num_samples, num_timepoints, num_channels)
    valid_channels_data = np.array(list(data_dict.values()))  # Shape: (num_samples, num_timepoints, num_channels)

    # Compute the mean and standard deviation across timepoints and samples for each channel
    mean_signal = np.mean(valid_channels_data, axis=(0, 1))  # Shape: (num_channels,)
    std_noise = np.std(valid_channels_data, axis=(0, 1))  # Shape: (num_channels,)

    # Compute Signal-to-Noise Ratio (SNR) for each channel (avoid division by zero)
    snr_values = mean_signal / (std_noise + 1e-10)

    # Plot SNR per channel
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(snr_values)), snr_values, color='blue', alpha=0.7)
    plt.xlabel("Channel Index", fontsize=12)
    plt.ylabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    plt.title("SNR for Each Channel", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Example usage:
    # plot_snr_per_channel(data_dict)


# def rename_directories(directory_path):
#     """
#     Renames directories in the specified path according to the transformation rules:
#       - Remove 'ML' prefix
#       - Replace 'Ester-' with 'Est'
#       - Keep the numeric part after 'ML' and move it after 'Est'
#       - Replace '_' with '-' in the remaining part of the name
#     """
#     for dir_name in os.listdir(directory_path):
#         old_path = os.path.join(directory_path, dir_name)
#
#         # Ensure it's a directory before renaming
#         if os.path.isdir(old_path) and dir_name.startswith("ML") and "Ester-" in dir_name:
#             # Extract the numeric part after 'ML'
#             parts = dir_name.split('_')
#             first_part = parts[0]  # ML23_Ester-CSA9
#             rest = '_'.join(parts[1:])  # 2.D
#
#             # Remove 'ML' prefix and extract the numeric value
#             ml_part, ester_part = dir_name.split("_Ester-")
#             number = ml_part.replace("ML", "", 1)
#
#             # Construct the new name
#             new_name = f"Est{number}{ester_part}".replace("_", "-")
#
#             new_path = os.path.join(directory_path, new_name)
#
#             try:
#                 os.rename(old_path, new_path)
#                 print(f"Renamed: {dir_name} -> {new_name}")
#             except Exception as e:
#                 print(f"Error renaming {dir_name}: {e}")


def rename_directories(directory_path):
    """
    Renames directories in the specified path according to the transformation rules:
      - Remove 'ML_' prefix
      - Extract the numeric part after 'ML_' and use it as the year (e.g., ML_MA1_3.D -> Est21MA1-3.D)
      - Replace '_' with '-' in the remaining part of the name
      - If there is no '_number' before '.D', add '_1' before '.D'
    """
    for dir_name in os.listdir(directory_path):
        old_path = os.path.join(directory_path, dir_name)

        # Ensure it's a directory before renaming
        if os.path.isdir(old_path) and dir_name.startswith("ML_"):
            parts = dir_name.split('_')

            if len(parts) >= 2:
                # Extract the numeric part (assuming it's a year reference)
                number = "21"  # Assuming '21' is the correct year reference

                # Check if there is a '_number' before '.D'
                base_name = '_'.join(parts[1:])
                if not re.search(r'_\d+(?=\.D$)', base_name):
                    base_name = base_name.replace(".D", "_1.D")

                # Construct the new name
                new_name = f"Est{number}{base_name}".replace("_", "-")
                new_path = os.path.join(directory_path, new_name)

                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {dir_name} -> {new_name}")
                except Exception as e:
                    print(f"Error renaming {dir_name}: {e}")

def join_datasets(selected_datasets, data_directories, n_decimation):
    """
    Joins multiple datasets based on the given indices.

    Parameters:
        dataset_indices (list of str): List of dataset names to join (e.g., ["press_2022", "merlot_2021"]).
        data_directories (dict): Dictionary mapping dataset names to their directories.
        n_decimation (int): Decimation factor for reducing data size.
    """
    merged_data = {}
    merged_origins = {}
    dataset_dicts = []
    dataset_origins_list = []

    row_start = 1  # Define row start
    first_dataset = selected_datasets[0]  # Use the first dataset to determine column indices
    first_directory = data_directories.get(first_dataset)

    if first_directory is None:
        raise ValueError(f"Dataset {first_dataset} does not have a corresponding directory.")

    row_end_1, fc_idx_1, lc_idx_1 = utils.find_data_margins_in_csv(first_directory)
    column_indices = list(range(fc_idx_1, lc_idx_1 + 1))

    # Load and process each selected dataset
    for dataset_name in selected_datasets:
        directory = data_directories.get(dataset_name)
        if directory is None:
            raise ValueError(f"Dataset {dataset_name} does not have a corresponding directory.")

        row_end, fc_idx, lc_idx = utils.find_data_margins_in_csv(directory)

        # Ensure column indices match
        assert fc_idx == fc_idx_1 and lc_idx == lc_idx_1, "Mismatch in column indices between datasets."

        data_dict = utils.load_ms_csv_data_from_directories(directory, column_indices, row_start, row_end)
        dataset_origins = {key: dataset_name for key in data_dict.keys()}

        dataset_dicts.append({'data': data_dict, 'origins': dataset_origins})

    # Determine the minimum sample length across all datasets
    min_length = min(
        min(array.shape[0] for array in dataset_dict['data'].values()) for dataset_dict in dataset_dicts
    )

    # Trim and decimate datasets
    for dataset_dict in dataset_dicts:
        dataset_dict['data'] = {key: array[:min_length, :] for key, array in dataset_dict['data'].items()}
        dataset_dict['data'] = {key: matrix[::n_decimation, :] for key, matrix in dataset_dict['data'].items()}
        dataset_origins_list.append(dataset_dict['origins'])

    # Merge datasets
    for dataset_dict in dataset_dicts:
        for key in dataset_dict['data'].keys():
            if key not in merged_data:
                merged_data[key] = dataset_dict['data'][key]
                merged_origins[key] = dataset_dict['origins'][key]
            else:
                merged_data[key] = np.vstack((merged_data[key], dataset_dict['data'][key]))

    return merged_data, merged_origins

def compute_pairwise_pearson(chrom_dict):
    """
    Compute pairwise Pearson correlation coefficients between chromatograms.

    Parameters
    ----------
    chrom_dict : dict
        Dictionary with keys as sample names and values as 1D NumPy arrays (chromatograms).

    Returns
    -------
    pairwise_results : list of tuples
        Each tuple contains (sample1, sample2, correlation).
    mean_corr : float
        Mean Pearson correlation coefficient across all unique pairs.
    std_corr : float
        Standard deviation of the Pearson correlation coefficients.
    """

    pairwise_results = []
    correlations = []

    # Iterate through all unique pairs of samples
    for (s1, c1), (s2, c2) in combinations(chrom_dict.items(), 2):
        # Ensure they are the same length
        if len(c1) != len(c2):
            raise ValueError(f"Chromatograms {s1} and {s2} have different lengths.")

        corr, _ = pearsonr(c1, c2)
        pairwise_results.append((s1, s2, corr))
        correlations.append(corr)

    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)

    return pairwise_results, mean_corr, std_corr

def average_confusion_matrices_ignore_empty_rows(confusion_matrices):
    """
    Averages a list of confusion matrices row-wise, ignoring zero-sum rows in each matrix.

    Parameters:
    -----------
    confusion_matrices : list of np.ndarray
        List of raw (unnormalized) confusion matrices of the same shape.

    Returns:
    --------
    np.ndarray
        A row-normalized confusion matrix where each row is averaged over non-zero rows only.
    """
    confusion_matrices = np.array(confusion_matrices)
    num_classes = confusion_matrices.shape[1]
    mean_cm = np.zeros((num_classes, num_classes), dtype=float)
    counts = np.zeros(num_classes, dtype=int)

    for cm in confusion_matrices:
        for i in range(num_classes):
            row_sum = np.sum(cm[i])
            if row_sum > 0:
                mean_cm[i] += cm[i]
                counts[i] += 1

    # Avoid division by zero by skipping zero-count rows
    for i in range(num_classes):
        if counts[i] > 0:
            mean_cm[i] /= counts[i]

    # Normalize row-wise
    row_sums = mean_cm.sum(axis=1, keepdims=True)
    normalized_mean_cm = np.divide(mean_cm, row_sums, where=row_sums != 0)
    return normalized_mean_cm

import os
import re
import pandas as pd

def create_dir_of_samples_from_bordeaux_oak(input_path, output_root=None):
    """
    Parse a wide-format file where columns are labeled like:
        'Tr Bois_V1990', 'Ab Bois_V1990', 'Tr Bois_A2000', 'Ab Bois_A2000', ...
    with optional empty spacer columns, and metadata lines above the header.

    - Detects the header row automatically (first row containing 'Tr Bois' or 'Ab Bois').
    - Extracts sample IDs from the part AFTER 'Tr Bois'/'Ab Bois' (e.g., 'V1990', 'A2000').
    - Pairs each Tr (retention time) with the matching Ab (signal) for the same sample ID.
    - Writes each pair into <output_root>/<SampleID>-<rep>.D/<SampleID>-<rep>.csv as:
        columns: ["Ret. Time", "TIC"]

    Returns:
        list[str]: paths to the created CSV files.
    """
    # --- Read CSV (no header) ---
    raw = pd.read_csv(input_path, header=None, dtype=str)

    # --- Find header row: first row that contains 'Tr Bois' or 'Ab Bois' in any cell ---
    header_row = None
    for r in range(min(len(raw), 100)):
        row_vals = " | ".join(str(x) for x in raw.iloc[r].fillna(""))
        if ("Tr Bois" in row_vals) or ("Ab Bois" in row_vals):
            header_row = r
            break
    if header_row is None:
        raise ValueError("Header row with 'Tr Bois'/'Ab Bois' not found.")

    headers = raw.iloc[header_row].fillna("").astype(str).tolist()
    # Data begins on next row
    data = raw.iloc[header_row + 1:].reset_index(drop=True)

    # --- Build column pairs by parsing headers ---
    # Accepts: 'Tr Bois_X', 'Ab Bois_X' (X = sample id). Flexible on spaces/underscore/dash.
    tr_pat = re.compile(r"^\s*Tr\s*Bois\s*[_\-\s]*([A-Za-z0-9]+)\s*$", re.IGNORECASE)
    ab_pat = re.compile(r"^\s*Ab\s*Bois\s*[_\-\s]*([A-Za-z0-9]+)\s*$", re.IGNORECASE)

    # Map sample_id -> indices
    tr_cols = {}
    ab_cols = {}

    for j, h in enumerate(headers):
        h_clean = h.strip()
        if not h_clean:
            continue  # skip empty columns
        m_tr = tr_pat.match(h_clean)
        if m_tr:
            sid = m_tr.group(1)
            tr_cols[sid] = j
            continue
        m_ab = ab_pat.match(h_clean)
        if m_ab:
            sid = m_ab.group(1)
            ab_cols[sid] = j
            continue

    # Intersect keys to get valid pairs
    sample_ids = sorted(set(tr_cols.keys()) & set(ab_cols.keys()))
    if not sample_ids:
        raise ValueError("No (Tr, Ab) pairs found with matching sample IDs.")

    # Output root
    if output_root is None:
        output_root = os.path.dirname(input_path)
    os.makedirs(output_root, exist_ok=True)

    created = []
    counts = {}

    for sid in sample_ids:
        rt_idx = tr_cols[sid]
        ab_idx = ab_cols[sid]

        # Sanitize sample id for filesystem
        base_name = re.sub(r"[^\w\-.()+]", "_", sid) or "Sample"
        counts[base_name] = counts.get(base_name, 0) + 1
        full_name = f"{base_name}-{counts[base_name]}"

        sub = data.iloc[:, [rt_idx, ab_idx]].copy()
        sub.columns = ["Ret. Time", "TIC"]
        # Drop fully empty rows
        sub = sub.dropna(how="all").reset_index(drop=True)

        # Also drop rows where both cells are blank strings
        sub = sub[~((sub["Ret. Time"].astype(str).str.strip() == "") &
                    (sub["TIC"].astype(str).str.strip() == ""))].reset_index(drop=True)

        out_dir = os.path.join(output_root, f"{full_name}.D")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{full_name}.csv")
        sub.to_csv(out_csv, index=False)
        created.append(out_csv)
        print(f"✅ Saved: {out_csv}")

    print("✔️ All samples extracted.")
    return created

from collections import defaultdict
def create_dir_of_samples_from_bordeaux_ester(input_path, output_root=None):
    """
    Parse wide-format GC-MS tables where headers look like:
      'Tr Esters V_1990-A', 'Ab Esters V_1990-A', 'Tr Esters V_1990-B', 'Ab Esters V_1990-B', ...

    Behavior:
      - Auto-detects the header row (first row containing 'Tr/Ab (Bois|Esters) ...').
      - Extracts sample IDs from the token AFTER 'Tr/Ab Esters|Bois', e.g. 'V_1990-A'.
      - Normalizes sample base ID to remove the underscore -> 'V1990'.
      - Averages replicates (suffix A/B/etc.) *per base ID* for both Tr (retention time) and Ab (signal).
      - Outputs a single pair of columns per base ID: ["Ret. Time", "TIC"].
      - Writes to <output_root>/<BaseID>-<rep>.D/<BaseID>-<rep>.csv (rep is a running index if same base appears multiple times).

    Returns:
      list[str]: paths to created CSV files.
    """

    # Read raw (no header)
    raw = pd.read_csv(input_path, header=None, dtype=str)

    # Regex that accepts both 'Bois' and 'Esters' and captures:
    # kind = Tr|Ab, letter = V/A/F/..., year = digits, opt suffix = A|B...
    header_pat = re.compile(
        r"^\s*(Tr|Ab)\s*(?:Bois|Esters)\s*[_\-\s]*([A-Za-z])[_\-\s]*(\d{2,4})(?:[_\-\s]*-?([A-Za-z]))?\s*$",
        re.IGNORECASE
    )

    # 1) Find header row
    header_row = None
    for r in range(min(len(raw), 200)):
        cells = raw.iloc[r].fillna("").astype(str)
        if any(header_pat.match(c.strip()) for c in cells):
            header_row = r
            break
    if header_row is None:
        raise ValueError("Header row with 'Tr/Ab (Bois|Esters)' not found.")

    headers = raw.iloc[header_row].fillna("").astype(str).tolist()
    data = raw.iloc[header_row + 1:].reset_index(drop=True)

    # 2) Parse columns -> collect indices by base_id and replicate suffix
    #    base_id = e.g. 'V1990'  (underscore removed)
    tr_cols = defaultdict(dict)  # base_id -> {suffix->col_index}
    ab_cols = defaultdict(dict)

    for j, h in enumerate(headers):
        h_clean = h.strip()
        if not h_clean:
            continue
        m = header_pat.match(h_clean)
        if not m:
            continue
        kind, letter, year, suffix = m.groups()
        base_id = f"{letter.upper()}{year}"     # remove underscore: V_1990 -> V1990
        suffix = (suffix.upper() if suffix else None)

        if kind.lower() == "tr":
            tr_cols[base_id][suffix] = j
        else:
            ab_cols[base_id][suffix] = j

    # Intersect to only bases that have both Tr and Ab
    base_ids = sorted(set(tr_cols.keys()) & set(ab_cols.keys()))
    if not base_ids:
        raise ValueError("No matched (Tr, Ab) base IDs found.")

    if output_root is None:
        output_root = os.path.dirname(input_path)
    os.makedirs(output_root, exist_ok=True)

    created = []
    name_counts = {}

    def _mean_across(df, col_idxs):
        """Rowwise mean over the given column indices (as floats), ignoring all-empty rows."""
        if len(col_idxs) == 1:
            return pd.to_numeric(df.iloc[:, col_idxs[0]], errors="coerce")
        # mean across columns
        arr = pd.concat([pd.to_numeric(df.iloc[:, k], errors="coerce") for k in col_idxs], axis=1)
        return arr.mean(axis=1)

    for base in base_ids:
        # Collect replicate indices for this base
        tr_idx_list = sorted([idx for idx in tr_cols[base].values() if idx is not None])
        ab_idx_list = sorted([idx for idx in ab_cols[base].values() if idx is not None])

        if not tr_idx_list or not ab_idx_list:
            # skip if missing any side
            continue

        # 3) Average replicates for Tr and Ab separately
        tr_series = _mean_across(data, tr_idx_list)
        ab_series = _mean_across(data, ab_idx_list)

        # Build output DataFrame with a single pair of columns
        sub = pd.DataFrame({
            "Ret. Time": tr_series,
            "TIC": ab_series
        })

        # Drop rows that are entirely NaN or both cells blank/NaN
        sub = sub.dropna(how="all").reset_index(drop=True)
        sub = sub[~(
            sub["Ret. Time"].astype(str).str.strip().isin(["", "nan"]) &
            sub["TIC"].astype(str).str.strip().isin(["", "nan"])
        )].reset_index(drop=True)

        # 4) Save: one file per base (rep index if base repeats)
        safe_base = re.sub(r"[^\w\-.()+]", "_", base) or "Sample"
        name_counts[safe_base] = name_counts.get(safe_base, 0) + 1
        full_name = f"{safe_base}-{name_counts[safe_base]}"

        out_dir = os.path.join(output_root, f"{full_name}.D")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{full_name}.csv")
        sub.to_csv(out_csv, index=False)
        created.append(out_csv)
        print(f"✅ Saved: {out_csv}")

    print("✔️ All samples extracted.")
    return created

# def create_dir_of_samples_from_bordeaux(input_csv_path, output_root=None):
#     """
#     Extracts individual chromatogram files from a wide-format GC-MS file.
#
#     Expected layout:
#       - Top rows may contain metadata.
#       - One row with sample names (e.g., A2003, A2010, ...) with blanks interleaved.
#       - One 'subheader' row with repeating pairs: RET TIME | NORMALIZED (or TIC / NORMALIZED SIGNAL / NORMALISED).
#       - Data rows follow.
#
#     The function:
#       1) Detects the subheader row automatically (first ~15 rows).
#       2) Uses the row above it as the sample names row; builds a cleaned list without blanks.
#       3) Iterates through detected (RET TIME, NORMALIZED/TIC) pairs, assigns names by pair index
#          from the cleaned name list, and exports each pair to:
#              <output_root>/<SampleName>-<rep>.D/<SampleName>-<rep>.csv
#
#     Returns:
#       list[str]: paths to the created CSV files.
#     """
#     print(f"📄 Reading raw CSV: {input_csv_path}")
#     raw = pd.read_csv(input_csv_path, header=None, dtype=str)
#
#     # --- detect the subheader row (RET TIME / NORMALIZED-like pairs) ---
#     def is_subheader(row):
#         vals = [str(x).strip().upper() for x in row.tolist()]
#         pairs = 0
#         i = 0
#         while i + 1 < len(vals):
#             if vals[i] in ("RET TIME", "RET.TIME", "RETENTION TIME") and vals[i+1] in ("NORMALIZED SIGNAL","NORMALIZED", "NORMALISED", "TIC"):
#                 pairs += 1
#                 i += 2
#             else:
#                 i += 1
#         return pairs >= 1
#
#     subhdr_idx = None
#     for r in range(min(15, len(raw))):
#         if is_subheader(raw.iloc[r]):
#             subhdr_idx = r
#             break
#     if subhdr_idx is None:
#         raise ValueError("Could not find the RET TIME / NORMALIZED subheader row.")
#
#     # --- sample names row is just above the subheaders ---
#     names_idx = subhdr_idx - 1
#     sample_names_row = raw.iloc[names_idx].fillna("").astype(str)
#     # Cleaned list: drop empties so it aligns one-to-one with (RET TIME, TIC) pairs
#     sample_names_cleaned = [name.strip() for name in sample_names_row if name.strip() != ""]
#
#     # --- data starts below the subheaders ---
#     data = raw.iloc[subhdr_idx + 1:].reset_index(drop=True)
#
#     # --- build column index pairs from subheader row ---
#     sh = raw.iloc[subhdr_idx].astype(str).str.strip().str.upper().tolist()
#     col_pairs = []
#     i = 0
#     while i + 1 < len(sh):
#         a, b = sh[i], sh[i+1]
#         if a in ("RET TIME", "RET.TIME", "RETENTION TIME") and b in ("NORMALIZED SIGNAL","NORMALIZED", "NORMALISED", "TIC"):
#             col_pairs.append((i, i+1))
#             i += 2
#         else:
#             i += 1
#     if not col_pairs:
#         raise ValueError("No RET TIME / NORMALIZED column pairs detected.")
#
#     # optional: helper fallback if lengths mismatch — walk left to nearest non-empty
#     def name_for_col_fallback(col_idx: int) -> str:
#         j = col_idx
#         while j >= 0 and (sample_names_row.iloc[j] is None or sample_names_row.iloc[j].strip() == ""):
#             j -= 1
#         name = (sample_names_row.iloc[j].strip() if j >= 0 else "Sample")
#         return name or "Sample"
#
#     # where to write
#     if output_root is None:
#         output_root = os.path.dirname(input_csv_path)
#     os.makedirs(output_root, exist_ok=True)
#
#     created = []
#     counts = {}
#
#     # --- export each pair using the cleaned names list by pair index ---
#     for pair_idx, (rt_idx, tic_idx) in enumerate(col_pairs):
#         if pair_idx < len(sample_names_cleaned):
#             base_name = sample_names_cleaned[pair_idx]
#         else:
#             # graceful fallback if names and pairs got out of sync
#             base_name = name_for_col_fallback(rt_idx)
#
#         # sanitize filesystem name
#         base_name = re.sub(r"[^\w\-.()+]", "_", base_name) or "Sample"
#
#         counts[base_name] = counts.get(base_name, 0) + 1
#         full_name = f"{base_name}-{counts[base_name]}"
#
#         dir_path = os.path.join(output_root, f"{full_name}.D")
#         os.makedirs(dir_path, exist_ok=True)
#
#         sub = data.iloc[:, [rt_idx, tic_idx]].copy()
#         sub.columns = ["Ret. Time", "TIC"]
#         sub = sub.dropna(axis=0, how="all").reset_index(drop=True)
#
#         out_csv = os.path.join(dir_path, f"{full_name}.csv")
#         sub.to_csv(out_csv, index=False)
#         created.append(out_csv)
#         print(f"✅ Saved: {out_csv}")
#
#     print("✔️ All samples extracted.")
#     return created

def create_dir_of_samples_from_champagnes(input_csv_path, output_root=".", use_cache=True):
    print(f"📄 Reading raw CSV: {input_csv_path}")
    raw_df = pd.read_csv(input_csv_path, header=None)

    # Use second row as header (row index 1), and keep the rest as data
    sample_names = raw_df.iloc[1]  # row 1 has names
    df = raw_df.iloc[2:].copy()    # skip rows 0 and 1
    df.columns = sample_names

    sample_counts = {}

    # Process columns in pairs (Ret. Time, TIC)
    col_names = list(df.columns)
    for i in range(0, len(col_names), 2):
        sample_name = col_names[i].strip()
        rt_col = col_names[i]
        tic_col = col_names[i + 1]

        # Update count for suffix
        sample_counts[sample_name] = sample_counts.get(sample_name, 0) + 1
        suffix = sample_counts[sample_name]

        full_sample_name = f"{sample_name}-{suffix}"
        dir_name = f"{full_sample_name}.D"
        dir_path = os.path.join(os.path.split(input_csv_path)[0], dir_name)
        os.makedirs(dir_path, exist_ok=True)

        # Extract and rename
        sub_df = df.iloc[:, [i, i + 1]].copy()
        sub_df.columns = ["Ret. Time", "TIC"]

        # Save
        output_csv_path = os.path.join(dir_path, f"{full_sample_name}.csv")
        sub_df.to_csv(output_csv_path, index=False)

        print(f"✅ Saved: {output_csv_path}")

    print("✔️ All samples extracted.")

def get_custom_order_for_pinot_noir_region(region):
    """
    Returns a custom label order for confusion matrix based on the specified region level.

    Parameters:
        region (str): One of 'winery', 'origin', 'country', 'continent'

    Returns:
        list or None: Custom ordering of labels, or None if no custom order is needed.
    """
    if region == 'winery':
        # return [
        #     "Clos Des Mouches. Drouhin (FR): D",
        #     "Les Petits Monts. Drouhin (FR): R",
        #     "Vigne de l’Enfant Jésus. Bouchard (FR): E",
        #     "Les Cailles. Bouchard (FR): Q",
        #     "Bressandes. Jadot (FR): P",
        #     "Les Boudots. Jadot (FR): Z",
        #     "Domaine Schlumberger (FR): C",
        #     "Domaine Jean Sipp (FR): W",
        #     "Domaine Weinbach (FR): Y",
        #     "Domaine Brunner (CH): M",
        #     "Vin des Croisés (CH): N",
        #     "Domaine Villard et Fils (CH): J",
        #     "Domaine de la République (CH): L",
        #     "Les Maladaires (CH): H",
        #     "Marimar Estate (US): U",
        #     "Domaine Drouhin (US): X",
        # ]
        return ['D', 'R', 'E', 'Q', 'P', 'Z', 'C', 'W', 'Y', 'M', 'N', 'J', 'L', 'H', 'U', 'X']
    elif region == 'estate':
        # Grouped estates collapsing multiple winery codes into a single label
        return ['D', 'B', 'T', 'C', 'W', 'Y', 'M', 'N', 'J', 'L', 'H', 'U', 'X']
        #     [
        #     'Drouhin',  # D, R
        #     'Bouchard',  # E, Q
        #     'Jadot',  # P, Z
        #     'Schlumberger',  # C
        #     'Jean Sipp',  # W
        #     'Weinbach',  # Y
        #     'Brunner',  # M
        #     'Vin des Croisés',  # N
        #     'Villard et Fils',  # J
        #     'République',  # L
        #     'Les Maladaires',  # H
        #     'Marimar Estate',  # U
        #     'Drouhin US',  # X
        # ]
    elif region == 'origin':
        # return ['Burgundy_North', 'Burgundy_South', 'Alsace', 'Neuchatel', 'Genève', 'Valais', 'Californie', 'Oregon']
        return ['Burgundy', 'Alsace', 'Neuchâtel', 'Geneva', 'Valais', 'California', 'Oregon']
    elif region == 'country':
        return ['France', 'Switzerland', 'US']
    elif region == 'continent':
        return ['Europe', 'America']
    elif region == 'burgundy':
        return ['NB', 'SB']
    else:
        return None

def infer_wine_kind(selected_datasets, dataset_directories):
    """
    Infers the wine kind ('bordeaux', 'pinot_noir', 'press', or 'champagne') from the dataset paths.
    Ensures all selected datasets belong to the same wine kind.
    """
    wine_kind = None
    for name in selected_datasets:
        path = dataset_directories[name].lower()

        if "bordeaux" in path:
            kind = "bordeaux"
        elif "pinot_noir" in path:
            kind = "pinot_noir"
        elif "press_wines" in path:
            kind = "press"
        elif "champagne" in path:
            kind = "champagne"
        else:
            raise ValueError(f"Could not determine wine_kind from dataset path: {path}")

        if wine_kind is None:
            wine_kind = kind
        elif wine_kind != kind:
            raise ValueError(
                f"Selected datasets include mixed wine kinds ('{wine_kind}' and '{kind}'). "
                f"Please use datasets from the same category."
            )

    return wine_kind

def assign_bordeaux_split_labels(labels, class_by_year=False):
    split_labels = []
    for lbl in labels:
        match = re.search(r'(\d{4})', lbl)
        if not match:
            split_labels.append(lbl)  # fallback: keep raw
            continue

        if class_by_year:
            # Winery + year
            letter = lbl[0]
            year = match.group(1)
            split_labels.append(f"{letter}{year}")
        else:
            # Keep full raw label (A1990, A1990B, etc.)
            split_labels.append(lbl)
    return np.array(split_labels)

def assign_bordeaux_label(labels, vintage=False):
    """
    Assigns labels for Bordeaux wines, optionally grouping by vintage or by composite label.

    Args:
        labels (list of str): A list of wine sample labels (e.g., 'A2022', 'B2021B').
        vintage (bool): If True, extract only the year (e.g., '2022').
                        If False, extract composite label like 'A2022' but remove trailing 'B' duplicates.

    Returns:
        np.ndarray: Processed labels as per selected mode.
    """
    processed_labels = []

    for label in labels:
        match = re.search(r'(\d{4})', label)
        if not match:
            processed_labels.append(None)
            continue

        if vintage:
            year = match.group(1)
            processed_labels.append(year)
        else:
            # Extract leading letter and year, ignore trailing B
            letter = label[match.start() - 1]
            year = match.group(1)
            processed_labels.append(f"{letter}{year}")

    return np.array(processed_labels)

def compute_features(data, channels=None, feature_type="concat_channels"):
    """
    Compute feature representations from GC-MS data.

    Parameters:
    ----------
    data : np.ndarray
        GC-MS data of shape (n_samples, n_timepoints, n_channels).
    channels : list or None
        List of channel indices to include. If None, all channels are used.
    feature_type : str
        One of 'concatenated', 'tic', 'tis', 'tic_tis':
        - 'concatenated': Flatten each selected channel and concatenate them.
        - 'tic': Total Ion Current (sum over channels).
        - 'tis': Total Ion Spectrum (sum over timepoints).
        - 'tic_tis': Concatenation of TIC and TIS features.

    Returns:
    -------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    """
    if channels is None:
        channels = list(range(data.shape[2]))

    if feature_type == "concatenated":
        return np.hstack([data[:, :, ch].reshape(data.shape[0], -1) for ch in channels])
    elif feature_type == "tic":
        return np.sum(data[:, :, channels], axis=2)
    elif feature_type == "tis":
        return np.sum(data[:, :, channels], axis=1)
    elif feature_type == "tic_tis":
        tic = np.sum(data[:, :, channels], axis=2)
        tis = np.sum(data[:, :, channels], axis=1)
        return np.hstack([tic, tis])
    else:
        raise ValueError("Invalid feature_type. Use 'concat_channels', 'tic', 'tis', or 'tic_tis'.")

from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
# from gcmswine.utils import print_letter_origine_and_date_from_sample_info
# root_path = "/Users/juliaroquette/Work/oenology/data/3D_data/PINOT_NOIR/Changins/220322_Pinot_Noir_Tom/"
# print_letter_origine_and_date_from_sample_info(root_path)
def print_letter_origine_and_date_from_sample_info(root_path):
    letter_to_origine = {
        "P": "Burgundy", "D": "Burgundy", "M": "Neuchâtel", "L": "Geneva",
        "X": "Oregon", "W": "Alsace", "C": "Alsace", "J": "Geneva",
        "Y": "Alsace", "K": "Alsace", "H": "Valais", "R": "Burgundy",
        "U": "California", "Z": "Burgundy", "Q": "Burgundy", "E": "Burgundy",
        "N": "Neuchâtel",
    }

    root = Path(root_path)
    entries = []

    for d_folder in root.glob("*.D"):
        acqdata_path = d_folder / "AcqData" / "sample_info.xml"
        letter = d_folder.name[:3]
        origine = letter_to_origine.get(letter[0], "Unknown")


        if acqdata_path.exists():
            try:
                tree = ET.parse(acqdata_path)
                root_elem = tree.getroot()

                # Find AcqTime
                display_elems = root_elem.findall(".//DisplayName")
                acq_time_value = None

                for i, elem in enumerate(display_elems):
                    if elem.text == "AcqTime":
                        value_elem = display_elems[i].getparent()[i+1] if hasattr(display_elems[i], 'getparent') else display_elems[i].find("../Value")
                        if value_elem is not None and value_elem.text:
                            acq_time_value = value_elem.text.strip()
                            break

                if not acq_time_value:
                    # fallback: brute-force search for Value after AcqTime
                    lines = acqdata_path.read_text(errors="ignore").splitlines()
                    for idx, line in enumerate(lines):
                        if "DisplayName>AcqTime" in line:
                            for j in range(idx+1, min(idx+4, len(lines))):
                                if "<Value>" in lines[j]:
                                    acq_time_value = lines[j].strip().replace("<Value>", "").replace("</Value>", "")
                                    break
                            break

                if acq_time_value:
                    dt = datetime.fromisoformat(acq_time_value.split("+")[0])  # Drop timezone offset
                    date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_str = "[Error: AcqTime not found]"

            except Exception as e:
                date_str = f"[Error: {e}]"

            entries.append((date_str, letter, origine))

    # Align and sort
    date_width = max(len(date_str) for date_str, _, _ in entries)
    entries.sort()

    for date_str, letter, origine in entries:
        print(f"{letter} — {origine:<12} — {date_str.rjust(date_width)}")