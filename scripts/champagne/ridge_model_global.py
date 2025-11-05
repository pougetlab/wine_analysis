"""
Ridge Regression for Predicting Champagne Sensory Profiles Using GC-MS Data
============================================================================

This script implements a regression pipeline to predict detailed sensory scores for Champagne wines
based on chemical information from GC-MS chromatograms and taster identity. It combines signal
processing, metadata handling, and machine learning in a reproducible way.

Use case:
---------
Designed to assess the predictive power of GC-MS data (flattened chromatograms or TICs) in modeling
human sensory perception, taking into account taster-specific biases via one-hot encoding.

Taster Modeling Approach:
-------------------------
A single Ridge regression model is trained across all tasters. Taster identity is provided as an input
feature using one-hot encoding, allowing the model to learn both general chemical–sensory relationships
and individual taster preferences. There is no separate model per taster.

Key Features:
-------------
1. **GC-MS Data Loading**:
   - Reads chromatographic signals from `.csv` files using a custom utility (`utils.py`).
   - Chromatograms are decimated (downsampled) along the retention time axis.

2. **Sensory Metadata Preprocessing**:
   - Loads a structured sensory dataset (`sensory_scores.csv`) with wine codes, tasters, and 100+ attributes.
   - Cleans and averages duplicate ratings by (wine, taster) pairs.

3. **Feature Construction**:
   - Each sample’s feature vector is the mean chromatogram + one-hot encoded taster ID.
   - Targets are sensory attribute vectors (e.g., fruity, toasted, acid, etc.).

4. **Model Training & Evaluation**:
   - Runs `N_REPEATS` Ridge regression fits using randomized train/test splits (80/20).
   - Evaluates with MAE and RMSE for each sensory attribute.
   - Aggregates performance statistics and prints per-taster error summaries.

5. **Interpretability**:
   - Separates the model weights into chromatogram vs. taster contributions.
   - Identifies which tasters introduce the most prediction error.

6. **Visualization**:
   - For each taster, plots the predicted vs. true sensory profiles across test wines.
   - Optional single-wine plot function available for inspection.

Parameters to Adjust:
---------------------
- `directory`: Path to chromatogram files.
- `N_DECIMATION`: Decimation factor (to reduce RT dimensionality).
- `N_REPEATS`: Number of random train/test splits.
- `TEST_SIZE`: Test proportion (e.g., 0.2 = 20%).

Dependencies:
-------------
- pandas, numpy, scikit-learn, matplotlib
- Custom `utils.py` for chromatogram loading

Example Output:
---------------
- Per-attribute MAE and RMSE scores
- Per-taster mean absolute errors
- Prediction plots showing model quality for each taster
"""
import sys
from locale import normalize

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from gcmswine import utils  # Your custom module
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn.metrics import r2_score
import os
import yaml
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from gcmswine.helpers import (
    load_config, get_model, load_chromatograms_decimated, load_and_clean_metadata, build_feature_target_arrays,
)
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
from gcmswine.wine_analysis import ChromatogramAnalysis, GCMSDataProcessor
from gcmswine.logger_setup import logger, logger_raw
from collections import Counter
from scipy.optimize import minimize
from gcmswine.helpers import average_by_wine
from gcmswine.dimensionality_reduction import DimensionalityReducer


# Define helper functions
def build_feature_target_arrays(metadata, sensory_cols, data_dict):
    X_raw, y, taster_ids, sample_ids = [], [], [], []
    skipped_count = 0

    for _, row in metadata.iterrows():
        sample_id = row['code vin']
        taster_id = row['taster']
        try:
            attributes = row[sensory_cols].astype(float).values
        except:
            continue  # skip row if attribute values can't be converted

        replicate_keys = [k for k in data_dict if k.startswith(sample_id)]
        if not replicate_keys:
            skipped_count += 1
            print(f"Warning: No chromatograms found for sample {sample_id}")
            continue

        replicates = np.array([data_dict[k] for k in replicate_keys])
        chromatogram = np.mean(replicates, axis=0).flatten()
        chromatogram = np.nan_to_num(chromatogram, nan=0.0)

        X_raw.append(chromatogram)
        y.append(attributes)
        taster_ids.append(taster_id)
        sample_ids.append(sample_id)

    print(f"\nTotal samples skipped due to missing chromatograms: {skipped_count}")
    return np.array(X_raw), np.array(y), np.array(taster_ids), np.array(sample_ids)

def compare_self_vs_group_models_per_taster(
    X_input, y, sample_ids, taster_ids, model, sensory_cols,
    num_repeats=5, test_size=0.2, normalize=True, random_seed=42,
    group_wines=False,
):
    """
    For each taster:
      - Train a model on their own scores (self-model)
      - Train a model on the group average (excluding them) and test on their scores
    Computes R² by aggregating predictions across all folds/repeats before scoring.

    Returns
    -------
    results_df : pd.DataFrame with R² per attribute and overall mean R² for both models.
    """
    unique_tasters = np.unique(taster_ids)
    rows = []

    n_splits = int(1 / test_size)

    for target_taster in unique_tasters:
        # === SELF MODEL ===
        mask_self = (taster_ids == target_taster)
        X_self = X_input[mask_self]
        y_self = y[mask_self]
        sample_ids_self = sample_ids[mask_self]

        r2_self_all = []

        for repeat in range(num_repeats):
            if group_wines:
                # Shuffle wines before splitting to get different splits per repeat
                unique_wines = np.unique(sample_ids_self)
                rng = np.random.default_rng(seed=random_seed + repeat)
                shuffled_wines = rng.permutation(unique_wines)

                # Map wine IDs to shuffled indices
                wine_to_idx = {wine: i for i, wine in enumerate(shuffled_wines)}
                groups_shuffled = np.array([wine_to_idx[w] for w in sample_ids_self])

                gkf = GroupKFold(n_splits=n_splits)
                splits = list(gkf.split(X_self, y_self, groups=groups_shuffled))
                # Cycle through splits by repeat number
                train_idx, test_idx = splits[repeat % len(splits)]
            else:
                # Simple random split if not grouping wines
                train_idx, test_idx = train_test_split(
                    np.arange(len(X_self)), test_size=test_size, random_state=random_seed + repeat
                )

            X_train, X_test = X_self[train_idx], X_self[test_idx]
            y_train, y_test = y_self[train_idx], y_self[test_idx]

            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y.shape[1])]
            r2_self_all.append(r2_scores)

        r2_self_avg = np.nanmean(r2_self_all, axis=0)

        # === GROUP MODEL (train on averaged others, test on target taster) ===
        mask_group = taster_ids != target_taster
        X_sub = X_input[mask_group]
        y_sub = y[mask_group]
        sample_ids_sub = sample_ids[mask_group]
        taster_ids_sub = taster_ids[mask_group]

        results = train_and_evaluate_average_scores_model(
            X_input=X_sub,
            y=y_sub,
            sample_ids=sample_ids_sub,
            model=model,
            num_repeats=num_repeats,
            test_size=test_size,
            normalize=normalize,
            random_seed=random_seed,
        )
        all_mae, all_rmse, all_r2, all_y_test, all_y_pred, all_sample_ids, all_taster_ids, *_ = results

        y_test_avg, y_pred_avg, _, _ = average_predictions_across_repeats(
            all_y_test, all_y_pred, all_sample_ids, None
        )
        r2_per_attr = []
        for i, col in enumerate(sensory_cols):
            try:
                r2 = r2_score(y_test_avg[:, i], y_pred_avg[:, i])
            except ValueError:
                r2 = float("nan")
            r2_per_attr.append(r2)

        # y_test_all = np.vstack(all_y_test)
        # y_pred_all = np.vstack(all_y_pred)

        # r2_per_attr = []
        # for i, col in enumerate(sensory_cols):
        #     try:
        #         r2 = r2_score(y_test_all[:, i], y_pred_all[:, i])
        #     except ValueError:
        #         r2 = float("nan")
        #     r2_per_attr.append(r2)

        row = {
            'taster': target_taster,
            'mean_r2_self': np.nanmean(r2_self_avg),
            'mean_r2_group': np.nanmean(r2_per_attr),
        }
        for i, attr in enumerate(sensory_cols):
            row[f'{attr}_r2_self'] = r2_self_avg[i]
            row[f'{attr}_r2_group'] = r2_per_attr[i]

        rows.append(row)

    df = pd.DataFrame(rows).set_index("taster")
    logger.info("\n=== Comparison of self vs group models ===")
    df_to_log = df[['mean_r2_self', 'mean_r2_group']].round(3).reset_index()
    print(df_to_log.to_string(index=False))
    logger_raw(df_to_log.to_string(index=False))
    means = df[['mean_r2_self', 'mean_r2_group']].mean()
    print(f"Mean r2_self: {means['mean_r2_self']:.3f}\nMean r2_group: {means['mean_r2_group']:.3f}")
    logger_raw(f"Mean r2_self: {means['mean_r2_self']:.3f}\nMean r2_group: {means['mean_r2_group']:.3f}")
    return df


def train_and_evaluate_average_scores_model(
        X_input, y, sample_ids, model, *,
        num_repeats=5, test_size=0.2, normalize=True,
        random_seed=42,taster_ids=None,
        reduce_targets=False, reduction_method="pca", reduced_dim=3
):
    """
    Train and evaluate a model on average sensory scores per wine sample.

    Returns updated to include R² scores averaged across repeats.
    """
    # Average feature vectors and scores by wine code
    X_input, y, sample_ids = average_by_wine(X_input, y, sample_ids)
    taster_ids = None  # drop taster info for average model

    all_mae, all_rmse = [], []
    all_r2 = []   # to accumulate R²

    last_y_test = last_y_pred = last_sample_ids = last_taster_ids = None

    n_splits = 5

    all_y_test, all_y_pred = [], []
    all_sample_ids, all_taster_ids = [], []

    for repeat in range(num_repeats):
        splits = [train_test_split(np.arange(len(X_input)), test_size=test_size, random_state=random_seed + repeat)]
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"[AVG] Repeat {repeat + 1}, Fold {fold_idx + 1}")

            X_train, X_test = X_input[train_idx], X_input[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            s_test = sample_ids[test_idx]
            t_test = taster_ids[test_idx] if taster_ids is not None else None

            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if reduce_targets:
                if reduction_method == "pca":
                    pca = PCA(n_components=reduced_dim, random_state=random_seed)
                    y_train_reduced = pca.fit_transform(y_train)
                    y_test_reduced = pca.transform(y_test)
                else:
                    # Implement UMAP or t-SNE similarly, but beware inverse transform missing
                    raise NotImplementedError("Only PCA supported for now")

                model.fit(X_train, y_train_reduced)
                y_pred_reduced = model.predict(X_test)

                # Invert predictions back to original space for metrics
                y_pred = pca.inverse_transform(y_pred_reduced)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Calculate average R²  for this fold
            try:
                r2_val = r2_score(y_test, y_pred, multioutput='uniform_average')
            except ValueError:
                r2_val = float("nan")

            all_r2.append(r2_val)

            # # Calculate and accumulate R² per attribute for this fold
            # r2_fold = []
            # for i in range(y.shape[1]):
            #     try:
            #         r2_val = r2_score(y_test[:, i], y_pred[:, i])
            #     except ValueError:
            #         r2_val = float("nan")
            #     r2_fold.append(r2_val)
            # all_r2.append(r2_fold)

            # Accumulate other metrics and predictions
            all_y_test.append(y_test)
            all_y_pred.append(y_pred)
            all_sample_ids.append(s_test)
            if t_test is not None:
                all_taster_ids.append(t_test)

            mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
            rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
            all_mae.append(mae)
            all_rmse.append(rmse)

            # Save last fold test data for optional use
            last_y_test = y_test
            last_y_pred = y_pred
            last_sample_ids = s_test
            last_taster_ids = t_test

    # Convert to numpy array for easier mean calculations
    all_r2 = np.array(all_r2)  # shape: (num_repeats * n_splits, n_attributes)

    return all_mae, all_rmse, all_r2, all_y_test, all_y_pred, all_sample_ids, all_taster_ids, last_y_test, last_y_pred, last_sample_ids, last_taster_ids

def average_predictions_across_repeats(all_y_test, all_y_pred, all_sample_ids, all_taster_ids=None):
    records = []
    for y_true, y_pred, s_ids, t_ids in zip(all_y_test, all_y_pred, all_sample_ids, all_taster_ids if all_taster_ids is not None else [None]*len(all_y_test)):
        for i in range(len(y_true)):
            record = {
                'sample_id': s_ids[i],
                'y_true': y_true[i],
                'y_pred': y_pred[i],
            }
            if t_ids is not None:
                record['taster_id'] = t_ids[i]
            else:
                # Mark a default taster id or skip grouping by taster
                record['taster_id'] = None
            records.append(record)

    df = pd.DataFrame(records)

    # Expand arrays in 'y_true' and 'y_pred' columns into separate numeric columns
    y_true_expanded = np.vstack(df['y_true'].values)
    y_pred_expanded = np.vstack(df['y_pred'].values)
    df = df.drop(columns=['y_true', 'y_pred'])

    for i in range(y_true_expanded.shape[1]):
        df[f'y_true_{i}'] = y_true_expanded[:, i]
        df[f'y_pred_{i}'] = y_pred_expanded[:, i]

    # Group by sample_id and optionally taster_id if available
    if all_taster_ids is not None:
        grouped = df.groupby(['sample_id', 'taster_id']).mean().reset_index()
    else:
        grouped = df.groupby(['sample_id']).mean().reset_index()

    y_true_avg = grouped[[f'y_true_{i}' for i in range(y_true_expanded.shape[1])]].values
    y_pred_avg = grouped[[f'y_pred_{i}' for i in range(y_pred_expanded.shape[1])]].values
    sample_ids_avg = grouped['sample_id'].values
    taster_ids_avg = grouped['taster_id'].values if 'taster_id' in grouped.columns else None

    return y_true_avg, y_pred_avg, sample_ids_avg, taster_ids_avg


def train_and_evaluate_model(
        X_input, y, taster_ids, sample_ids, model, *,
        num_repeats=5, test_size=0.2, normalize=True,
        taster_scaling=False, group_wines=False, random_seed=42,
        reduce_targets=False, reduction_method="pca", reduced_dim=3,
):
    """
    Train and evaluate the model with or without grouping by wines.

    Parameters
    ----------
    X_input : ndarray
        Input feature matrix (chromatogram + one-hot taster).
    y : ndarray
        Target matrix (sensory attributes).
    taster_ids : ndarray
        Array of taster identifiers (1D).
    sample_ids : ndarray
        Array of wine identifiers (1D).
    model : regressor object
        The regression model to train.
    num_repeats : int
        Number of repeats.
    test_size : float
        Proportion of data to use for test split.
    normalize : bool
        Whether to normalize features before training.
    taster_scaling : bool
        Whether to apply per-taster prediction scaling.
    useEffect(() => {
  if (tasterTests.length > 0 && groupWines) {
    setGroupWines(false);
  }
}, [tasterTests]); : bool
        Whether to group folds by wine ID.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    all_mae : list of np.ndarray
    all_rmse : list of np.ndarray
    taster_mae_summary : dict
    last_y_test : np.ndarray
    last_y_pred : np.ndarray
    last_sample_ids : np.ndarray
    last_taster_ids : np.ndarray
    """
    all_mae, all_rmse = [], []
    taster_mae_summary = defaultdict(list)
    last_y_test = last_y_pred = last_sample_ids = last_taster_ids = None
    taster_r2_summary = defaultdict(list)  # Accumulate R2 scores per taster
    all_r2_values = []

    all_y_test = []
    all_y_pred = []
    all_sample_ids = []
    all_taster_ids = []

    n_splits = 5

    for repeat in range(num_repeats):
        if group_wines:
            # Shuffle groups before GroupKFold to avoid order bias
            unique_groups = np.unique(sample_ids)
            rng = np.random.default_rng(seed=random_seed + repeat)
            shuffled_groups = rng.permutation(unique_groups)

            # Map original groups to shuffled order
            group_to_order = {group: i for i, group in enumerate(shuffled_groups)}
            # Get sorting order based on shuffled groups
            order = np.argsort([group_to_order[g] for g in sample_ids])

            # Shuffle X_input, y, sample_ids, taster_ids accordingly
            X_input_shuffled = X_input[order]
            y_shuffled = y[order]
            sample_ids_shuffled = sample_ids[order]
            taster_ids_shuffled = taster_ids[order]

            gkf = GroupKFold(n_splits=n_splits)
            splits = gkf.split(X_input_shuffled, y_shuffled, groups=sample_ids_shuffled)

            # gkf = GroupKFold(n_splits=n_splits)
            # splits = gkf.split(X_input, y, groups=sample_ids)
        else:
            splits = [train_test_split(
                np.arange(len(X_input)),
                test_size=test_size,
                random_state=random_seed + repeat
            )]

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"Repeat {repeat + 1}, Fold {fold_idx + 1}")

            X_train, X_test = X_input[train_idx], X_input[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            t_train, t_test = taster_ids[train_idx], taster_ids[test_idx]
            s_test = sample_ids[test_idx]

            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if reduce_targets:
                if reduction_method == "pca":
                    pca = PCA(n_components=reduced_dim, random_state=random_seed)
                    y_train_reduced = pca.fit_transform(y_train)
                    y_test_reduced = pca.transform(y_test)
                else:
                    # Implement UMAP or t-SNE similarly, but beware inverse transform missing
                    raise NotImplementedError("Only PCA supported for now")

                model.fit(X_train, y_train_reduced)
                y_pred_reduced = model.predict(X_test)

                # Invert predictions back to original space for metrics
                y_pred = pca.inverse_transform(y_pred_reduced)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Append predictions and metadata for averaging across splits:
            all_y_test.append(y_test)
            all_y_pred.append(y_pred)
            all_sample_ids.append(s_test)
            all_taster_ids.append(t_test)

            # model.fit(X_train, y_train)
            # y_pred = model.predict(X_test)

            if taster_scaling:
                taster_scalers = {}

                # Predict on training data
                if group_wines:
                    # Perform nested GroupKFold on training set to get y_train_pred
                    inner_gkf = GroupKFold(n_splits=5)
                    y_train_pred = np.zeros_like(y_train)

                    for inner_train_idx, inner_val_idx in inner_gkf.split(X_train, y_train, groups=sample_ids[train_idx]):
                        model.fit(X_train[inner_train_idx], y_train[inner_train_idx])
                        y_train_pred[inner_val_idx] = model.predict(X_train[inner_val_idx])
                    # y_train_pred = model.predict(X_train)
                else:
                    y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)

                for t in np.unique(t_train):
                    mask = (t_train == t)
                    if np.sum(mask) < 2:
                        continue  # Not enough samples to compute reliable scaling

                    scales = compute_positive_scales(y_train_pred[mask], y_train[mask])
                    taster_scalers[t] = scales

                # Apply learned scales to test predictions
                for i, t in enumerate(t_test):
                    if t in taster_scalers:
                        y_pred[i] *= taster_scalers[t]

            # if taster_scaling:
            #     taster_scalers = {}
            #
            #     # Predict on training data
            #     if group_wines:
            #         # If ratings are averaged across tasters, use direct prediction
            #         y_train_pred = model.predict(X_train)
            #     else:
            #         # Otherwise, use cross-validated predictions to avoid overfitting to tasters
            #         y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)
            #
            #     # Loop over all unique tasters in the training set
            #     for t in np.unique(t_train):
            #         mask = (t_train == t)  # Select samples from taster `t`
            #
            #         if np.sum(mask) < 2:
            #             continue  # Not enough samples to compute a reliable scaling
            #
            #         # Fit a linear model (no intercept) to map predicted to true scores
            #         # One model per attribute (MultiOutputRegressor)
            #         scaler_model = MultiOutputRegressor(Ridge(alpha=1, fit_intercept=False))
            #         # scaler_model = MultiOutputRegressor(LinearRegression(fit_intercept=False))
            #         scaler_model.fit(y_train_pred[mask], y_train[mask])
            #
            #         # Extract scaling coefficients (slopes) for each attribute
            #         scale = np.array([est.coef_[0] for est in scaler_model.estimators_])
            #
            #         # Store the scaling factor for this taster
            #         taster_scalers[t] = scale
            #
            #     # Apply the learned taster-specific scaling to test predictions
            #     for i, t in enumerate(t_test):
            #         if t in taster_scalers:
            #             # Scale each prediction vector by the taster-specific factor
            #             y_pred[i] *= taster_scalers[t]

            mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
            rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

            all_mae.append(mae)
            all_rmse.append(rmse)

            for i, t in enumerate(t_test):
                taster_mae_summary[t].append(np.abs(y_test[i] - y_pred[i]))

            # **Accumulate R2 per sample (per taster) here:**
            for i in range(len(y_test)):
                try:
                    r2 = r2_score(y_test[i], y_pred[i])
                except ValueError:
                    r2 = float('nan')
                taster_r2_summary[t_test[i]].append(r2)
                all_r2_values.append(r2)

            if repeat == num_repeats - 1 and (not group_wines or fold_idx == n_splits - 1):
                last_y_test = y_test
                last_y_pred = y_pred
                last_sample_ids = s_test
                last_taster_ids = t_test

    # return all_mae, all_rmse, taster_mae_summary, last_y_test, last_y_pred, last_sample_ids, last_taster_ids
    # return all_mae, all_rmse, taster_mae_summary, taster_r2_summary, all_r2_values, last_y_test, last_y_pred, last_sample_ids, last_taster_ids
    return (all_mae, all_rmse, taster_mae_summary, taster_r2_summary, all_r2_values, all_y_test, all_y_pred,
            all_sample_ids, all_taster_ids, last_y_test, last_y_pred, last_sample_ids, last_taster_ids)



def reduce_targets(y, method='pca', n_components=2):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    y_reduced = reducer.fit_transform(y)
    return y_reduced


def compute_positive_scales(y_pred_taster, y_true_taster):
    """
    Compute positive scaling factors per attribute for a single taster.

    Parameters
    ----------
    y_pred_taster : np.ndarray, shape (n_samples, n_attributes)
        Predictions for samples of this taster.
    y_true_taster : np.ndarray, shape (n_samples, n_attributes)
        True targets for samples of this taster.

    Returns
    -------
    scales : np.ndarray, shape (n_attributes,)
        Positive scaling factors per attribute.
    """
    def mse_loss(scales):
        # scales is 1D array of length n_attributes
        y_scaled = y_pred_taster * scales  # broadcast multiply
        return np.mean((y_true_taster - y_scaled) ** 2)

    n_attributes = y_true_taster.shape[1]
    initial_scales = np.ones(n_attributes)

    bounds = [(0, None)] * n_attributes  # positive constraints

    result = minimize(mse_loss, initial_scales, bounds=bounds, method='L-BFGS-B')
    return result.x

def compute_shap(
    X,
    y,
    model,
    sensory_cols,
    retention_time_range=None,
    decimation_factor=10,
    normalize=True,
    return_values_only=False
):
    """
    Compute and optionally plot SHAP values for a multi-output regression model trained on GC-MS TIC features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (e.g., averaged TICs), shape (n_samples, n_features)
    y : np.ndarray
        Target matrix (e.g., averaged sensory scores), shape (n_samples, n_attributes)
    model : sklearn regressor
        A multi-output-compatible model (e.g., Ridge, XGBRegressor).
    sensory_cols : list of str
        List of names for each sensory attribute (used in plot titles).
    retention_time_range : tuple (float, float) or None
        Optional start and end RT in minutes, overrides default from decimation.
    decimation_factor : int
        Number of seconds between TIC samples (used if RT range not provided).
    normalize : bool
        Whether to standardize features before SHAP (matches training).
    return_values_only : bool
        If True, returns SHAP values and skips plotting.

    Returns
    -------
    If return_values_only:
        shap_values.values : np.ndarray
            SHAP values of shape (n_samples, n_features, n_attributes)
        retention_times : np.ndarray
            Index or time values for x-axis (e.g., retention time bins)
    Else:
        None (shows plots)
    """
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    from sklearn.preprocessing import StandardScaler

    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    model.fit(X_scaled, y)

    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    n_features = X.shape[1]
    if retention_time_range:
        start_rt, end_rt = retention_time_range
        retention_times = np.linspace(start_rt, end_rt, n_features)
    else:
        retention_times = np.arange(n_features)

    if return_values_only:
        return shap_values.values, retention_times

    # Plotting block
    n_attrs = y.shape[1]
    n_cols = 3
    n_rows = math.ceil(n_attrs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=True)

    axes = axes.flatten()
    for i in range(n_attrs):
        mean_shap = shap_values.values[:, :, i].mean(axis=0)
        ax = axes[i]
        ax.plot(retention_times, mean_shap)
        title = sensory_cols[i] if sensory_cols else f"Attribute {i}"
        ax.set_title(title)
        ax.set_xlabel("Retention Time Index")
        ax.set_ylabel("SHAP Value")
        ax.grid(True)

    for j in range(n_attrs, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.suptitle("SHAP Profiles by Sensory Attribute", fontsize=16, y=1.02)
    plt.show()


def plot_wine_across_tasters(y_true, y_pred, sample_ids, taster_ids, wine_code, sensory_cols):
        import matplotlib.pyplot as plt
        import numpy as np

        indices = np.where(sample_ids == wine_code)[0]
        if len(indices) < 2:
            logger.info(f"Wine {wine_code} is not rated by multiple tasters in the test set.\n")
            return

        n = len(indices)
        fig, axes = plt.subplots(1, n, figsize=(n * 4, 4), sharey=True)

        if n == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            ax = axes[i]
            taster = taster_ids[idx]
            ax.plot(y_true[idx], label="True", color="black", linewidth=2)
            ax.plot(y_pred[idx], label="Pred", color="red", linestyle="--")
            ax.set_title(f"Taster {taster}\nMAE: {np.mean(np.abs(y_true[idx] - y_pred[idx])):.2f}")
            ax.set_xticks(range(len(sensory_cols)))
            ax.set_xticklabels(sensory_cols, rotation=90, fontsize=6)
            ax.set_ylim(0, 100)
            ax.grid(True)

        plt.suptitle(f"Wine {wine_code}: Sensory predictions across tasters", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.legend()
        plt.show()

def plot_single_wine(y_true, y_pred, sample_ids, wine_code):
    import matplotlib.pyplot as plt
    import numpy as np

    indices = np.where(sample_ids == wine_code)[0]
    if len(indices) == 0:
        print(f"No wine with code {wine_code} found in the test set.")
        return

    idx = indices[0]
    true_profile = y_true[idx]
    pred_profile = y_pred[idx]
    mae = np.mean(np.abs(true_profile - pred_profile))

    plt.figure(figsize=(8, 4))
    plt.plot(true_profile, label='True', color='black', linewidth=2)
    plt.plot(pred_profile, label='Predicted', color='red', linestyle='--')
    plt.title(f"Wine {wine_code} – MAE: {mae:.2f}")
    plt.xlabel("Sensory attributes")
    plt.ylabel("Score (0–100)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
# plot_single_wine(last_y_test, last_y_pred, last_sample_ids, wine_code='141T-N-27')
plt.show()


def plot_profiles_grouped_by_taster(y_true, y_pred, sample_ids, taster_ids, n_cols=10):
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import r2_score


        # unique_tasters = sorted(set(taster_ids))
        all_indices = [np.where(taster_ids == t)[0] for t in unique_tasters]
        n_rows = len(unique_tasters)
        max_len = max(len(idx) for idx in all_indices)
        n_cols = min(n_cols, max_len)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2), sharey=True)
        if n_rows == 1:
            axes = [axes]  # make iterable
        axes = np.array(axes).reshape(n_rows, n_cols)

        for row_idx, (taster, indices) in enumerate(zip(unique_tasters, all_indices)):
            for i in range(min(n_cols, len(indices))):
                idx = indices[i]
                ax = axes[row_idx, i]
                ax.plot(y_true[idx], color='black', linewidth=1.5)
                ax.plot(y_pred[idx], color='red', linewidth=1)
                true_profile = y_true[idx]
                pred_profile = y_pred[idx]
                mae_i = np.mean(np.abs(true_profile - pred_profile))
                try:
                    r2_i = r2_score(true_profile, pred_profile)
                except ValueError:
                    r2_i = float('nan')  # in case y_true is constant
                ax.set_title(f"{sample_ids[idx]}; MAE: {mae_i:.2f}; R²: {r2_i:.2f}", fontsize=8, pad=2)
                # ax.set_title(f"{sample_ids[idx]}\n{mae_i:.2f}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            for j in range(len(indices), n_cols):
                axes[row_idx, j].axis('off')

        for row_idx, taster in enumerate(unique_tasters):
            axes[row_idx, 0].set_ylabel(f"Taster {taster}", fontsize=9, rotation=0, labelpad=40)

        plt.suptitle("All predicted sensory profiles by taster (true in black, predicted in red)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)

def natural_key(t):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', t)]

def plot_r2_per_taster(taster_r2_summary, title="Average R² per Taster"):
    """
    Logs and plots average R² per taster and overall R² across all tasters.

    Parameters
    ----------
    taster_r2_summary : dict
        Dictionary where keys are taster names/IDs and values are lists of R² scores.
    title : str
        Plot title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    tasters = []
    avg_r2_scores = []
    all_r2_values = []

    for taster in sorted(taster_r2_summary.keys(), key=natural_key):
        r2_list = np.array(taster_r2_summary[taster])
        r2_list = r2_list[~np.isnan(r2_list)]
        if len(r2_list) > 0:
            avg_r2 = np.mean(r2_list)
            tasters.append(str(taster))
            avg_r2_scores.append(avg_r2)
            all_r2_values.extend(r2_list)

            print(f"Taster {taster}: Average R² = {avg_r2:.3f}")

    # Average of per-taster averages (equal weight per taster)
    mean_per_taster_r2 = np.mean(avg_r2_scores) if avg_r2_scores else float('nan')

    # Overall average R² across all individual scores (weighted by sample count)
    overall_avg_r2 = np.mean(all_r2_values) if all_r2_values else float('nan')

    print(f"\nMean R² (average per taster): {mean_per_taster_r2:.3f}")
    print(f"Overall average R² (all samples combined): {overall_avg_r2:.3f}")

    plt.figure(figsize=(10, 5))
    plt.bar(tasters, avg_r2_scores, color="skyblue")
    # plt.axhline(mean_per_taster_r2, color="red", linestyle="--", label="Mean R² (per taster)")
    plt.axhline(overall_avg_r2, color="red", linestyle="--", label="Overall Avg R²")
    plt.xlabel("Taster")
    plt.ylabel("Average R²")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_self_vs_group_r2_df(df, save_path=None):
    """
    Plot mean R² comparison between self and group models from a DataFrame indexed by taster.
    Also shows horizontal average R² lines and ensures y-axis covers both negative and positive space.

    Parameters
    ----------
    df : pd.DataFrame
        Must include 'mean_r2_self' and 'mean_r2_group' columns. Index = taster labels (e.g., 'D1', 'D2', ...)
    save_path : str or None
        If provided, saves the figure to the given path.
    """
    # Reset index to access taster labels
    df = df.reset_index()

    # Natural sort: extract number from taster labels (e.g., D1 -> 1)
    df["taster_num"] = df["taster"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("taster_num")

    # Values for plotting
    tast_ids = df["taster"].values
    r2_self = df["mean_r2_self"].values
    r2_group = df["mean_r2_group"].values

    avg_r2_self = np.nanmean(r2_self)
    avg_r2_group = np.nanmean(r2_group)

    x = np.arange(len(tast_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, r2_self, width, label="Taster Model", color="tab:blue")
    ax.bar(x + width/2, r2_group, width, label="Group Model", color="tab:orange")

    # Plot horizontal lines for averages
    ax.axhline(avg_r2_self, color="tab:blue", linestyle="--", linewidth=1.5,
               label=f"Mean R² (Taster): {avg_r2_self:.3f}")
    ax.axhline(avg_r2_group, color="tab:orange", linestyle="--", linewidth=1.5,
               label=f"Mean R² (Group): {avg_r2_group:.3f}")

    # Customize x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(tast_ids)

    # Customize y-axis
    all_values = np.concatenate([r2_self, r2_group])
    min_y = min(np.min(all_values), avg_r2_self, avg_r2_group)
    max_y = max(np.max(all_values), avg_r2_self, avg_r2_group)
    pad = (max_y - min_y) * 0.1
    ax.set_ylim(min_y - pad, max_y + pad)

    ax.set_ylabel("Mean R²")
    ax.set_title("Mean R²: Self vs Group Models per Taster")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def plot_r2_comparison():
    # Hardcoded R² values: rows = models, columns = methods
    methods = [
        "OHE",
        "OHE + Taster Scaling",
        "OHE + Shuffle Labels",
        "Average Scores",
        "Individual Tasters",
        "N-1 Tasters"
    ]

    models = [
        "Ridge",
        "Random Forest",
        "SVR"
    ]

    # Example R² scores (shape: len(models) x len(methods))
    r2_scores = np.array([
        [0.248, 0.257, -0.153, 0.383, -1.041, 0.365],  # Ridge
        [0.265, 0.270, -0.140, 0.395, -0.950, 0.370],  # Random Forest
        [0.230, 0.245, -0.170, 0.375, -1.100, 0.350],  # SVR
    ])

    x = np.arange(len(models))
    width = 0.13  # width of each bar

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each method
    for i, method in enumerate(methods):
        ax.bar(x + i * width, r2_scores[:, i], width, label=method)

    # Labels, title, and legend
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel("R² Value")
    ax.set_title("R² Comparison Across Regression Models and Test Methods")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=0)

    # Annotate bars with R² values
    for i in range(len(models)):
        for j in range(len(methods)):
            height = r2_scores[i, j]
            ax.text(
                x[i] + j * width,
                height + 0.02 if height >= 0 else height - 0.05,
                f"{height:.2f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8
            )

    plt.tight_layout()
    plt.show()

import seaborn as sns
def plot_r2_comparison_heatmap():
    # data = {
    #     "Individual Tasters": {
    #         "KNN": -0.141,
    #         "Ridge": -0.894,
    #         "ElasticNet": -0.513,
    #         "Lasso": -0.612,
    #         "Random Forest": 0.086,
    #     },
    #     "Shuffle Labels": {
    #         "KNN": -0.186,
    #         "Ridge": -0.158,
    #         "ElasticNet": -0.029,
    #         "Lasso": -0.015,
    #         "Random Forest": -0.006,
    #     },
    #     "OHE": {
    #         "KNN": -0.025,
    #         "Ridge": 0.233,
    #         "ElasticNet": 0.329,
    #         "Lasso": 0.344,
    #         "Random Forest": 0.403,
    #     },
    #     "OHE + Taster Scaling": {
    #         "KNN": 0.146,
    #         "Ridge": 0.242,
    #         "ElasticNet": 0.354,
    #         "Lasso": 0.357,
    #         "Random Forest": 0.400,
    #     },
    #     "N-1 Group Avg": {
    #         "KNN": 0.400,
    #         "Ridge": 0.345,
    #         "ElasticNet": 0.563,
    #         "Lasso": 0.569,
    #         "Random Forest": 0.575,
    #     },
    #     "Average Scores": {
    #         "KNN": 0.408,
    #         "Ridge": 0.364,
    #         "ElasticNet": 0.578,
    #         "Lasso": 0.583,
    #         "Random Forest": 0.591,
    #     },
    # }
    data = {
        "Individual Tasters": {
            "KNN": -0.152,
            "Ridge": -1.208,
            "ElasticNet": -0.556,
            "Lasso": -0.774,
            "Random Forest": 0.026,
        },
        "Subtract Avg Scores": {
            "KNN": -0.567,
            "Ridge": -0.372,
            "ElasticNet": -0.109,
            "Lasso": -0.121,
            "Random Forest": -0.063,
        },
        "Shuffle Labels": {
            "KNN": -0.193,
            "Ridge": -0.155,
            "ElasticNet": -0.024,
            "Lasso": -0.027,
            "Random Forest": -0.006,
        },
        "Constant OHE": {
            "KNN": -0.117,
            "Ridge": 0.044,
            "ElasticNet": 0.157,
            "Lasso": 0.152,
            "Random Forest": 0.185,
        },
        "OHE": {
            "KNN": -0.025,
            "Ridge": 0.245,
            "ElasticNet": 0.334,
            "Lasso": 0.342,
            "Random Forest": 0.408,
        },
        "OHE + Taster Scaling": {
            "KNN": 0.148,
            "Ridge": 0.253,
            "ElasticNet": 0.360,
            "Lasso": 0.355,
            "Random Forest": 0.405,
        },
        # "N-1 Group Avg": {
        #     "KNN": 0.422,
        #     "Ridge": 0.343,
        #     "ElasticNet": 0.557,
        #     "Lasso": 0.588,
        #     "Random Forest": 0.596,
        # },
        "Average Scores": {
            "KNN": 0.431,
            "Ridge": 0.361,
            "ElasticNet": 0.592,
            "Lasso": 0.601,
            "Random Forest": 0.608,
        },
    }

    df = pd.DataFrame(data).T  # Transpose so methods are rows, regressors are columns

    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="Blues", linewidths=0.5, linecolor='gray')
    plt.title("R² Comparison Across Test Methods and Regression Models")
    plt.ylabel("Test Method", size=14)
    plt.xlabel("Regression Model", size=14)
    plt.tight_layout()
    plt.show()

from matplotlib import colormaps
def plot_champagne_projection(
    embedding,
    title,
    wine_ids,
    color_values=None,         # Optional, e.g., sweetness or floral prediction
    attribute_name=None,
    test_sample_names=None,
    unique_samples_only=False,
    n_neighbors=None,
    random_state=None,
    invert_x=False,
    invert_y=False
):
    """
    Scatter plot of wines using UMAP/PCA/t-SNE projections.

    Parameters
    ----------
    embedding : np.ndarray
        Projected coordinates (n_samples, 2 or 3)
    title : str
        Title of the plot
    wine_ids : list of str
        Wine sample names (for hover/label)
    color_values : np.ndarray, optional
        Values to color the dots (e.g., predicted floral score)
    test_sample_names : list of str
        Names to annotate (usually same as wine_ids)
    unique_samples_only : bool
        If True, deduplicate based on name
    n_neighbors, random_state : int, optional
        For subtitle
    invert_x, invert_y : bool
        Flip axes
    """
    """
       Scatter plot of wines using UMAP/PCA/t-SNE projections.

       Parameters
       ----------
       embedding : np.ndarray
           Projected coordinates (n_samples, 2 or 3)
       title : str
           Title of the plot
       wine_ids : list of str
           Wine sample names (for hover/label)
       color_values : np.ndarray, optional
           Values to color the dots (e.g., predicted floral score)
       attribute_name : str, optional
           Name of the sensory attribute used for coloring
       test_sample_names : list of str
           Names to annotate (usually same as wine_ids)
       unique_samples_only : bool
           If True, deduplicate based on name
       n_neighbors, random_state : int, optional
           For subtitle
       invert_x, invert_y : bool
           Flip axes
       """
    import matplotlib.pyplot as plt
    import numpy as np

    is_3d = embedding.shape[1] == 3
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d' if is_3d else None)

    if invert_x:
        embedding[:, 0] *= -1
    if invert_y:
        embedding[:, 1] *= -1

    if color_values is not None:
        colorbar_label = f"{attribute_name} intensity" if attribute_name else "Color scale"
        if is_3d:
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=color_values, cmap='viridis', s=80
            )
        else:
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=color_values, cmap='viridis', s=80
            )
        plt.colorbar(scatter, ax=ax, label=colorbar_label)
    else:
        if is_3d:
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], color='dodgerblue', s=80)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], color='dodgerblue', s=80)

    if test_sample_names is not None:
        for i, name in enumerate(test_sample_names):
            ax.annotate(name, (embedding[i, 0], embedding[i, 1]), fontsize=9, alpha=0.6, xytext=(2, 2),
                        textcoords="offset points")

    subtitle = ""
    if n_neighbors is not None:
        subtitle += f"n_neighbors={n_neighbors}  "
    if random_state is not None:
        subtitle += f"random_state={random_state}"
    full_title = title if subtitle == "" else f"{title}\n{subtitle.strip()}"

    # Append attribute to title if provided
    if attribute_name:
        full_title += f"\nColored by: {attribute_name}"

    ax.set_title(full_title, fontsize=14)
    plt.tight_layout()
    plt.show()


# Main logic
if __name__ == "__main__":
    # Load configuration parameters from YAML
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"))

    # --- Parameters ---
    N_DECIMATION = 10
    TEST_SIZE = 0.2
    CHROM_CAP = None
    directory = "/Users/juliaroquette/Work/oenology/data/Champagnes/HETEROCYC"
    # directory = "/Users/juliaroquette/Work/oenology/data/Champagnes/DMS"
    metadata_path = "/Users/juliaroquette/Work/oenology/data/Champagnes/sensory_scores.csv"
    random_seed = 42                   # For reproducibility
    column_indices = None  # or specify which columns to use
    row_start, row_end = 0, None  # if you want to trim rows

    # Load user-defined options
    wine_kind = "Champagne"
    task= "Model Global"  # hard-coded for now
    dataset = os.path.split(directory)[1]
    model_name = config.get("regressor", "ridge").lower()
    feature_type = config["feature_type"]
    cv_type = config['cv_type']
    num_repeats = config.get("num_repeats", 10)
    retention_time_range = config.get("rt_range", None)
    normalize_flag = config.get("normalize", True)
    show_taster_predictions = config.get("show_predicted_profiles", False)
    group_wines = config.get("group_wines", False)
    taster_scaling = config.get("taster_scaling", False)
    shuffle_labels = config.get("shuffle_labels", False)
    test_average_scores = config.get("test_average_scores", False)
    taster_vs_mean = config.get("taster_vs_mean", False)
    plot_all_tests = config.get("plot_all_tests", False)
    plot_r2 = config.get("plot_r2", False)
    plot_shap = config.get("plot_shap", False)
    reduce_targets_flag = config.get("reduce_dims", False)
    reduce_method = config.get("reduction_method", "pca")  # 'pca', 'umap', or 'tsne'
    reduce_dim = config.get("reduction_dims", 2)
    remove_avg_scores= config.get("remove_avg_scores", False)
    constant_ohe= config.get("constant_ohe", False)
    plot_projection = config.get("plot_projection", False)
    projection_method = config.get("projection_method", "UMAP").upper()
    projection_source = config.get("projection_source", False) if plot_projection else False
    projection_dim = config.get("projection_dim", 2)
    n_neighbors = config.get("n_neighbors", 30)
    perplexity = config.get("perplexity", 5)
    random_state = config.get("random_state", 42)
    selected_attribute = config.get("selected_attribute", "fruity")



    summary = {
        "Wine kind": wine_kind,
        "Task": task,
        "Dataset": dataset,
        "Regressor": model_name,
        "Feature type": feature_type,
        "CV type": cv_type,
        "Repeats": num_repeats,
        "RT range": retention_time_range,
        "Normalize": normalize_flag,
        "Group wines": group_wines,
        "Taster scaling": taster_scaling,
        "Shuffle labels": shuffle_labels,
        "Avg. scores": test_average_scores,
        "Taster vs mean": taster_vs_mean,
        "Remove average scores": remove_avg_scores,
        "Use constant OHE": constant_ohe,
        "Show taster profiles": show_taster_predictions,
        "Plot all tests": plot_all_tests
    }

    if plot_all_tests:
        # plot_r2_comparison()
        plot_r2_comparison_heatmap()
        plt.show()
        sys.exit(0)

    logger_raw("\n")  # Blank line without timestamp
    logger.info('------------------------ RUN SCRIPT -------------------------')
    logger.info("Configuration Parameters (Champagne - Model Global)")
    for k, v in summary.items():
        logger_raw(f"{k:>22s}: {v}")

    # Select the model based on user input
    model = get_model(model_name, random_seed=random_seed)

    # --- Load chromatograms ---
    data_dict, chome_length = load_chromatograms_decimated(
        directory, row_start, row_end, N_DECIMATION, retention_time_range=retention_time_range
    )

    # --- Load metadata ---
    metadata = load_and_clean_metadata(metadata_path)

    # --- Average duplicates ---
    known_metadata = ['code vin', 'taster', 'prod area', 'variety', 'cave', 'age']
    sensory_cols = [col for col in metadata.columns if col not in known_metadata and pd.api.types.is_numeric_dtype(metadata[col])]
    metadata[sensory_cols] = metadata[sensory_cols].apply(pd.to_numeric, errors='coerce')
    metadata = metadata.groupby(['code vin', 'taster'], as_index=False)[sensory_cols].mean()

    # --- Build input (X) and output (y) ---
    X_raw, y, taster_ids, sample_ids = [], [], [], []
    skipped_count = 0

    # Extract input features and target attributes from metadata and matched chromatograms
    X_raw, y, taster_ids, sample_ids = build_feature_target_arrays(metadata, sensory_cols, data_dict)

    if shuffle_labels:
        logger.info("Shuffle Labels selected")
        np.random.seed(random_seed)
        y = y.copy()
        np.random.shuffle(y)
        print("⚠️ Sensory labels have been shuffled across samples (diagnostic run).")

    # One-hot encode taster identities
    encoder = OneHotEncoder(sparse_output=False)
    taster_onehot = encoder.fit_transform(np.array(taster_ids).reshape(-1, 1))
    if constant_ohe:
        taster_onehot = np.ones_like(taster_onehot)

    X_input = np.concatenate([X_raw, taster_onehot], axis=1)

    # Create a mask to filter out any samples with NaNs in input or output
    mask = ~np.isnan(X_input).any(axis=1) & ~np.isnan(y).any(axis=1)
    X_input = X_input[mask]
    y = y[mask]
    taster_ids = np.array(taster_ids)[mask]
    sample_ids = np.array(sample_ids)[mask]

    if plot_shap:
        # Average all tasters
        X_avg, y_avg, wine_ids = average_by_wine(X_input, y, sample_ids)

        compute_shap(
        X=X_avg,
        y=y_avg,
        model=model,
        sensory_cols=sensory_cols,
        retention_time_range=retention_time_range,  # from config or None
        decimation_factor=N_DECIMATION,
        normalize=normalize_flag
        )
        sys.exit(0)

    print(f"Removed {np.sum(~mask)} samples with NaNs.")

    all_mae, all_rmse, all_r2 = [], [], []
    taster_mae_summary = defaultdict(list)
    last_y_test, last_y_pred, last_sample_ids, last_taster_ids = None, None, None, None
    saved_model_coefs = []

    if test_average_scores:
        logger.info("Average Scores selected")

        if plot_projection:
            if projection_source == "avgtics":
                X_avg, y_avg, wine_ids = average_by_wine(X_input, y, sample_ids)
                data_for_projection = StandardScaler().fit_transform(X_avg)
                reducer = DimensionalityReducer(data_for_projection)

                show_sample_names = True
                plot_title = f"{projection_method} of Averaged TICs"
                color_values = y_avg[:, sensory_cols.index(selected_attribute)] if selected_attribute in sensory_cols else None

            elif projection_source == "shap":
                X_avg, y_avg, wine_ids = average_by_wine(X_input, y, sample_ids)
                shap_array, rt = compute_shap(X_avg, y_avg, model, sensory_cols=sensory_cols,  return_values_only=True)

                # Collapse SHAP to 2D (samples x features) by averaging over attributes
                attr_idx = sensory_cols.index(selected_attribute)
                data_for_projection = shap_array[:, :, attr_idx]
                # data_for_projection = shap_array.mean(axis=2)  # this would be for general SHAP for all attributes
                data_for_projection = StandardScaler().fit_transform(data_for_projection)

                reducer = DimensionalityReducer(data_for_projection)
                color_values = y_avg[:, attr_idx]  # color using same attribute
                show_sample_names = True
                plot_title = f"{projection_method} of SHAP (Attribute = {selected_attribute})"

            elif projection_source == "pscores":
                X_avg, y_avg, wine_ids = average_by_wine(X_input, y, sample_ids)

                # Train and predict with model
                model.fit(X_avg, y_avg)
                y_pred = model.predict(X_avg)  # shape (n_samples, n_attributes)

                # Use predicted scores as projection features
                data_for_projection = StandardScaler().fit_transform(y_pred)

                reducer = DimensionalityReducer(data_for_projection)
                attr_idx = sensory_cols.index(selected_attribute)
                color_values = y_avg[:, attr_idx]  # coloring by ground truth of selected attribute

                show_sample_names = True
                plot_title = f"{projection_method} of Predicted Scores (Attribute = {selected_attribute})"

            else:
                raise ValueError(f"Unknown projection source: {projection_source}")

            if projection_method == "UMAP":
                embedding = reducer.umap(components=projection_dim, n_neighbors=n_neighbors, random_state=random_state)
            elif projection_method == "PCA":
                embedding = reducer.pca(components=projection_dim)
            elif projection_method == "T-SNE":
                embedding = reducer.tsne(components=projection_dim, perplexity=perplexity, random_state=random_state)
            else:
                raise ValueError(f"Unsupported projection method: {projection_method}")

            plot_champagne_projection(
                embedding,
                title=plot_title,
                wine_ids=wine_ids,
                color_values=color_values,
                attribute_name=selected_attribute,
                test_sample_names=wine_ids if show_sample_names else None,
                unique_samples_only=False,
                n_neighbors=n_neighbors,
                random_state=random_state,
                invert_x=False,
                invert_y=False
            )
            print("Entered Plot Projection with averaged TICs")
            sys.exit(0)


        results = train_and_evaluate_average_scores_model(
            X_input=X_input,
            y=y,
            sample_ids=sample_ids,
            model=model,
            num_repeats=num_repeats,
            test_size=TEST_SIZE,
            normalize=normalize_flag,
            random_seed=random_seed,
            reduce_targets=reduce_targets_flag,
            reduction_method=reduce_method,
            reduced_dim=reduce_dim
        )
        (all_mae, all_rmse, all_r2, all_y_test, all_y_pred, all_sample_ids, all_taster_ids, last_y_test, last_y_pred,
         last_sample_ids, last_taster_ids) = results

        # Averaging predictions and true values across repeats/folds
        y_true_avg, y_pred_avg, _, _ = average_predictions_across_repeats(
            all_y_test, all_y_pred, all_sample_ids, None
        )

        r2_per_attr = []
        for i, col in enumerate(sensory_cols):
            try:
                r2 = r2_score(y_true_avg[:, i], y_pred_avg[:, i])
            except ValueError:
                r2 = float("nan")
            r2_per_attr.append(r2)

        overall_avg_r2 = np.nanmean(r2_per_attr)

        logger.info("\nPer-attribute R² after averaging predictions across repeats/folds:")
        for col, r2 in zip(sensory_cols, r2_per_attr):
            line = f"{col:<20} R² = {r2:.3f}"
            logger_raw(line)
            print(line)

        logger_raw(f"Overall average R² across attributes: {overall_avg_r2:.3f}")
        print(f"Overall average R² across attributes: {overall_avg_r2:.3f}")
        sys.exit(0)

    if taster_vs_mean:
        logger.info("Taster vs Mean selected")
        unique_tasters = np.unique(taster_ids)
        results = []

        results_df = compare_self_vs_group_models_per_taster(
            X_input, y, sample_ids, taster_ids,
            model=model,
            sensory_cols=sensory_cols,
            num_repeats=num_repeats,
            group_wines=group_wines,
        )
        if plot_r2:
            plot_self_vs_group_r2_df(results_df)

        sys.exit(0)

    if remove_avg_scores:
        logger.info("Removing average scores per taster from targets.")
        import pandas as pd

        # Convert to DataFrame for easier grouping
        y_df = pd.DataFrame(y, columns=sensory_cols)
        y_df['taster'] = taster_ids

        # Compute per-taster mean per attribute
        taster_means = y_df.groupby('taster')[sensory_cols].transform('mean')

        # Subtract taster means to center scores
        y_centered = y_df[sensory_cols] - taster_means

        # Convert back to numpy array for training
        y = y_centered.values


    results = train_and_evaluate_model(
        X_input=X_input,
        y=y,
        taster_ids=taster_ids,
        sample_ids=sample_ids,
        model=model,
        num_repeats=num_repeats,
        test_size=TEST_SIZE,
        normalize=normalize_flag,
        taster_scaling=taster_scaling,
        group_wines=group_wines,
        random_seed=random_seed,
        reduce_targets=reduce_targets_flag,
        reduction_method= reduce_method,
        reduced_dim=reduce_dim
    )

    (all_mae, all_rmse, taster_mae_summary,
     taster_r2_summary, all_r2_values,
     all_y_test, all_y_pred,
     all_sample_ids, all_taster_ids,
     last_y_test, last_y_pred,
     last_sample_ids, last_taster_ids) = results

    # all_mae, all_rmse, taster_mae_summary, last_y_test, last_y_pred, last_sample_ids, last_taster_ids = results

    y_true_avg, y_pred_avg, sample_ids_avg, taster_ids_avg = average_predictions_across_repeats(
        all_y_test=all_y_test,
        all_y_pred=all_y_pred,
        all_sample_ids=all_sample_ids,
        all_taster_ids=all_taster_ids
    )

    wine_counts = Counter(last_sample_ids)
    multi_taster_wines = [wine for wine, count in wine_counts.items() if count > 1]

    mean_mae = np.mean(all_mae, axis=0)
    mean_rmse = np.mean(all_rmse, axis=0)
    rmse_pct = mean_rmse

    # mean_coef = np.mean(saved_model_coefs, axis=0)  # shape (n_outputs, n_features)

    # Separate chromatogram and taster weights
    n_chromatogram_features = X_raw.shape[1]
    n_taster_features = taster_onehot.shape[1]

    # chromatogram_weights = mean_coef[:, :n_chromatogram_features]
    # taster_weights = mean_coef[:, n_chromatogram_features:]

    unique_tasters = sorted(set(taster_ids), key=natural_key)

    if reduce_targets_flag:
        # Use generic names for reduced dimensions: Dim 1, Dim 2, ...
        for i in range(y.shape[1]):
            logger_raw(
                f"Dim {i + 1:>3d}: MAE = {mean_mae[i]:5.2f}, RMSE = {mean_rmse[i]:5.2f} ({rmse_pct[i]:.1f}% of scale)")
    else:
        for i, col in enumerate(sensory_cols):
            logger_raw(
                f"{col:>16s}: MAE = {mean_mae[i]:5.2f}, RMSE = {mean_rmse[i]:5.2f} ({rmse_pct[i]:.1f}% of scale)")
    # logger.info("\nPer-attribute average errors over multiple splits:")
    # for i, col in enumerate(sensory_cols):
    #     logger_raw(f"{col:>16s}: MAE = {mean_mae[i]:5.2f}, RMSE = {mean_rmse[i]:5.2f} ({rmse_pct[i]:.1f}% of scale)")

    logger_raw(f"Overall RMSE across repeats: {np.sqrt(np.mean(mean_rmse**2)):.4f}")

    if not test_average_scores:
        logger.info("\nPer-taster average MAE (across all attributes):")
        for taster in sorted(taster_mae_summary.keys(), key=natural_key):
            all_errors = np.array(taster_mae_summary[taster])
            avg_mae_per_attr = np.mean(all_errors, axis=0)
            overall_avg = np.mean(avg_mae_per_attr)
            logger_raw(f"Taster {taster}: MAE = {overall_avg:.2f}")

        # R² estimation
        overall_avg_r2 = np.mean(all_r2_values) if len(all_r2_values) > 0 else float('nan')
        print(f"Overall average R² (all tasters combined): {overall_avg_r2:.3f}")
        logger_raw(f"Overall average R² (all tasters combined): {overall_avg_r2:.3f}")

        if plot_r2:
            plot_r2_per_taster(taster_r2_summary, title="Average R² per taster (across all attributes)")
    else:
        logger.info("\nSkipped per-taster MAE and R² because test_average_scores=True")


    if show_taster_predictions:
        logger.info("\nPlotting all predicted sensory profiles grouped by taster (single figure)...")
        plot_profiles_grouped_by_taster(
            y_true=y_true_avg,
            y_pred=y_pred_avg,
            sample_ids=sample_ids_avg,
            taster_ids=taster_ids_avg,
            n_cols=10
        )
        plt.show()
    # if show_taster_predictions:
    #     logger.info("\nPlotting all predicted sensory profiles grouped by taster (single figure)...")
    #     plot_profiles_grouped_by_taster(
    #         y_true=last_y_test,
    #         y_pred=last_y_pred,
    #         sample_ids=last_sample_ids,
    #         taster_ids=last_taster_ids,
    #         n_cols=10
    #     )
    #     plt.show()



