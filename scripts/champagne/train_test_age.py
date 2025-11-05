"""
Age Prediction of Champagne Wines using Ridge Regression or Classification

This script performs regression or classification analysis to predict the age of Champagne wines
based on averaged GC-MS chromatogram signals per wine sample.

Regression uses continuous age, classification bins age into discrete categories.

Repeated stratified K-Fold CV is used for evaluation.
"""

import time
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from gcmswine import utils  # Utility functions for GC-MS data loading
import os
import yaml
import matplotlib.patches as patches

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from gcmswine.logger_setup import logger, logger_raw
from scipy.stats import pearsonr

try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

if __name__ == "__main__":
    # --- Load configuration file ---
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- General parameters and defaults ---
    directory = "/Users/juliaroquette/Work/oenology/data/Champagnes/HETEROCYC"
    csv_path = "/Users/juliaroquette/Work/oenology/data/Champagnes/sensory_scores.csv"
    n_splits = 5
    random_seed = 42
    N_DECIMATION = 10  # Downsampling factor for chromatograms
    CHROM_CAP = config.get("chrom_cap", False)
    row_start, row_end = 0, None
    year_tolerances = [1, 2, 3, 4, 5]  # Tolerance window for "within ±x years"
    normalize = config.get("normalize", True)
    show_confusion = config.get("show_confusion_matrix", False)
    show_pred_plot = config.get("show_pred_plot", False)
    show_age_histograma = config.get("show_age_histogram", False)
    show_chromatograms = config.get("show_chromatograms", False)
    retention_time_range = config.get("rt_range", False)
    do_classification = config.get("do_classification", False)

    # --- Decide whether we do regression or classification ---
    if do_classification:
        classifier_name = (config.get("classifier") or "ridgeclassifier").lower()
        regressor_name = None
    else:
        classifier_name = None
        regressor_name = (config.get("regressor") or "ridge").lower()

    num_repeats = config.get("num_repeats", 10)

    # --- Basic metadata logging ---
    wine_kind = "Champagne"
    task = "Age prediction"
    dataset = os.path.split(directory)[1]
    feature_type = config["feature_type"]
    cv_type = config['cv_type']
    retention_time_range = config.get("rt_range", None)
    normalize = config.get("normalize", True)

    # Collect run summary for logging
    summary = {
        "Wine kind": wine_kind,
        "Task": task,
        "Dataset": dataset,
        "Classifier": classifier_name,
        "Regressor": regressor_name,
        "Feature type": feature_type,
        "Repeats": num_repeats,
        "RT range": retention_time_range,
        "Normalize": normalize,
        "Do Classification": do_classification
    }

    # --- Log configuration summary ---
    logger_raw("\n")
    logger.info('------------------------ RUN SCRIPT -------------------------')
    logger.info("Configuration Parameters (Champagne - Predict Age)")
    for k, v in summary.items():
        logger_raw(f"{k:>22s}: {v}")

    if normalize:
        print("🔄 Normalized features using z-score (per CV fold)")

    # ------------------ Model selection ------------------
    if do_classification:
        name = classifier_name

        # Function to dynamically create the chosen classifier
        def get_classifier(name):
            name = name.upper()
            if name == "DTC":
                from sklearn.tree import DecisionTreeClassifier
                return DecisionTreeClassifier(random_state=random_seed)
            elif name == "GNB":
                from sklearn.naive_bayes import GaussianNB
                return GaussianNB()
            elif name == "KNN":
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier()
            elif name == "LDA":
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                return LinearDiscriminantAnalysis()
            elif name == "LR":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=random_seed, max_iter=1000)
            elif name == "PAC":
                from sklearn.linear_model import PassiveAggressiveClassifier
                return PassiveAggressiveClassifier(random_state=random_seed, max_iter=1000)
            elif name == "PER":
                from sklearn.linear_model import Perceptron
                return Perceptron(random_state=random_seed, max_iter=1000)
            elif name == "RFC":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(random_state=random_seed)
            elif name == "RGC":
                from sklearn.linear_model import RidgeClassifier
                return RidgeClassifier()
            elif name == "SGD":
                from sklearn.linear_model import SGDClassifier
                return SGDClassifier(random_state=random_seed, max_iter=1000)
            elif name == "SVM":
                from sklearn.svm import SVC
                return SVC(random_state=random_seed)
            else:
                raise ValueError(f"Unknown classifier: {name}")

        model = get_classifier(classifier_name)
        print(f"Classifier is {classifier_name}")

    else:
        # Choose regression model
        model_name = regressor_name
        if model_name == "ridge":
            model = Ridge()
        elif model_name == "lasso":
            model = Lasso()
        elif model_name == "elasticnet":
            model = ElasticNet()
        elif model_name == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=random_seed)
        elif model_name == "gbr":
            model = GradientBoostingRegressor(random_state=random_seed)
        elif model_name == "hgb":
            model = HistGradientBoostingRegressor(random_state=random_seed)
        elif model_name == "svr":
            model = SVR()
        elif model_name == "knn":
            model = KNeighborsRegressor()
        elif model_name == "dt":
            model = DecisionTreeRegressor(random_state=random_seed)
        elif model_name == "xgb" and xgb_available:
            model = XGBRegressor(random_state=random_seed)
        else:
            raise ValueError(f"Unsupported or unavailable model: {model_name}")

        print(f"Regressor is {model_name}")

    # ------------------ Load chromatograms ------------------
    # Identify first and last chromatogram columns in CSVs and load all signals
    row_end_1, fc_idx_1, lc_idx_1 = utils.find_data_margins_in_csv(directory)
    column_indices = list(range(fc_idx_1, lc_idx_1 + 1))
    data_dict = utils.load_ms_csv_data_from_directories(directory, column_indices, row_start, row_end)

    # Optional chromatogram visualization for sanity check
    if show_chromatograms:
        def plot_chromatograms(data_dict, keys=None, max_traces=None, title="Chromatograms", decimation_factor=1):
            if keys is None:
                keys = list(data_dict.keys())
            if not keys:
                print("❌ No chromatogram keys found.")
                return
            plt.figure(figsize=(10, 6))
            trace_count = 0
            for key in keys:
                if trace_count == max_traces:
                    break
                chrom = data_dict[key]
                chrom = np.squeeze(chrom)
                if chrom.ndim == 1:
                    trace = chrom[::decimation_factor]
                    plt.plot(trace, label=key)
                    trace_count += 1
                else:
                    print(f"⚠️ Unexpected shape after squeeze for {key}: {chrom.shape}")
            plt.title(title)
            plt.xlabel("Retention Index (or Time)")
            plt.ylabel("Intensity")
            plt.legend(fontsize='small', loc='upper right')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        plot_chromatograms(data_dict, keys=None, max_traces=None, title="Chromatograms", decimation_factor=1)

    chrom_length = len(list(data_dict.values())[0])
    print(f'Chromatogram length: {chrom_length}')

    # Optionally apply retention time trimming
    if retention_time_range:
        min_rt = retention_time_range['min']
        raw_max_rt = retention_time_range['max']
        max_rt = min(raw_max_rt, chrom_length)
        print(f"Applying RT range: {min_rt} to {max_rt} (capped at {chrom_length})")
        data_dict = {key: value[min_rt:max_rt] for key, value in data_dict.items()}

    # ------------------ Load and clean metadata ------------------
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = [col.strip().lower() for col in df.columns]
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])

    # ------------------ Build feature matrix (X) and target vector (y) ------------------
    X, y, groups = [], [], []
    df_unique = df.drop_duplicates(subset='code vin')  # Year and chromatograms are the same for the same wine

    # Average replicates chromatograms per wine to form one chromatogram per sample
    for _, row in df_unique.iterrows():
        code_vin = row['code vin']
        if pd.isna(code_vin):
            continue
        replicate_keys = [k for k in data_dict if k.startswith(code_vin)]
        if not replicate_keys:
            continue
        replicates = np.array([data_dict[k][::N_DECIMATION] for k in replicate_keys])
        chromatogram = np.mean(replicates, axis=0).flatten()
        chromatogram = np.nan_to_num(chromatogram)
        X.append(chromatogram)
        y.append(row['age'])
        groups.append(code_vin)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    # ------------------ Stratified binning for CV splits ------------------
    wine_df = pd.DataFrame({'code_vin': groups, 'age': y})
    wine_df_unique = wine_df.drop_duplicates(subset='code_vin').copy()

    if do_classification:
        bin_edges = np.arange(np.floor(y.min()), np.ceil(y.max()) + 2, 1)  # one year intervals
    else:
        bin_edges = [0, 5, 10, 15, 20, 25, 30]
    wine_df_unique['age_bin'] = np.digitize(wine_df_unique['age'], bins=bin_edges, right=False)

    # ------------------ Cross-validation loops ------------------
    all_mae, all_rmse, all_r2, all_acc, all_acc_tol, all_y_true, all_y_pred = [], [], [], [], [], [], []
    all_resid_std, all_pearson = [], []

    for repeat in range(num_repeats):
        print(f"\n🔁 Repeat {repeat + 1}/{num_repeats}")
        mae_list, rmse_list, r2_list, resid_std_list, pearson_list = [], [], [], [], []
        acc_list, acc_tol_dict = [], {tol: [] for tol in year_tolerances}
        y_true_all, y_pred_all = [], []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed + repeat)

        # --- CV fold loop ---
        for train_idx, test_idx in skf.split(wine_df_unique['code_vin'], wine_df_unique['age_bin']):
            train_codes = wine_df_unique.iloc[train_idx]['code_vin'].values
            test_codes = wine_df_unique.iloc[test_idx]['code_vin'].values

            train_mask = np.isin(groups, train_codes)
            test_mask = np.isin(groups, test_codes)

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # --- Classification branch ---
            if do_classification:
                binning = KBinsDiscretizer(n_bins=len(bin_edges)-1, encode='ordinal', strategy='uniform')
                y_train_binned = binning.fit_transform(y_train.reshape(-1, 1)).ravel()
                y_test_binned = binning.transform(y_test.reshape(-1, 1)).ravel()

                model.fit(X_train, y_train_binned)
                y_pred_binned = model.predict(X_test)

                acc = accuracy_score(y_test_binned, y_pred_binned)
                acc_list.append(acc)

                # Treat classes as ordinal values and compute regression-style residuals
                r, _ = pearsonr(y_test_binned, y_pred_binned)
                pearson_list.append(r)
                residuals = y_test_binned - y_pred_binned
                mae_list.append(np.mean(np.abs(residuals)))
                rmse_list.append(np.sqrt(np.mean(residuals ** 2)))
                r2_list.append(r2_score(y_test_binned, y_pred_binned))
                resid_std_list.append(np.std(residuals, ddof=1))

                # Evaluate accuracy within tolerance bins
                for tol in year_tolerances:
                    acc_tol = np.mean(np.abs(y_pred_binned - y_test_binned) <= tol)
                    acc_tol_dict[tol].append(acc_tol)

                y_true_all.extend(y_test_binned)
                y_pred_all.extend(y_pred_binned)

                print(f"Classification accuracy: {acc:.3f}")

            else:
                # --- Regression branch ---
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r, _ = pearsonr(y_test, y_pred)
                pearson_list.append(r)
                residuals = y_test - y_pred
                residual_std = np.std(residuals, ddof=1)
                resid_std_list.append(residual_std)
                mae_list.append(mean_absolute_error(y_test, y_pred))
                rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2_list.append(r2_score(y_test, y_pred))
                acc_list.append(np.mean(np.round(y_pred) == np.round(y_test)))

                # Evaluate tolerance-based accuracies
                for tol in year_tolerances:
                    acc_tol_dict[tol].append(np.mean(np.abs(np.round(y_pred) - np.round(y_test)) <= tol))

                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)

                print(f"Regression R²: {r2_list[-1]:.3f}, MAE: {mae_list[-1]:.3f}")

        # Store average metrics per repeat
        all_pearson.append(np.mean(pearson_list))
        all_resid_std.append(np.mean(resid_std_list))
        all_mae.append(np.mean(mae_list) if mae_list else None)
        all_rmse.append(np.mean(rmse_list) if rmse_list else None)
        all_r2.append(np.mean(r2_list) if r2_list else None)
        all_acc.append(np.mean(acc_list) if acc_list else None)
        all_acc_tol.append({tol: np.mean(acc_tol_dict[tol]) for tol in year_tolerances})

    # ------------------ Summary metrics ------------------
    if do_classification:
        logger.info("\nCross-validated Classification Performance:")
        print("\nCross-validated Classification Performance:")
        print(f"Classifier: {classifier_name}")
        print(f"Accuracy: {np.mean(all_acc):.3f} ± {np.std(all_acc):.3f}")
        for tol in year_tolerances:
            tol_scores = [rep[tol] for rep in all_acc_tol]
            print(f"Accuracy within ±{tol} bins: {np.mean(tol_scores):.3f} ± {np.std(tol_scores):.3f}")
        print(f"Pearson R: {np.mean(all_pearson):.3f} ± {np.std(all_pearson):.3f}")
        print(f"MAE:  {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f}")
        print(f"RMSE: {np.mean(all_rmse):.2f} ± {np.std(all_rmse):.2f}")
        print(f"R²:   {np.mean(all_r2):.3f} ± {np.std(all_r2):.3f}")
        print(f"Residual Std Dev: {np.mean(all_resid_std):.2f} ± {np.std(all_resid_std):.2f}")


    else:
        logger.info("\nCross-validated Regression Performance:")
        print("\nCross-validated Regression Performance:")
        print(f"Regressor: {model_name}")
        print(f"Rounded Accuracy (Exact): {np.mean(all_acc):.3f} ± {np.std(all_acc):.3f}")
        for tol in year_tolerances:
            tol_scores = [rep[tol] for rep in all_acc_tol]
            print(f"Rounded Accuracy (±{tol} yrs): {np.mean(tol_scores):.3f} ± {np.std(tol_scores):.3f}")
        print(f"Pearson R: {np.mean(all_pearson):.3f} ± {np.std(all_pearson):.3f}")
        print(f"MAE:  {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f}")
        print(f"RMSE: {np.mean(all_rmse):.2f} ± {np.std(all_rmse):.2f}")
        print(f"R²:   {np.mean(all_r2):.3f} ± {np.std(all_r2):.3f}")
        print(f"Residual Std Dev: {np.mean(all_resid_std):.2f} ± {np.std(all_resid_std):.2f}")

    # ------------------ Visualization section ------------------
    plots_to_show = {
        "confusion": show_confusion,
        "histogram": show_age_histograma,
        "scatter": show_pred_plot,
    }

    n_plots = sum(plots_to_show.values())
    sys.stdout.flush()

    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # --- Confusion matrix ---
        if plots_to_show["confusion"]:
            if do_classification:
                y_true_arr = np.array(y_true_all).astype(int)
                y_pred_arr = np.array(y_pred_all).astype(int)
            else:
                y_true_arr = np.round(np.array(y_true_all)).astype(int)
                y_pred_arr = np.round(np.array(y_pred_all)).astype(int)

            unique_labels = np.unique(np.concatenate((y_true_arr, y_pred_arr)))
            cm = confusion_matrix(y_true_arr, y_pred_arr, labels=unique_labels, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
            disp.plot(ax=axes[plot_idx], cmap="Blues", xticks_rotation=45, colorbar=False)
            axes[plot_idx].set_title("Confusion Matrix (Rounded Age or Binned Classes)")
            ax = axes[plot_idx]
            num_classes = cm.shape[0]
            for i in range(num_classes):
                rect = patches.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                         linewidth=0.5, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
            plot_idx += 1

        # --- Scatter: true vs predicted ---
        if plots_to_show["scatter"]:
            y_pred_all = np.array(y_pred_all)
            y_true_all = np.array(y_true_all)
            if do_classification:
                error_distance = np.abs(y_pred_all - y_true_all)
                sc = axes[plot_idx].scatter(
                    y_true_all, y_pred_all, c=error_distance, s=50, cmap='coolwarm', alpha=0.8, edgecolors='none'
                )
                fig.colorbar(sc, ax=axes[plot_idx], label='Absolute Class Difference')
                axes[plot_idx].plot(
                    [min(y_true_all), max(y_true_all)],
                    [min(y_true_all), max(y_true_all)],
                    'r--', lw=2
                )
                axes[plot_idx].set_xlabel("True Age Bin")
                axes[plot_idx].set_ylabel("Predicted Age Bin")

            else:
                relative_error = np.abs(y_pred_all - y_true_all) / np.where(y_true_all != 0, y_true_all, 1)
                sc = axes[plot_idx].scatter(
                    y_true_all, y_pred_all, c=relative_error, s=50, cmap='coolwarm', alpha=0.8, edgecolors='none'
                )
                fig.colorbar(sc, ax=axes[plot_idx], label='Relative Error')
                axes[plot_idx].plot(
                    [min(y_true_all), max(y_true_all)],
                    [min(y_true_all), max(y_true_all)],
                    'r--', lw=2
                )
                axes[plot_idx].set_xlabel("True Age")
                axes[plot_idx].set_ylabel("Predicted Age")
            axes[plot_idx].set_title("True vs Predicted Age")
            axes[plot_idx].grid(True)
            plot_idx += 1

        # --- Histogram of true ages ---
        if plots_to_show["histogram"]:
            axes[plot_idx].hist(
                y, bins=np.arange(np.floor(y.min()), np.ceil(y.max()) + 1),
                color="gray", edgecolor="black"
            )
            axes[plot_idx].set_title("Age Distribution")
            axes[plot_idx].set_xlabel("True Age")
            axes[plot_idx].set_ylabel("Count")
            plot_idx += 1

        plt.tight_layout()
        plt.show()





# """
# Age Prediction of Champagne Wines using Ridge Regression
#
# This script performs regression analysis to predict the age of Champagne wines
# based on averaged GC-MS chromatogram signals per wine sample.
#
# Each sample is obtained by averaging multiple replicate chromatograms associated
# with the same wine code, decimated for dimensionality reduction.
#
# The regression model used is Ridge, applied with repeated Stratified K-Fold
# cross-validation. Stratification is done by binning the target (age) into discrete
# quantile-based bins for better balanced splits.
#
# Performance metrics such as MAE, RMSE, R², and rounded accuracy (within exact
# and ±tolerance) are reported. Additionally, visualizations include metric distributions,
# a confusion matrix of rounded predictions, and a scatter plot of true vs. predicted ages.
#
# """
# import time
#
# import pandas as pd
# import numpy as np
# import sys
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
# from tqdm import tqdm
# # import matplotlib
# # matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from gcmswine import utils  # Assumes this is your utility module for GC-MS data loading
# from sklearn.model_selection import GroupKFold
# from sklearn.utils import shuffle
# import os
# import yaml
# import matplotlib.patches as patches
#
# from sklearn.linear_model import Ridge, Lasso, ElasticNet
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# try:
#     from xgboost import XGBRegressor
#     xgb_available = True
# except ImportError:
#     xgb_available = False
#
# if __name__ == "__main__":
#     config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
#     config_path = os.path.abspath(config_path)
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#
#     # ------------------ Parameters ------------------
#     directory = "/Users/juliaroquette/Work/oenology/data/Champagnes/HETEROCYC"
#     csv_path = "/Users/juliaroquette/Work/oenology/data/Champagnes/sensory_scores.csv"
#     n_splits = 5                      # Number of CV folds
#     random_seed = 42                   # For reproducibility
#     N_DECIMATION = 10                  # Downsampling factor
#     CHROM_CAP = config.get("chrom_cap", False)
#     row_start, row_end = 0, None       # Limit which rows to read
#     year_tolerances = [1, 2, 3, 4, 5]        # Multiple tolerance values to evaluate
#     normalize = config.get("normalize", True)
#     show_confusion = config.get("show_confusion_matrix", False)
#     show_pred_plot = config.get("show_pred_plot", False)
#     show_age_histograma = config.get("show_age_histogram", False)
#     show_chromatograms = config.get("show_chromatograms", False)
#     retention_time_range = config.get("rt_range", False)
#     do_classification = config.get("do_classification", False)
#     print(retention_time_range)
#     if normalize:
#         print("🔄 Normalized features using z-score (per CV fold)")
#     num_repeats = config.get("num_repeats", 10)
#     model_name = config.get("regressor", "ridge").lower()
#
#     if do_classification:
#
#
#     else:
#         if model_name == "ridge":
#             model = Ridge()
#         elif model_name == "lasso":
#             model = Lasso()
#         elif model_name == "elasticnet":
#             model = ElasticNet()
#         elif model_name == "rf":
#             model = RandomForestRegressor(n_estimators=100, random_state=random_seed)
#         elif model_name == "gbr":
#             model = GradientBoostingRegressor(random_state=random_seed)
#         elif model_name == "hgb":
#             model = HistGradientBoostingRegressor(random_state=random_seed)
#         elif model_name == "svr":
#             model = SVR()
#         elif model_name == "knn":
#             model = KNeighborsRegressor()
#         elif model_name == "dt":
#             model = DecisionTreeRegressor(random_state=random_seed)
#         elif model_name == "xgb" and xgb_available:
#             model = XGBRegressor(random_state=random_seed)
#         else:
#             raise ValueError(f"Unsupported or unavailable model: {model_name}")
#
#     print(f'Regressor is {model_name}')
#
#     # ------------------ Load chromatograms ------------------
#     row_end_1, fc_idx_1, lc_idx_1 = utils.find_data_margins_in_csv(directory)  # Get signal column bounds
#     column_indices = list(range(fc_idx_1, lc_idx_1 + 1))
#     data_dict = utils.load_ms_csv_data_from_directories(directory, column_indices, row_start, row_end)  # Load chromatograms
#
#     if show_chromatograms:
#         def plot_chromatograms(data_dict, keys=None, max_traces=None, title="Chromatograms", decimation_factor=1):
#             """
#             Plot individual chromatograms (no averaging), handling 1D and (n, 1) formats.
#
#             Parameters:
#                 data_dict (dict): Dictionary of chromatograms.
#                 keys (list): Specific keys to plot. Defaults to all.
#                 max_traces (int): Maximum number of traces to show.
#                 title (str): Plot title.
#                 decimation_factor (int): Downsampling factor for x-axis.
#             """
#             if keys is None:
#                 keys = list(data_dict.keys())
#
#             if not keys:
#                 print("❌ No chromatogram keys found.")
#                 return
#
#             plt.figure(figsize=(10, 6))
#             trace_count = 0
#
#             for key in keys:
#                 if trace_count == max_traces:
#                     break
#
#                 chrom = data_dict[key]
#                 chrom = np.squeeze(chrom)  # Ensures shape becomes (n,) if (n,1)
#
#                 if chrom.ndim == 1:
#                     trace = chrom[::decimation_factor]
#                     plt.plot(trace, label=key)
#                     trace_count += 1
#                 else:
#                     print(f"⚠️ Unexpected shape after squeeze for {key}: {chrom.shape}")
#
#             plt.title(title)
#             plt.xlabel("Retention Index (or Time)")
#             plt.ylabel("Intensity")
#             plt.legend(fontsize='small', loc='upper right')
#             plt.grid(True)
#             plt.tight_layout()
#             plt.show()
#
#         plot_chromatograms(data_dict, keys=None, max_traces=None, title="Chromatograms", decimation_factor=1)
#
#     # if CHROM_CAP:
#     #     data_dict = {key: value[:CHROM_CAP] for key, value in data_dict.items()}  # Optionally truncate replicates
#     chrom_length = len(list(data_dict.values())[0])
#     print(f'Chromatogram length: {chrom_length}')
#
#     if retention_time_range:
#         min_rt = retention_time_range['min']
#         raw_max_rt = retention_time_range['max']
#         max_rt = min(raw_max_rt, chrom_length)
#         print(f"Applying RT range: {min_rt} to {max_rt} (capped at {chrom_length})")
#         data_dict = {key: value[min_rt:max_rt] for key, value in data_dict.items()}
#
#     # ------------------ Load and clean metadata ------------------
#     df = pd.read_csv(csv_path, skiprows=1)
#     df.columns = [col.strip().lower() for col in df.columns]
#     df['age'] = pd.to_numeric(df['age'], errors='coerce')  # Ensure age is numeric
#     df = df.dropna(subset=['age'])  # Drop rows without age
#
#     # ------------------ Prepare data matrix ------------------
#     X, y, groups = [], [], []
#     df_unique = df.drop_duplicates(subset='code vin')  # One entry per wine
#
#     for _, row in df_unique.iterrows():
#         code_vin = row['code vin']
#         if pd.isna(code_vin):
#             continue
#
#         replicate_keys = [k for k in data_dict if k.startswith(code_vin)]  # Match all replicates
#         if not replicate_keys:
#             continue
#
#         replicates = np.array([data_dict[k][::N_DECIMATION] for k in replicate_keys])  # Decimate and collect replicates
#         chromatogram = np.mean(replicates, axis=0).flatten()  # Average replicates into single sample
#         chromatogram = np.nan_to_num(chromatogram)  # Replace NaNs with 0
#
#         X.append(chromatogram)
#         y.append(row['age'])
#         groups.append(code_vin)
#
#     X = np.array(X)
#     y = np.array(y)
#     groups = np.array(groups)
#
#     # ------------------ Bin ages for stratified CV ------------------
#     wine_df = pd.DataFrame({'code_vin': groups, 'age': y})
#     wine_df_unique = wine_df.drop_duplicates(subset='code_vin').copy()
#
#     # Custom binning: 0-5, 5-10, 10-15, 15-20, 20-25, 25-30
#     bin_edges = [0, 5, 10, 15, 20, 25, 30]
#     # bin_edges = [0, 10, 20, 30]
#     wine_df_unique['age_bin'] = np.digitize(wine_df_unique['age'], bins=bin_edges, right=False)
#
#     all_mae, all_rmse, all_r2, all_acc, all_acc_tol, all_y_true, all_y_pred = [], [], [], [], [], [], []
#     for repeat in range(num_repeats):
#         print(f"\n🔁 Repeat {repeat + 1}/{num_repeats}")
#         mae_list, rmse_list, r2_list = [], [], []
#         acc_list, acc_tol_dict = [], {tol: [] for tol in year_tolerances}
#         y_true_all, y_pred_all = [], []
#
#         skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed + repeat)
#
#         for train_idx, test_idx in skf.split(wine_df_unique['code_vin'], wine_df_unique['age_bin']):
#             train_codes = wine_df_unique.iloc[train_idx]['code_vin'].values
#             test_codes = wine_df_unique.iloc[test_idx]['code_vin'].values
#
#             train_mask = np.isin(groups, train_codes)
#             test_mask = np.isin(groups, test_codes)
#
#             X_train, X_test = X[train_mask], X[test_mask]
#             y_train, y_test = y[train_mask], y[test_mask]
#
#             if normalize:
#                 scaler = StandardScaler()
#                 X_train = scaler.fit_transform(X_train)
#                 X_test = scaler.transform(X_test)
#
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#
#             mae_list.append(mean_absolute_error(y_test, y_pred))
#             rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
#             r2_list.append(r2_score(y_test, y_pred))
#             acc_list.append(np.mean(np.round(y_pred) == np.round(y_test)))
#             for tol in year_tolerances:
#                 acc_tol_dict[tol].append(np.mean(np.abs(np.round(y_pred) - np.round(y_test)) <= tol))
#
#             y_true_all.extend(y_test)
#             y_pred_all.extend(y_pred)
#
#         all_mae.append(np.mean(mae_list))
#         all_rmse.append(np.mean(rmse_list))
#         all_r2.append(np.mean(r2_list))
#         all_acc.append(np.mean(acc_list))
#         all_acc_tol.append({tol: np.mean(acc_tol_dict[tol]) for tol in year_tolerances})
#         all_y_true.extend(y_true_all)
#         all_y_pred.extend(y_pred_all)
#
#
#     print("\nCross-validated Regression Performance:")
#     print(f"Regressor: {model_name}")
#     print(f"MAE:  {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f}")
#     print(f"RMSE: {np.mean(all_rmse):.2f} ± {np.std(all_rmse):.2f}")
#     print(f"R²:   {np.mean(all_r2):.3f} ± {np.std(all_r2):.3f}")
#     print(f"Rounded Accuracy (Exact): {np.mean(all_acc):.3f} ± {np.std(all_acc):.3f}")
#     for tol in year_tolerances:
#         tol_scores = [rep[tol] for rep in all_acc_tol]
#         print(f"Rounded Accuracy (±{tol} yrs): {np.mean(tol_scores):.3f} ± {np.std(tol_scores):.3f}")
#
#     # Determine which plots are to be shown
#     plots_to_show = {
#         "confusion": show_confusion,
#         "histogram": show_age_histograma,
#         "scatter": show_pred_plot,
#     }
#
#     # Count how many plots to display
#     n_plots = sum(plots_to_show.values())
#
#     sys.stdout.flush()
#
#     if n_plots > 0:
#         fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
#         if n_plots == 1:
#             axes = [axes]  # Ensure axes is iterable
#
#         plot_idx = 0
#
#         if plots_to_show["confusion"]:
#             y_true_arr = np.round(np.array(y_true_all)).astype(int)
#             y_pred_arr = np.round(np.array(y_pred_all)).astype(int)
#             unique_labels = np.unique(np.concatenate((y_true_arr, y_pred_arr)))
#             cm = confusion_matrix(y_true_arr, y_pred_arr, labels=unique_labels, normalize='true')
#             disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
#             disp.plot(ax=axes[plot_idx], cmap="Blues", xticks_rotation=45, colorbar=False)
#             axes[plot_idx].set_title("Confusion Matrix (Rounded Age)")
#             # Bold the border of diagonal cells
#             ax = axes[plot_idx]
#             num_classes = cm.shape[0]
#
#             for i in range(num_classes):
#                 rect = patches.Rectangle((i - 0.5, i - 0.5), 1, 1,
#                                          linewidth=0.5, edgecolor='black', facecolor='none')
#                 ax.add_patch(rect)
#             plot_idx += 1
#
#         if plots_to_show["scatter"]:
#             # Calculate relative error: abs(pred - true) / true (avoid division by zero)
#             y_pred_all = np.array(y_pred_all)
#             y_true_all = np.array(y_true_all)
#             relative_error = np.abs(y_pred_all - y_true_all) / np.where(y_true_all != 0, y_true_all, 1)
#
#             plt.scatter(y_true_all, y_pred_all, c=relative_error, s=50, cmap='coolwarm', alpha=0.8, edgecolors='none')
#             plt.colorbar(label='Relative Error')
#             plt.plot([min(y_true_all), max(y_true_all)], [min(y_true_all), max(y_true_all)], 'r--', lw=2)
#             plt.xlabel("True Age")
#             plt.ylabel("Predicted Age")
#             plt.title("True vs Predicted Age with Relative Error Coloring")
#             plt.grid(True)
#
#             # axes[plot_idx].scatter(y_true_all, y_pred_all, color='#1f77b4',
#             #                        alpha=0.5, edgecolors='none')
#             # axes[plot_idx].plot([min(y_true_all), max(y_true_all)],
#             #                     [min(y_true_all), max(y_true_all)], 'r--', lw=2)
#             # axes[plot_idx].set_xlabel("True Age")
#             # axes[plot_idx].set_ylabel("Predicted Age")
#             # axes[plot_idx].set_title("True vs Predicted Age")
#             # axes[plot_idx].grid(True)
#             plot_idx += 1
#
#         if plots_to_show["histogram"]:
#             axes[plot_idx].hist(y, bins=np.arange(np.floor(y.min()), np.ceil(y.max()) + 1),
#                                 color="gray", edgecolor="black")
#             axes[plot_idx].set_title("Age Distribution")
#             axes[plot_idx].set_xlabel("True Age")
#             axes[plot_idx].set_ylabel("Count")
#             plot_idx += 1
#
#         plt.tight_layout()
#         plt.show()