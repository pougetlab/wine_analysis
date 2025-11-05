import os
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml
import math
from gcmswine.classification import Classifier
from plot_projection import compute_projection, plot_projection_with_labels


"""
Taster Score Classification using the script **train_test_labels.py**

This script trains and evaluates a Ridge classifier to predict categorical labels (e.g. taster identity, wine variety,
cave, age) based on averaged sensory evaluation scores of Champagne wines.
The features correspond to numerical scores given by tasters on different attributes (e.g. acid, balance),
and the labels are drawn from associated metadata.

Data is preprocessed by collapsing replicates per (wine, taster) pair and cleaning non-numeric values.
Stratified K-Fold or Group K-Fold cross-validation is repeated multiple times to obtain a robust estimate of
classification accuracy, and a normalized confusion matrix is plotted at the end for each classification target
to visualize model performance across classes.
"""

if __name__ == "__main__":
    # --- Load config file (config.yaml) from project root ---
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- Parse configurable parameters with default fallbacks ---
    label_columns = config.get("label_targets", ["taster"])
    if isinstance(label_columns, str):
        label_columns = [label_columns]

    show_confusion = config.get("show_confusion_matrix", False)
    normalize = config.get("normalize", True)
    classifier = config.get("classifier", "RGC")
    projection_enabled = config.get("plot_projection", False)
    projection_source = config.get("projection_source", "scores")
    projection_method = config.get("projection_method", "UMAP")
    projection_dim = config.get("projection_dim", 2)
    n_neighbors = config.get("n_neighbors", 15)
    perplexity = config.get("perplexity", 30)
    random_state = config.get("random_state", 42)
    n_repeats = config.get("num_repeats", 10)

    # --- Instantiate Classifier class (wrapper around scikit-learn models) ---
    dummy_data = np.zeros((1, 10))         # 1 sample, 10 features
    dummy_labels = np.array(["A"])
    clf = Classifier(data=dummy_data, labels=dummy_labels, classifier_type=classifier)

    if normalize:
        print("🔄 Normalized features using z-score (per CV fold)")

    # --- Parameters ---
    print(f"Using label target from config.yaml: {label_columns}")
    csv_path = "/Users/juliaroquette/Work/oenology/data/Champagnes/sensory_scores.csv"
    n_splits = 10
    # n_repeats = 20 # Repeat the cross validation to stabilize results
    random_seed = 42

    # --- Load and clean dataset ---
    df = pd.read_csv(csv_path, skiprows=1)
    df = df.iloc[1:]  # Skip redundant second header row
    df.columns = [col.strip().lower() for col in df.columns]  # Normalize column names

    # --- Prepare feature table ---
    df_selected = df.loc[:, 'code vin':'ageing']
    df_selected['taster'] = df['taster']

    # Convert numeric sensory scores; coerce invalids to NaN
    df_selected.iloc[:, 1:-1] = df_selected.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce')
    df_selected = df_selected.dropna(axis=1, how='all')

    # Group by wine and taster, averaging repeated tastings
    df_grouped = df_selected.groupby(['code vin', 'taster']).mean().dropna().reset_index()

    # --- Prepare containers for results ---
    conf_matrices = []
    label_encoders = []
    class_counts_per_label = []
    summary_stats = []

    # --- Loop through classification targets ---
    for label_column in label_columns:
        labels_df = df[['code vin', label_column, 'taster']].dropna()
        labels_df = labels_df.drop_duplicates(subset=['code vin', 'taster'])
        labels_df.columns = ['code vin', 'label', 'taster']

        merged_df = df_grouped.merge(labels_df, on=['code vin', 'taster'])

        def balance_classes(df, label_col, max_samples_per_class=10, random_state=42):
            """Optional helper to limit max samples per class (currently unused)."""
            return (
                df.groupby(label_col, group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=random_state))
                .reset_index(drop=True)
            )
        # merged_df = balance_classes(merged_df, label_col='label', max_samples_per_class=10)

        wine_labels = merged_df['label'].values
        feature_matrix = merged_df.drop(columns=['code vin', 'taster', 'label']).to_numpy()

        le = LabelEncoder()
        y = le.fit_transform(wine_labels)
        sample_counts = np.bincount(y)
        class_counts = dict(zip(le.classes_, sample_counts))
        class_counts_per_label.append(class_counts)

        # # Print sample counts per class
        # print("Samples per class:")
        # for cls, count in class_counts.items():
        #     print(f"  {cls}: {count}")

        balanced_accuracies = []
        raw_accuracies = []
        all_projection_scores = []
        all_projection_labels = []

        for repeat in tqdm(range(n_repeats)):
            all_true = []
            all_pred = []

            # --- Choose appropriate cross-validator depending on target type ---
            if label_column.lower() == "taster":
                # Stratify by label to preserve class balance
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                           random_state=random_seed + repeat)
                split_args = (feature_matrix, y)
            else:
                # Group by wine to prevent leakage across tasters of the same wine
                groups = merged_df["code vin"]
                splitter = GroupKFold(n_splits=n_splits)
                split_args = (feature_matrix, y, groups)

            # --- Run CV ---
            for train_idx, test_idx in splitter.split(*split_args):
                X_train, X_test = feature_matrix[train_idx], feature_matrix[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if normalize:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                # Re-instantiate classifier for each fold to avoid state carry-over
                clf = Classifier(data=None, labels=None, classifier_type=classifier)
                # clf = RidgeClassifier(class_weight='balanced')
                clf.classifier.fit(X_train, y_train)
                y_pred = clf.classifier.predict(X_test)

                if projection_enabled and projection_source == "scores":
                    if hasattr(clf.classifier, "decision_function"):
                        scores = clf.classifier.decision_function(X_test)
                        # Ensure same number of columns across folds
                        if scores.shape[1] < len(le.classes_):
                            pad = len(le.classes_) - scores.shape[1]
                            scores = np.pad(scores, ((0, 0), (0, pad)), mode='constant')
                        elif scores.shape[1] > len(le.classes_):
                            scores = scores[:, :len(le.classes_)]
                        all_projection_scores.append(scores)
                        all_projection_labels.append(y_test)
                    else:
                        print("⚠️ Classifier does not support decision_function; skipping score projection.")

                all_true.extend(y_test)
                all_pred.extend(y_pred)

            # Compute metrics for this repeat
            bal_acc = balanced_accuracy_score(all_true, all_pred)
            raw_acc = accuracy_score(all_true, all_pred)
            balanced_accuracies.append(bal_acc)
            raw_accuracies.append(raw_acc)

        mean_bal = np.mean(balanced_accuracies)
        std_bal = np.std(balanced_accuracies)
        mean_raw = np.mean(raw_accuracies)
        std_raw = np.std(raw_accuracies)

        print(f"\n📊 Results for label: '{label_column}'")
        print(f"   ✅ Balanced Accuracy: {mean_bal:.3f} ± {std_bal:.3f}")
        print(f"   ℹ️  Raw Accuracy     : {mean_raw:.3f} ± {std_raw:.3f}")

        summary_stats.append((label_column, mean_bal, std_bal, mean_raw, std_raw))

        cm = confusion_matrix(all_true, all_pred, labels=np.arange(len(le.classes_)), normalize='true')
        conf_matrices.append(cm)
        label_encoders.append(le)

    # --- Plot all confusion matrices ---
    n_labels = len(label_columns)
    cols = min(n_labels, 2)
    rows = math.ceil(n_labels / cols)

    if show_confusion:
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
        if n_labels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (cm, le, title) in enumerate(zip(conf_matrices, label_encoders, label_columns)):
            print(f"\nLabel column: {title}")
            print("Label order (classes_):", list(le.classes_))
            print("Class counts:", class_counts_per_label[i])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
            disp.plot(ax=axes[i], cmap="Blues", values_format=".2f", xticks_rotation=45, colorbar=False)
            axes[i].set_title(f"Confusion Matrix – {title}")

        plt.tight_layout()
        plt.show()

    print("\n📋 Overall Summary:")
    for label, mean_bal, std_bal, mean_raw, std_raw in summary_stats:
        print(f"  {label:<12} | Balanced: {mean_bal:.3f} ± {std_bal:.3f} | Raw: {mean_raw:.3f} ± {std_raw:.3f}")

    # --- Optional projection of samples in 2D/3D space ---
    if projection_enabled:
        if projection_source == "scores":
            if all_projection_scores:
                X_to_plot = np.vstack(all_projection_scores)
                y_to_plot = np.hstack(all_projection_labels)
            else:
                print("⚠️ No scores collected for projection; skipping plot.")
                X_to_plot = None
                y_to_plot = None
        else:
            X_to_plot = feature_matrix
            y_to_plot = y

        if X_to_plot is not None:
            X_proj = compute_projection(
                X=X_to_plot,
                method=config.get("projection_method", "UMAP"),
                dim=config.get("projection_dim", 2),
                random_state=config.get("random_state", 42),
                n_neighbors=config.get("n_neighbors", 10),
                perplexity=config.get("perplexity", 5)
            )

            plot_projection_with_labels(
                X_proj,
                y_to_plot,
                method="UMAP",
                n_components=projection_dim,
                label_encoder=le,
                n_neighbors=30,
                random_state=42
            )
