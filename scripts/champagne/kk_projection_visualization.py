"""
Projection Visualization
========================
In this section, we focus on the classification of Champagne wine samples using the script **train_test_champagne.py**.
The dataset includes sensory profiles from 10 tasters across 50 wines of the same vintage, with each wine rated along
187 sensory dimensions: fruity, citrus, mature, candied, toasted, nuts, spicy, petrol, undergrowth, bakery, honey,
dairy, herbal, tobacco, texture, acid, and aging.

There is no 3D GC-MS data for this dataset, meaning we have to deal directly with TICs as individual m/z channels are not available.

Sensory Data Visualization
--------------------------

The script **kk_projection_visualization.py** provides a flexible tool to visualize sensory data for Champagne wine samples.
It enables exploratory analysis of sensory descriptors using unsupervised methods to uncover latent patterns related to
ageing, variety, production site, or taster.

This script operates on a dataset of 50 wines, each evaluated across 187 sensory dimensions by 12 different tasters.
Unlike other datasets in the project, this one does **not include full 3D GC-MS chromatograms**. Therefore, the
analysis is performed directly on the available **Total Ion Current (TIC)** data aggregated at the sensory level.

The main steps of the script are as follows:

1. **Data Loading and Cleaning**:

   - Loads a CSV file containing sensory ratings for Champagne wines.
   - Cleans column headers, removes duplicated rows, and ensures numerical consistency.
   - Averages replicate measurements for each wine and taster.

2. **Label Handling**:

   - Allows the user to select a column (e.g., `age`, `variety`, `cave`, `taster`) to use as the target label for coloring or annotating the plots.
   - Handles missing labels and ensures one label per wine-taster pair.

3. **Standardization and Dimensionality Reduction**:

   - Standardizes all sensory features using `StandardScaler`.
   - Applies three different dimensionality reduction techniques:

     - **PCA** (Principal Component Analysis)
     - **t-SNE** (t-Distributed Stochastic Neighbor Embedding)
     - **UMAP** (Uniform Manifold Approximation and Projection)

4. **Clustering (Optional)**:

   - Optionally applies **KMeans clustering** to the PCA-reduced data.
   - Evaluates silhouette scores to automatically determine the optimal number of clusters (k).

5. **Plotting and Visualization**:

   - Generates 2D or 3D scatter plots based on the chosen dimensionality reduction method.
   - Colors points either by cluster membership (if clustering is enabled) or by known labels (e.g., ageing).
   - Optionally displays sample labels on the plot for better interpretability.

6. **Interactivity**:

   - Modifiable parameters at the top of the script allow for easy reconfiguration:

     - `label_column`: determines what label to use for coloring
     - `plot_3d`: toggles 3D vs 2D visualization
     - `do_kmeans`: enables/disables clustering
     - `show_point_labels`: controls whether labels are shown on points

This visualization script is particularly useful for:

- Identifying latent structure in sensory data
- Evaluating whether tasters are consistent
- Comparing sensory signatures across caves, ageing conditions, or grape varieties
- Exploring whether known labels correspond to natural clustering in the data

Usage
~~~~~

To run the script:

.. code-block:: bash

   python scripts/champagne/kk_projection_visualization.py

Before running, modify the top of the script to select the appropriate label column:

.. code-block:: python

   label_column = 'ageing'  # or 'variety', 'cave', 'taster', etc.

Output
~~~~~~

The script generates plots showing how wines are distributed in reduced feature space, helping to visually assess
whether sensory profiles group according to the chosen label (e.g., ageing style). It also enables exploratory use
of clustering to identify potential new categories or sensory trends.
"""
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from fontTools.unicodedata import block
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder
    import umap

    # --- Parameters ---
    label_column = 'taster'  # <-- CHANGE THIS to 'prod area', 'variety',  'cave', 'age', 'ageing', etc.
    plot_3d = False  # <-- Toggle this to switch between 2D and 3D plots
    do_kmeans = False  # Set to True to enable KMeans clustering after PCA
    show_point_labels = True  # Set to False to hide text labels on plot points

    # --- Load CSV and preprocess ---
    df = pd.read_csv("/Users/juliaroquette/Work/oenology/data/Champagnes/sensory_scores.csv", skiprows=1)
    df = df.iloc[1:]  # remove second header row if needed
    df.columns = [col.strip().lower() for col in df.columns]  # lowercase and clean headers

    # Extract relevant columns and average replicates
    df_selected = df.loc[:, 'code vin':'acid']
    df_selected['taster'] = df['taster']  # add the taster info so we can group by it
    df_selected.iloc[:, 1:-1] = df_selected.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce') # all except first and last ('code vin' and 'taster')
    df_grouped = df_selected.groupby(['code vin', 'taster']).mean().dropna()

    # Prepare labels
    labels_df = df[['code vin', label_column, 'taster']].dropna()
    labels_df = labels_df.drop_duplicates(subset=['code vin', 'taster'])
    labels_df.columns = ['code vin', 'label', 'taster']
    df_grouped = df_grouped.reset_index().merge(labels_df, on=['code vin', 'taster'])
    wine_labels = df_grouped['label'].values
    df_grouped = df_grouped.drop(columns=['code vin', 'taster', 'label'])

    # --- Standardize and PCA ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_grouped)

    n_components_pca = 3 if plot_3d else 2
    pca = PCA(n_components=n_components_pca)
    X_pca = pca.fit_transform(X_scaled)

    # --- PCA Plot ---
    # fig = plt.figure(figsize=(10, 8))
    if do_kmeans:
        # --- KMeans on PCA ---
        best_k = 2
        best_score = -1
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_pca)
            score = silhouette_score(X_pca, labels)
            print(f'k: {k}, silhouette score: {score:.3f}')
            if score > best_score:
                best_k = k
                best_score = score

        # Final clustering
        kmeans_final = KMeans(n_clusters=best_k, random_state=42)
        # kmeans_final = KMeans(n_clusters=10, random_state=42)
        final_labels = kmeans_final.fit_predict(X_pca)

        colors = final_labels
        legend_label = "Cluster"
        title_suffix = f"with KMeans Clusters (k={best_k})"
    else:
        # Just color by known labels
        label_encoder = LabelEncoder()
        label_ids = label_encoder.fit_transform(wine_labels)

        colors = label_ids
        legend_label = label_column
        title_suffix = f"(Colored by {label_column})"

    # 3D or 2D PCA plot
    fig = plt.figure(figsize=(10, 8))
    if plot_3d:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colors, cmap='tab20', s=60)
        if show_point_labels:
            for i, label in enumerate(wine_labels):
                ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], label, fontsize=9, alpha=0.8)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")

    else:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='tab20', s=60)
        if show_point_labels:
            for i, label in enumerate(wine_labels):
                plt.annotate(label, (X_pca[i, 0], X_pca[i, 1]), fontsize=11, alpha=1)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.title(f"PCA Projection – {title_suffix}")
    plt.show(block=False)

    # --- t-SNE Projection and Plot ---
    label_encoder = LabelEncoder()
    label_ids = label_encoder.fit_transform(wine_labels)

    n_components_tsne = 3 if plot_3d else 2
    tsne = TSNE(n_components=n_components_tsne, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    fig = plt.figure(figsize=(10, 8))
    if plot_3d:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=label_ids, cmap='tab20', s=60)
        if show_point_labels:
            for i, label in enumerate(wine_labels):
                ax.text(X_pca[i, 0], X_tsne[i, 1], X_tsne[i, 2], label, fontsize=9, alpha=0.8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")
    else:
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label_ids, cmap='tab20', s=60)
        if show_point_labels:
            for i, label in enumerate(wine_labels):
                plt.annotate(label, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=11, alpha=1)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
    plt.title(f"t-SNE of Champagne Sensory Data (Label: {label_column})")
    plt.grid(True)
    plt.colorbar(scatter, label=label_column)
    plt.tight_layout()
    plt.show(block=False)

    # --- UMAP Projection and Plot ---
    reducer = umap.UMAP(n_components=3 if plot_3d else 2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    fig = plt.figure(figsize=(10, 8))

    if plot_3d:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], c=label_ids, cmap='tab20', s=60)
        if show_point_labels:
            for i, label in enumerate(wine_labels):
                ax.text(X_pca[i, 0], X_umap[i, 1], X_umap[i, 2], label, fontsize=9, alpha=0.8)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
    else:
        scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=label_ids, cmap='tab20', s=60)
        if show_point_labels:
            for i, label in enumerate(wine_labels):
                plt.annotate(label, (X_umap[i, 0], X_umap[i, 1]), fontsize=9, alpha=1)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

    plt.title(f"UMAP of Champagne Sensory Data (Label: {label_column})")
    plt.colorbar(scatter, label=label_column)
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)