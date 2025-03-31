import numpy as np
import threading

from sklearn.cluster import DBSCAN
from alphashape import alphashape


class WeightManager:
    """
    Asynchronous manager:
      - recalc_clusters_async() -> cluster in all specified dimensions,
        then build alpha-shape boundaries in (x_col, y_col).
      - recalc_boundaries_async() -> recompute main_clusters + alpha-shape boundaries
        WITHOUT re-running DBSCAN. Useful if only coverage or alpha changed.
      - compute_weights_async() -> compute row-wise weights.
    """

    def __init__(self):
        # DBSCAN parameters:
        self.eps = 0.25
        self.min_samples = 10

        # Fraction of total points to cover in "main" clusters
        self.main_cluster_coverage = 0.9

        # Alpha-shape margin (for boundary polygons)
        self.cluster_boundary_margin = 0.05
        # You can tune alpha for alphashape, or choose to optimize it
        self.alpha = 1.0

        # Thread references
        self._cluster_thread = None
        self._weight_thread = None

        # Outputs stored after clustering
        self.labels = None
        self.main_labels = None
        self.main_clusters = None
        self.boundaries = None

    def recalc_clusters_async(self, df, feature_cols, x_col, y_col, on_done=None):
        """
        Starts thread to do:
          1) DBSCAN clustering in all `feature_cols` (e.g., 8D).
          2) Identify main clusters.
          3) Build alpha-shape boundaries in the plane of (x_col, y_col).
        on_done: optional callback run after finishing (in the worker thread).
        """
        def worker():
            self._recalc_clusters_internal(df, feature_cols, x_col, y_col)
            if on_done is not None:
                on_done()

        self._cluster_thread = threading.Thread(target=worker)
        self._cluster_thread.start()

    def recalc_boundaries_async(self, df, x_col, y_col, on_done=None):
        """
        Recomputes main_clusters (using current coverage)
        and alpha-shape boundaries, WITHOUT re-running DBSCAN.
        Useful if only coverage or alpha changed.
        """
        def worker():
            self._recalc_boundaries_internal(df, x_col, y_col)
            if on_done is not None:
                on_done()

        self._cluster_thread = threading.Thread(target=worker)
        self._cluster_thread.start()

    def compute_weights_async(self, df, density_col=None, error_col=None, on_done=None):
        """
        Starts a thread to compute weights.
        on_done: optional callback run after finishing (in the worker thread).
        """
        def worker():
            self._compute_weights_internal(df, density_col, error_col)
            if on_done is not None:
                on_done()

        self._weight_thread = threading.Thread(target=worker)
        self._weight_thread.start()

    def _recalc_clusters_internal(self, df, feature_cols, x_col, y_col):
        """
        Main logic for clustering. Runs in a thread.
        """
        # 1) Perform DBSCAN on the specified feature columns
        self.labels = self._cluster(df, feature_cols)

        # 2) Identify main clusters
        self.main_clusters, self.main_labels = self.find_main_clusters(df, self.labels)


    def _recalc_boundaries_internal(self, df, x_col, y_col):
        """
        Re-run main_clusters coverage logic + alpha-shapes,
        but do NOT re-run DBSCAN. Assumes self.labels is already set.
        """
        if self.main_labels is None:
            print("No existing cluster labels. Boundaries can't be recalculated.")
            return

        # Recompute alpha-shape boundaries
        self.boundaries = self.compute_cluster_boundaries(
            df, self.main_labels, self.main_clusters, x_col, y_col
        )

    def _cluster(self, df, feature_cols):
        """
        Use scikit-learn DBSCAN with parallelization to handle large data.
        We'll do minimal progress printing since DBSCAN doesn't provide partial progress.
        """
        print("Starting DBSCAN clustering with scikit-learn...")

        data = df[feature_cols].values  # shape: (n_samples, n_features)
        print(f"Data shape for clustering: {data.shape}")
        db = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='euclidean',
            n_jobs=-1  # use all cores
        )

        labels = db.fit_predict(data)
        n_clusters = len(set(labels) - {-1})
        n_noise = np.sum(labels == -1)

        print("DBSCAN done.")
        print(f"Total clusters found (excluding noise): {n_clusters}")
        print(f"Noise points: {n_noise}")
        return labels

    def find_main_clusters(self, df, labels):
        """
        Accumulate cluster sizes (excluding noise=-1) until coverage >= main_cluster_coverage.
        Return a set of cluster IDs that are considered "main."
        """
        unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]
        cluster_sizes = [(lbl, np.sum(labels == lbl)) for lbl in unique_labels]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)

        total_points = len(df)
        cum_sum = 0
        main_clusters = set()

        for (lbl, size) in cluster_sizes:
            main_clusters.add(lbl)
            cum_sum += size
            if (cum_sum / total_points) >= self.main_cluster_coverage:
                break

        coverage = 100.0 * (cum_sum / total_points)
        print(
            f"Main clusters cover ~{coverage:.1f}% of points "
            f"(goal was {100 * self.main_cluster_coverage}%)."
        )
        print(f"Main clusters identified: {main_clusters}")

        new_labels = np.where(np.isin(labels, list(main_clusters)), 1, 0)

        return main_clusters, new_labels

    def compute_cluster_boundaries(self, df, labels, x_col, y_col):
        """
        Build alpha-shape polygons (2D) for each main cluster in columns (x_col, y_col).
          - Extract points in (x_col,y_col) for each cluster.
          - Compute the alpha-shape using `alphashape`.
          - Buffer the resulting polygon by self.cluster_boundary_margin.
        Returns a dict: {cluster_id: shapely_polygon, ...}.
        """
        from shapely.geometry import MultiPolygon

        boundaries = None
        print("Computing alpha-shape boundaries for main clusters...")

        cluster_points = df.loc[labels == 1, [x_col, y_col]].values
        if len(cluster_points) < 3:
            return None

        shape = alphashape(cluster_points, alpha=self.alpha)
        if shape is None:
            return None

        # If we get a MultiPolygon, unify it
        if isinstance(shape, MultiPolygon):
            shape = shape.buffer(0)

        # Expand by margin
        polygon_with_margin = shape.buffer(self.cluster_boundary_margin)
        boundaries = polygon_with_margin

        print("Alpha-shape boundary computation complete.")
        return boundaries

    def _compute_weights_internal(self, df, density_col=None, error_col=None):
        """
        Actually compute weights on the current df, using
        self.labels & self.main_clusters from the last cluster calc.
        """
        if self.main_labels is None or self.main_clusters is None:
            print("No clusters => setting all weights=1")
            w = np.ones(len(df), dtype=float)
            df["weights"] = w
            return

        if len(self.main_clusters) == 0:
            print("All points labeled noise => setting all weights=1")
            df["weights"] = 1.0
            return

        self.compute_weights(df, self.main_labels, density_col, error_col)

    def compute_weights(self, df, main_labels, density_col=None, error_col=None):
        """
        Combine density-based weighting and error-based weighting for points in main clusters.
        For noise/out-of-cluster points, set weight = max(in-cluster weight).

        Example:
          w_density = 1/(1 + density)
          w_error   = 1 + error
          w_final   = w_density * w_error
        """
        print("Computing weights...")

        n = len(df)
        w = np.ones(n, dtype=float)

        # Density factor
        if density_col and (density_col in df.columns):
            d = df[density_col].values
            w_density = 1.0 / (1.0 + d)
            w_density = np.clip(w_density, 0.0, 1.0)
        else:
            w_density = np.ones(n, dtype=float)

        # Error factor
        if error_col and (error_col in df.columns):
            e = df[error_col].values
            w_error = 1.0 + e
            w_error = np.clip(w_error, 0.0, None)
        else:
            w_error = np.ones(n, dtype=float)

        w_combined = w_density * w_error

        in_cluster_mask = np.array(
            [lbl != 0 for lbl in main_labels]
        )
        if not np.any(in_cluster_mask):
            print("No points in main clusters => all weight=1.")
            df["weights"] = 1.0
            return

        w = w_combined
        max_in_cluster_weight = np.max(w[in_cluster_mask])

        out_cluster_mask = ~in_cluster_mask
        w[out_cluster_mask] = np.minimum(w[out_cluster_mask], max_in_cluster_weight)

        df["weights"] = w
        print(f"Weight computation done. Max in-cluster weight = {max_in_cluster_weight:.4f}")
        return df

    def compute_density(self, df, x_col, y_col, bins=50):
        """
        Basic 2D histogram-based density for each row (only uses x_col,y_col).
        """
        xvals = df[x_col].values
        yvals = df[y_col].values
        hist, xedges, yedges = np.histogram2d(xvals, yvals, bins=bins)
        density = np.zeros_like(xvals, dtype=float)

        x_bin_idx = np.digitize(xvals, xedges) - 1
        y_bin_idx = np.digitize(yvals, yedges) - 1

        x_bin_idx = np.clip(x_bin_idx, 0, bins - 1)
        y_bin_idx = np.clip(y_bin_idx, 0, bins - 1)

        for i in range(len(xvals)):
            density[i] = hist[x_bin_idx[i], y_bin_idx[i]]

        return density
