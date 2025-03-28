import numpy as np
import pandas as pd
import threading

from sklearn.cluster import DBSCAN
from shapely.geometry import Point
from alphashape import alphashape
from tqdm import tqdm

class WeightManager:
    """
    Asynchronous manager:
      - recalc_clusters_async() -> cluster in all specified dimensions,
        then build alpha-shape boundaries in (x_col, y_col).
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
        self.main_clusters = None
        self.boundaries = None

    def recalc_clusters_async(self, df, feature_cols, x_col, y_col, on_done=None):
        """
        Starts thread to do:
          1) DBSCAN clustering in all `feature_cols` (8D).
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
        Main logic for clustering + boundary building. Runs in a thread.
        """
        # 1) Perform DBSCAN on the specified feature columns (8D or more).
        self.labels = self._cluster(df, feature_cols)

        # 2) Identify main clusters
        self.main_clusters = self.find_main_clusters(df, self.labels)

        # 3) Compute alpha-shape boundaries for each main cluster
        self.boundaries = self.compute_cluster_boundaries(df,
                                                          self.labels,
                                                          self.main_clusters,
                                                          x_col, y_col)

    def _cluster(self, df, feature_cols):
        """
        Use scikit-learn DBSCAN with parallelization to handle large data in multiple dimensions.
        We'll do minimal progress printing. DBSCAN does not provide partial progress.
        """
        print("Starting DBSCAN clustering with scikit-learn...")

        data = df[feature_cols].values  # shape: (n_samples, n_features)
        print(f"Data shape for clustering: {data.shape}")
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples,
                    metric='euclidean', n_jobs=-1)  # n_jobs=-1 => use all cores

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
        # Unique cluster labels except noise
        unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]
        cluster_sizes = [(lbl, np.sum(labels == lbl)) for lbl in unique_labels]
        # Sort descending by size
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
        print(f"Main clusters cover ~{coverage:.1f}% of points (goal was {100*self.main_cluster_coverage}%).")
        print(f"Main clusters identified: {main_clusters}")
        return main_clusters

    def compute_cluster_boundaries(self, df, labels, main_clusters, x_col, y_col):
        """
        Build alpha-shape polygons (2D) for each main cluster in columns (x_col, y_col).
        We'll do a simple approach:
          - Extract points in (x_col,y_col) for each cluster.
          - Compute the alpha-shape using `alphashape`.
          - Buffer the resulting polygon by self.cluster_boundary_margin.
        Returns a dict: {cluster_id: shapely_polygon, ...}.
        """
        from shapely.geometry import MultiPolygon

        # If no main clusters, return empty
        if not main_clusters:
            return {}

        boundaries = {}
        main_clusters_list = sorted(list(main_clusters))
        print("Computing alpha-shape boundaries for main clusters...")

        for i, c_lbl in enumerate(main_clusters_list):
            # Show progress with tqdm or a simple print
            # Using tqdm for a small example here:
            # We'll do an iteration but typically you'd wrap the entire loop with tqdm.
            print(f"Boundary for cluster {c_lbl} ({i+1}/{len(main_clusters_list)})...")

            cluster_points = df.loc[labels == c_lbl, [x_col, y_col]].values
            if len(cluster_points) < 3:
                # Not enough points to form a polygon
                boundaries[c_lbl] = None
                continue

            # Potentially you might downsample large clusters if alpha-shape is too slow
            # e.g. if len(cluster_points) > 50000: sample them
            # cluster_points = cluster_points[np.random.choice(len(cluster_points), 50000, replace=False)]

            # Create alphashape
            shape = alphashape(cluster_points, alpha=self.alpha)
            if shape is None:
                # Might occur if alpha is too small or too large
                boundaries[c_lbl] = None
                continue

            # If we get a MultiPolygon, unify it
            if isinstance(shape, MultiPolygon):
                shape = shape.buffer(0)  # unify the polygons if they touch

            # Buffer to expand by margin
            polygon_with_margin = shape.buffer(self.cluster_boundary_margin)

            boundaries[c_lbl] = polygon_with_margin

        print("Alpha-shape boundary computation complete.")
        return boundaries

    def _compute_weights_internal(self, df, density_col=None, error_col=None):
        """
        Actually compute weights on the current df, using
        self.labels & self.main_clusters from the last cluster calc.
        """
        if self.labels is None or self.main_clusters is None:
            print("No clusters => setting all weights=1")
            w = np.ones(len(df), dtype=float)
            df["weights"] = w
            return

        # Check if we actually have any main cluster
        # If main_clusters is empty, then effectively all points are noise
        # => set all weight=1
        if len(self.main_clusters) == 0:
            print("All points labeled noise => setting all weights=1")
            df["weights"] = 1.0
            return

        self.compute_weights(df, self.labels, self.main_clusters, density_col, error_col)

    def compute_weights(self, df, labels, main_clusters, density_col=None, error_col=None):
        """
        Combine density-based weighting and error-based weighting for points in main clusters.
        For noise/out-of-cluster points, set weight = max(in-cluster weight).

        Formula example (you can tweak):
          w_density = 1/(1 + density)
          w_error   = 1 + error
          w_final   = w_density * w_error
        """
        print("Computing weights...")

        n = len(df)
        w = np.ones(n, dtype=float)

        # Prepare density factor
        if density_col and (density_col in df.columns):
            d = df[density_col].values
            # Inverse of (1 + d)
            w_density = 1.0 / (1.0 + d)
            # Just in case, clip to something reasonable
            w_density = np.clip(w_density, 0.0, 1.0)
        else:
            w_density = np.ones(n, dtype=float)

        # Prepare error factor
        if error_col and (error_col in df.columns):
            e = df[error_col].values
            # e.g. (1 + e)
            w_error = 1.0 + e
            # If you expect negative or extremely large errors, you may want more robust logic
            w_error = np.clip(w_error, 0.0, None)  # no upper clip here
        else:
            w_error = np.ones(n, dtype=float)

        # Combine
        w_combined = w_density * w_error

        # Now assign w_combined for in-cluster, and figure out the max in-cluster weight
        in_cluster_mask = np.array([ (lbl != -1) and (lbl in main_clusters) for lbl in labels ])
        if not np.any(in_cluster_mask):
            # Means no points actually in main clusters => everything is noise => set weight=1
            print("No points in main clusters => all weight=1.")
            df["weights"] = 1.0
            return

        # Set those in main clusters to their combined weight
        w[in_cluster_mask] = w_combined[in_cluster_mask]

        # Find max weight among in-cluster points
        max_in_cluster_weight = np.max(w[in_cluster_mask])

        # For out-of-cluster/noise points, set them to that max_in_cluster_weight,
        # but not exceeding it.
        out_cluster_mask = ~in_cluster_mask
        w[out_cluster_mask] = max_in_cluster_weight

        df["weights"] = w
        print(f"Weight computation done. Max in-cluster weight = {max_in_cluster_weight:.4f}")
        return df

    def compute_density(self, df, x_col, y_col, bins=50):
        """
        Histogram-based 2D density for each row.
        You might want a more advanced k-NN or KDE approach for 8D,
        but this is an example in just two columns (x_col,y_col).
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
