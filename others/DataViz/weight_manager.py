import numpy as np
import pandas as pd
import shapely.geometry as geom
from shapely.geometry import MultiPoint
from shapely.ops import polygonize, unary_union
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from tqdm import tqdm
import threading
import time

class WeightManager:
    """
    Handles custom BFS-based clustering (via radius search),
    identifies main clusters, builds non-convex boundaries
    via alpha shapes, and computes row-wise weights.

    We support async (threaded) calls:
      - recalc_clusters_async() for clustering + boundary.
      - compute_weights_async() for weighting.
    """

    def __init__(self):
        # BFS radius
        self.eps = 0.5
        # Min points for a cluster (otherwise marked noise)
        self.min_samples = 10

        # Fraction of total points to cover in 'main' clusters
        self.main_cluster_coverage = 0.9

        # Weighting parameters
        self.alpha_density = 0.02
        self.alpha_error = 1.0
        self.out_of_cluster_factor = 0.5

        # Alpha-shape parameter
        self.alpha_shape_alpha = 1.0

        # Internal data from last cluster calc
        self.labels = None
        self.main_clusters = None
        self.boundaries = {}

        # Thread references (if needed to check or join)
        self._cluster_thread = None
        self._weight_thread = None

    # -----------------------------------------------------
    # ASYNC LAUNCHERS
    # -----------------------------------------------------
    def recalc_clusters_async(self, df, x_col, y_col, on_done=None):
        """
        Starts a thread to:
          1) BFS-cluster the points
          2) Identify main clusters
          3) Build alpha-shape boundaries
        on_done: optional callback run after finishing (in the worker thread!).
        """
        def worker():
            self._recalc_clusters_internal(df, x_col, y_col)
            if on_done is not None:
                on_done()

        self._cluster_thread = threading.Thread(target=worker)
        self._cluster_thread.start()

    def compute_weights_async(self, df, density_col=None, error_col=None, on_done=None):
        """
        Starts a thread to compute weights, given existing self.labels, self.main_clusters, etc.
        on_done: optional callback run after finishing (in the worker thread!).
        """
        def worker():
            self._compute_weights_internal(df, density_col, error_col)
            if on_done is not None:
                on_done()

        self._weight_thread = threading.Thread(target=worker)
        self._weight_thread.start()

    # -----------------------------------------------------
    # INTERNAL CLUSTER/BOUNDARY METHODS
    # -----------------------------------------------------
    def _recalc_clusters_internal(self, df, x_col, y_col):
        """
        Main logic for BFS clustering + boundary building. Runs in a thread.
        """
        # 1) BFS-based clustering
        self.labels = self._bfs_cluster(df, x_col, y_col)

        # 2) Identify main clusters
        self.main_clusters = self.find_main_clusters(df, self.labels)

        # 3) Build alpha-shape boundaries for main clusters
        self.boundaries = self.compute_cluster_boundaries(df, self.labels, self.main_clusters,
                                                          x_col, y_col)

    def _bfs_cluster(self, df, x_col, y_col):
        """
        BFS-based clustering with radius self.eps using a KDTree, with a working tqdm progress bar.
        """
        coords = df[[x_col, y_col]].values
        n = len(coords)
        labels = np.full(n, -1, dtype=int)  # noise by default
        visited = np.zeros(n, dtype=bool)

        tree = KDTree(coords)
        cluster_id = 0

        with tqdm(total=n, desc="Building clusters", mininterval=0.25) as pbar:
            # The approach: each time we find an unvisited point i, BFS from it.
            # We count pbar.update(1) whenever we mark a new point visited.
            for i in range(n):
                if visited[i]:
                    # Already visited => still count as done for the bar
                    pbar.update(1)
                    continue

                # Mark i visited
                visited[i] = True
                pbar.update(1)
                queue = [i]
                tmp_cluster_points = [i]

                # BFS
                while queue:
                    current = queue.pop()
                    neighbors = tree.query_radius([coords[current]], r=self.eps)[0]
                    for neigh in neighbors:
                        if not visited[neigh]:
                            visited[neigh] = True
                            pbar.update(1)
                            tmp_cluster_points.append(neigh)
                            queue.append(neigh)

                # Check if cluster is large enough
                if len(tmp_cluster_points) >= self.min_samples:
                    for idx in tmp_cluster_points:
                        labels[idx] = cluster_id
                    cluster_id += 1
                else:
                    # Mark them noise
                    for idx in tmp_cluster_points:
                        labels[idx] = -1

        return labels

    def find_main_clusters(self, df, labels):
        """
        Accumulate cluster sizes (excluding noise=-1) until coverage >= main_cluster_coverage.
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
        return main_clusters

    def compute_cluster_boundaries(self, df, labels, main_clusters, x_col, y_col):
        """
        Build alpha-shape boundaries for each main cluster, with a multi-cluster progress bar for the Delaunay approach.
        """
        cluster_points = {}
        total_simplices = 0
        coords = df[[x_col, y_col]].values

        for lbl in main_clusters:
            pts = coords[labels == lbl]
            cluster_points[lbl] = pts
            if len(pts) >= 4:
                tri = Delaunay(pts)
                total_simplices += len(tri.simplices)

        boundaries = {}
        with tqdm(total=total_simplices, desc="Building alpha-shape boundaries", mininterval=0.25) as pbar:
            for lbl, pts in cluster_points.items():
                if len(pts) < 4:
                    boundaries[lbl] = MultiPoint(pts).convex_hull
                    continue
                boundaries[lbl] = self._alpha_shape_with_progress(pts, self.alpha_shape_alpha, pbar)

        return boundaries

    def _alpha_shape_with_progress(self, points, alpha, pbar):
        """
        Same alpha-shape approach, but we increment 'pbar' for each triangle processed.
        """
        pts = np.array(points)
        if len(pts) < 4:
            return MultiPoint(pts).convex_hull

        tri = Delaunay(pts)
        edges = []
        threshold = 1.0 / alpha

        for ia, ib, ic in tri.simplices:
            # Each triangle => update progress
            pbar.update(1)

            pa, pb, pc = pts[ia], pts[ib], pts[ic]
            dab = np.linalg.norm(pa - pb)
            dbc = np.linalg.norm(pb - pc)
            dca = np.linalg.norm(pc - pa)
            for (p1, p2, dist) in [(pa, pb, dab), (pb, pc, dbc), (pc, pa, dca)]:
                if dist < threshold:
                    edges.append((tuple(p1), tuple(p2)))

        line_segments = [geom.LineString([seg[0], seg[1]]) for seg in edges]
        polygons = list(polygonize(line_segments))
        if not polygons:
            return MultiPoint(pts).convex_hull
        return unary_union(polygons)

    # -----------------------------------------------------
    # ASYNC WEIGHT CALC
    # -----------------------------------------------------
    def _compute_weights_internal(self, df, density_col=None, error_col=None):
        """
        Actually compute weights on the current df, using
        self.labels, self.main_clusters from the last cluster calc.
        """
        if self.labels is None or self.main_clusters is None:
            # No clusters => can't do out-of-cluster factor
            w = np.ones(len(df), dtype=float)
            df["weights"] = w
            return

        self.compute_weights(df, self.labels, self.main_clusters, density_col, error_col)

    # -----------------------------------------------------
    # DIRECT (NON-ASYNC) WEIGHTING METHOD
    # -----------------------------------------------------
    def compute_weights(self, df, labels, main_clusters, density_col=None, error_col=None):
        """
        Combine density-based weighting, error-based weighting,
        and cluster membership to produce final row-wise weights.
        """
        w = np.ones(len(df), dtype=float)

        if density_col and (density_col in df.columns):
            d = df[density_col].values
            # Example weighting: high density => smaller weight
            w_density = np.exp(-self.alpha_density * d)
            w_density = np.clip(w_density, 0.05, 0.95)
            w *= w_density

        if error_col and (error_col in df.columns):
            e = df[error_col].values
            w_err = 1.0 / (1.0 + self.alpha_error * e)
            w *= w_err

        for i in range(len(df)):
            lbl = labels[i]
            if lbl == -1 or lbl not in main_clusters:
                w[i] *= self.out_of_cluster_factor

        df["weights"] = w
        return df

    # -----------------------------------------------------
    # DENSITY METHOD
    # -----------------------------------------------------
    def compute_density(self, df, x_col, y_col, bins=50):
        """
        Histogram-based 2D density for each row.
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
