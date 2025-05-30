# gui_elements/frames_weight.py

import os
import tkinter as tk
from tkinter import filedialog, messagebox

class WeightingFrame(tk.LabelFrame):
    """Frame for BFS-based clustering and weighting controls."""
    def __init__(self, parent, main_app, **kwargs):
        super().__init__(parent, text="Weighting", **kwargs)
        self.main_app = main_app

        # Coverage slider
        tk.Label(self, text="Coverage (%)").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        coverage_scale = tk.Scale(
            self,
            from_=50, to=100,
            orient=tk.HORIZONTAL,
            variable=main_app.coverage_var,
            command=lambda val: self.main_app._on_coverage_changed(val)
        )
        coverage_scale.grid(row=0, column=1, padx=2, pady=2)

        # Eps slider
        tk.Label(self, text="Eps:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        eps_scale = tk.Scale(
            self,
            from_=0.01, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=main_app.eps_var
        )
        eps_scale.grid(row=1, column=1, padx=2, pady=2)

        # Alpha shape slider
        tk.Label(self, text="Alpha Shape:").grid(row=2, column=0, padx=2, pady=2, sticky="w")
        alpha_scale = tk.Scale(
            self,
            from_=0.1, to=5.0, resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=main_app.alpha_var,
            command=lambda val: self.main_app._on_alpha_changed(val)
        )
        alpha_scale.grid(row=2, column=1, padx=2, pady=2)

        tk.Label(self, text="Error-Density Ratio:").grid(row=3, column=0, padx=2, pady=2, sticky="w")
        ratio_scale = tk.Scale(
            self,
            from_=0.0, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=main_app.error_density_ratio_var
        )
        ratio_scale.grid(row=3, column=1, padx=2, pady=2)

        # Show Main Clusters
        self.show_clusters_chk = tk.Checkbutton(
            self,
            text="Show Main Clusters",
            variable=main_app.show_main_clusters_var,
            command=main_app.update_plot
        )
        self.show_clusters_chk.grid(row=4, column=0, padx=2, pady=2, sticky="w")

        # Row for Color by Weight & Color by Cluster
        self.color_by_weight_chk = tk.Checkbutton(
            self,
            text="Color by Weight",
            variable=main_app.color_by_weight_var,
            command=main_app.update_plot
        )
        self.color_by_weight_chk.grid(row=5, column=0, padx=2, pady=2, sticky="w")

        self.color_by_cluster_chk = tk.Checkbutton(
            self,
            text="Color by Cluster",
            variable=main_app.color_by_cluster_var,
            command=main_app.update_plot
        )
        self.color_by_cluster_chk.grid(row=5, column=1, padx=2, pady=2, sticky="w")

        # Row for Recalc Clusters & Recalc Boundaries
        recalc_clusters_btn = tk.Button(
            self,
            text="Recalc Clusters",
            command=self._on_recalc_clusters
        )
        recalc_clusters_btn.grid(row=6, column=0, padx=2, pady=2)

        recalc_boundaries_btn = tk.Button(
            self,
            text="Recalc Boundaries",
            command=self._on_recalc_boundaries
        )
        recalc_boundaries_btn.grid(row=6, column=1, padx=2, pady=2)

        # Row for Compute Weights & Export Weighted CSV
        compute_wts_btn = tk.Button(
            self,
            text="Get Weights",
            command=self._on_compute_weights
        )
        compute_wts_btn.grid(row=7, column=0, padx=2, pady=2)

        export_wts_btn = tk.Button(
            self,
            text="Export Weighted CSV",
            command=main_app._on_export_weights
        )
        export_wts_btn.grid(row=7, column=1, padx=2, pady=2)

    def _on_recalc_clusters(self):
        main_app = self.main_app
        x_col = main_app.x_var.get()
        y_col = main_app.y_var.get()

        main_app.weight_manager.main_cluster_coverage = float(main_app.coverage_var.get()) / 100.0
        main_app.weight_manager.alpha = float(main_app.alpha_var.get())
        main_app.weight_manager.eps = float(main_app.eps_var.get())

        def on_done():
            main_app.after(0, self._clusters_done)

        feature_cols = getattr(main_app.config, "cluster_features", [x_col, y_col])
        main_app.weight_manager.recalc_clusters_async(
            main_app.df, feature_cols, x_col, y_col, on_done=on_done
        )

    def _clusters_done(self):
        main_app = self.main_app
        main_app.last_labels = main_app.weight_manager.labels
        main_app.last_main_labels = main_app.weight_manager.main_labels
        main_app.main_clusters = main_app.weight_manager.main_clusters
        main_app.boundaries = main_app.weight_manager.boundaries

        main_app.df["cluster_label"] = main_app.last_labels
        main_app.df["cluster_main_label"] = main_app.last_main_labels

        main_app.update_plot()

    def _on_recalc_boundaries(self):
        main_app = self.main_app
        x_col = main_app.x_var.get()
        y_col = main_app.y_var.get()

        main_app.weight_manager.alpha = float(main_app.alpha_var.get())
        main_app.weight_manager.main_cluster_coverage = float(main_app.coverage_var.get()) / 100.0

        def on_done():
            main_app.after(0, self._boundaries_done)

        main_app.weight_manager.recalc_boundaries_async(
            main_app.df, x_col, y_col, on_done=on_done
        )

    def _boundaries_done(self):
        main_app = self.main_app
        main_app.boundaries = main_app.weight_manager.boundaries
        main_app.update_plot()

    def _on_compute_weights(self):
        main_app = self.main_app

        x_col = main_app.x_var.get()
        y_col = main_app.y_var.get()
        bins = main_app._parse_bins(main_app.density_bins_var.get())
        main_app.df["__density_for_weight"] = main_app.weight_manager.compute_density(
            main_app.df, x_col, y_col, bins=bins
        )
        density_col = "__density_for_weight"

        s_col = main_app.student_col_var.get()
        error_col = f"{s_col}_abs_err"

        def on_done():
            main_app.after(0, self._weights_done)

        ratio_value = main_app.error_density_ratio_var.get()
        main_app.weight_manager.compute_weights_async(
            main_app.df,
            density_col=density_col,
            error_col=error_col,
            ratio=ratio_value,
            on_done=on_done
        )

    def _weights_done(self):
        self.main_app.update_plot()


def on_coverage_changed(main_app, val):
    """Called when coverage slider changes."""
    main_app.weight_manager.main_cluster_coverage = float(val) / 100.0

def on_alpha_changed(main_app, val):
    """Called when alpha slider changes."""
    main_app.weight_manager.alpha = float(val)


def on_export_weights(main_app):
    """Called when user exports the weighted CSV."""
    os.makedirs(main_app.config.data_folder_with_weights, exist_ok=True)
    for source_file, group in main_app.original_df.groupby("__source_file"):
        group = group.copy()

        appended_group = main_app.df[main_app.df["__source_file"] == source_file]

        if "weight" in appended_group.columns:
            group["weight"] = appended_group["weight"].values

        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read original file {source_file}: {e}")
            continue

        comments = [line for line in original_lines if line.strip().startswith("#")]

        csv_data = group.to_csv(index=False)

        new_file_content = "".join(comments) + csv_data

        out_path = os.path.join(main_app.config.data_folder_with_weights,
                                f"{os.path.splitext(source_file)[0]}_with_weight.csv")

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(new_file_content)

    messagebox.showinfo("Saved", f"Weights saved to {main_app.config.data_folder_with_weights}")


