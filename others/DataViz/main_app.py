# main_app.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression

from data_processor import DataProcessor
from sampler import Sampler

# NEW: import your WeightManager class
from weight_manager import WeightManager

matplotlib.use("TkAgg")

class MainApplication(tk.Tk):
    """Main Tkinter GUI."""
    def __init__(self, config, df):
        super().__init__()
        self.config = config
        self.original_df = df
        self.df = df.copy()
        self.processor = DataProcessor(config)

        # WeightManager and cluster-related data
        self.weight_manager = WeightManager()  # BFS clustering + alpha shapes + weighting
        self.last_labels = None
        self.main_clusters = None
        self.boundaries = {}

        self.title("Robot State Data Analysis")
        self.geometry("1400x1200")
        self.lift()
        self.attributes("-topmost", True)
        self.after_idle(lambda: self.attributes("-topmost", False))
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.already_warned_about_local_norm = False

        self.create_widgets()

        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.populate_control_inputs_dropdowns()

        self.x_dropdown["values"] = self.config.state_columns
        self.y_dropdown["values"] = self.config.state_columns
        if len(self.config.state_columns) >= 2:
            self.x_dropdown.current(0)
            self.y_dropdown.current(1)

        self.feature_filter_dropdown["values"] = sorted(df.columns)
        if len(df.columns) > 0:
            self.feature_filter_dropdown.current(0)

        all_ctrl_cols = []
        if self.config.teacher_control_columns:
            all_ctrl_cols.extend(self.config.teacher_control_columns)
        if self.config.student_control_columns:
            all_ctrl_cols.extend(self.config.student_control_columns)
        all_ctrl_cols = list(dict.fromkeys(all_ctrl_cols))
        all_ctrl_cols += ["Absolute Error", "Relative Error", "Density"]
        self.control_filter_dropdown["values"] = all_ctrl_cols
        if len(all_ctrl_cols) > 0:
            self.control_filter_dropdown.current(0)

        self.x_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.y_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.teacher_col_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.student_col_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.plot_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.plot_type_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.feature_filter_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.control_filter_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        self.filter_min_var.trace_add("write", lambda *args: self.update_plot())
        self.filter_max_var.trace_add("write", lambda *args: self.update_plot())
        self.feature_filter_min_var.trace_add("write", lambda *args: self.update_plot())
        self.feature_filter_max_var.trace_add("write", lambda *args: self.update_plot())
        self.density_bins_var.trace_add("write", lambda *args: self.update_plot())
        self.log_scale_var.trace_add("write", lambda *args: self.update_plot())
        self.clip_control_var.trace_add("write", lambda *args: self.update_plot())
        self.use_pls_var.trace_add("write", lambda *args: self.update_plot())
        self.use_ols_var.trace_add("write", lambda *args: self.update_plot())
        self.normalize_var.trace_add("write", lambda *args: self.update_plot())
        self.n_steps_var.trace_add("write", lambda *args: self.apply_n_step_removal())

        self.update_plot()

    def create_widgets(self):
        # -- CHANGED LINES START: Use a scrollable container on the LEFT side --

        # Create a parent container for a scrollbar
        self.slider_frame_container = tk.Frame(self)
        self.slider_frame_container.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=5)

        # Create a parent container for the controls
        self.control_frame_container = tk.Frame(self)
        self.control_frame_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Create a canvas inside the container
        self.control_canvas = tk.Canvas(self.control_frame_container)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a vertical scrollbar linked to the canvas
        self.scrollbar = ttk.Scrollbar(self.slider_frame_container, orient="vertical",
                                       command=self.control_canvas.yview)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create an actual frame to hold all the widgets, and place it in the canvas
        self.control_frame = tk.Frame(self.control_canvas)
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor='nw')

        # Ensure the canvas scroll region is updated whenever its size changes
        self.control_canvas.bind("<Configure>",
            lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        )

        # -- CHANGED LINES END --

        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        axes_frame = tk.LabelFrame(self.control_frame, text="Axes Selection")
        axes_frame.pack(fill=tk.X, pady=5)

        tk.Label(axes_frame, text="Select X-axis:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.x_var = tk.StringVar()
        self.x_dropdown = ttk.Combobox(axes_frame, textvariable=self.x_var, state="readonly", width=16)
        self.x_dropdown.grid(row=0, column=1, padx=2, pady=2)

        tk.Label(axes_frame, text="Select Y-axis:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.y_var = tk.StringVar()
        self.y_dropdown = ttk.Combobox(axes_frame, textvariable=self.y_var, state="readonly", width=16)
        self.y_dropdown.grid(row=1, column=1, padx=2, pady=2)

        self.use_pls_var = tk.BooleanVar(value=False)
        self.use_pls_check = tk.Checkbutton(axes_frame, text="Use PLS for X-Y (2D)", variable=self.use_pls_var)
        self.use_pls_check.grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky="w")

        self.use_ols_var = tk.BooleanVar(value=False)
        self.use_ols_check = tk.Checkbutton(axes_frame, text="Use OLS (Teacher vs. Student)", variable=self.use_ols_var)
        self.use_ols_check.grid(row=3, column=0, columnspan=2, padx=2, pady=2, sticky="w")

        self.normalize_var = tk.BooleanVar(value=False)
        self.normalize_check = tk.Checkbutton(axes_frame, text="Normalize [-1..1]", variable=self.normalize_var)
        self.normalize_check.grid(row=4, column=0, columnspan=2, padx=2, pady=2, sticky="w")

        ctrl_frame = tk.LabelFrame(self.control_frame, text="Teacher/Student Control")
        ctrl_frame.pack(fill=tk.X, pady=5)

        tk.Label(ctrl_frame, text="Teacher Control:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.teacher_col_var = tk.StringVar()
        self.teacher_col_dropdown = ttk.Combobox(ctrl_frame, textvariable=self.teacher_col_var,
                                                 state="readonly", width=16)
        self.teacher_col_dropdown.grid(row=0, column=1, padx=2, pady=2)

        tk.Label(ctrl_frame, text="Student Control:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.student_col_var = tk.StringVar()
        self.student_col_dropdown = ttk.Combobox(ctrl_frame, textvariable=self.student_col_var,
                                                 state="readonly", width=16)
        self.student_col_dropdown.grid(row=1, column=1, padx=2, pady=2)

        plotopts_frame = tk.LabelFrame(self.control_frame, text="Plot Options")
        plotopts_frame.pack(fill=tk.X, pady=5)

        tk.Label(plotopts_frame, text="Plot Option:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.plot_var = tk.StringVar()
        self.plot_options = ["Teacher Control", "Student Control", "Absolute Error", "Relative Error", "Density"]
        self.plot_dropdown = ttk.Combobox(plotopts_frame, textvariable=self.plot_var,
                                          values=self.plot_options, state="readonly", width=16)
        self.plot_dropdown.current(0)
        self.plot_dropdown.grid(row=0, column=1, padx=2, pady=2)

        tk.Label(plotopts_frame, text="Plot Type:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.plot_type_var = tk.StringVar(value="Scatter")
        self.plot_type_dropdown = ttk.Combobox(plotopts_frame, textvariable=self.plot_type_var,
                                               values=["Scatter", "Heatmap"], state="readonly", width=16)
        self.plot_type_dropdown.current(0)
        self.plot_type_dropdown.grid(row=1, column=1, padx=2, pady=2)

        self.clip_control_var = tk.BooleanVar(value=True)
        self.clip_control_check = tk.Checkbutton(plotopts_frame, text="Clip Student [-1,1]", variable=self.clip_control_var)
        self.clip_control_check.grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky="w")

        filter_frame = tk.LabelFrame(self.control_frame, text="Control Input Filter")
        filter_frame.pack(pady=5, fill=tk.X)

        tk.Label(filter_frame, text="Filter by:").pack(pady=2)
        self.control_filter_var = tk.StringVar()
        self.control_filter_dropdown = ttk.Combobox(filter_frame, textvariable=self.control_filter_var,
                                                    state="readonly", width=20)
        self.control_filter_dropdown.pack(pady=2)

        row_cf = tk.Frame(filter_frame)
        row_cf.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(row_cf, text="Min:").pack(side=tk.LEFT, padx=2)
        self.filter_min_var = tk.StringVar(value="")
        self.filter_min_entry = tk.Entry(row_cf, textvariable=self.filter_min_var, width=8)
        self.filter_min_entry.pack(side=tk.LEFT, padx=2)

        tk.Label(row_cf, text="Max:").pack(side=tk.LEFT, padx=2)
        self.filter_max_var = tk.StringVar(value="")
        self.filter_max_entry = tk.Entry(row_cf, textvariable=self.filter_max_var, width=8)
        self.filter_max_entry.pack(side=tk.LEFT, padx=2)

        feature_filter_frame = tk.LabelFrame(self.control_frame, text="Additional Feature Filter")
        feature_filter_frame.pack(pady=5, fill=tk.X)

        tk.Label(feature_filter_frame, text="Feature:").pack(side=tk.LEFT, padx=2)
        self.feature_filter_var = tk.StringVar(value="")
        self.feature_filter_dropdown = ttk.Combobox(feature_filter_frame, textvariable=self.feature_filter_var,
                                                    state="readonly", width=12)
        self.feature_filter_dropdown.pack(side=tk.LEFT, padx=2)

        tk.Label(feature_filter_frame, text="Min:").pack(side=tk.LEFT, padx=2)
        self.feature_filter_min_var = tk.StringVar(value="")
        self.feature_filter_min_entry = tk.Entry(feature_filter_frame,
                                                 textvariable=self.feature_filter_min_var, width=8)
        self.feature_filter_min_entry.pack(side=tk.LEFT, padx=2)

        tk.Label(feature_filter_frame, text="Max:").pack(side=tk.LEFT, padx=2)
        self.feature_filter_max_var = tk.StringVar(value="")
        self.feature_filter_max_entry = tk.Entry(feature_filter_frame,
                                                 textvariable=self.feature_filter_max_var, width=8)
        self.feature_filter_max_entry.pack(side=tk.LEFT, padx=2)

        density_frame = tk.LabelFrame(self.control_frame, text="Density Settings")
        density_frame.pack(pady=5, fill=tk.X)

        row_density = tk.Frame(density_frame)
        row_density.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(row_density, text="Bins:").pack(side=tk.LEFT, padx=2)
        self.density_bins_var = tk.StringVar(value="50")
        self.density_bins_entry = tk.Entry(row_density, textvariable=self.density_bins_var, width=6)
        self.density_bins_entry.pack(side=tk.LEFT, padx=2)

        self.log_scale_var = tk.BooleanVar(value=False)
        self.log_scale_check = tk.Checkbutton(density_frame, text="Log Scale", variable=self.log_scale_var)
        self.log_scale_check.pack(pady=2)

        sampling_frame = tk.LabelFrame(self.control_frame, text="Sampling")
        sampling_frame.pack(pady=5, fill=tk.X)

        btn_high_err = tk.Button(sampling_frame, text="High Error", command=self.sample_high_error)
        btn_high_err.pack(side=tk.LEFT, padx=2, pady=2)

        btn_low_density = tk.Button(sampling_frame, text="Low Density", command=self.sample_poorly_represented)
        btn_low_density.pack(side=tk.LEFT, padx=2, pady=2)

        btn_uniform = tk.Button(sampling_frame, text="Uniform", command=self.sample_uniform)
        btn_uniform.pack(side=tk.LEFT, padx=2, pady=2)

        steps_frame = tk.Frame(self.control_frame)
        steps_frame.pack(fill=tk.X, pady=5)
        tk.Label(steps_frame, text="Remove N steps:").pack(side=tk.LEFT, padx=2)
        self.n_steps_var = tk.IntVar(value=0)
        self.n_steps_entry = tk.Entry(steps_frame, textvariable=self.n_steps_var, width=8)
        self.n_steps_entry.pack(side=tk.LEFT, padx=2)

        error_frame = tk.LabelFrame(self.control_frame, text="Error Settings")
        error_frame.pack(pady=5, fill=tk.X)
        tk.Label(error_frame, text="Relative Error Cap:").pack(side=tk.LEFT, padx=2)
        self.error_cap_var = tk.StringVar(value=str(self.config.relative_error_cap))
        self.error_cap_entry = tk.Entry(error_frame, textvariable=self.error_cap_var, width=8)
        self.error_cap_entry.pack(side=tk.LEFT, padx=2)
        apply_cap_button = tk.Button(error_frame, text="Apply", command=self.apply_error_cap)
        apply_cap_button.pack(side=tk.LEFT, padx=2)

        # ---------- Weighting/Clusters ----------
        weighting_frame = tk.LabelFrame(self.control_frame, text="Weighting")
        weighting_frame.pack(fill=tk.X, pady=5)

        tk.Label(weighting_frame, text="Coverage (%)").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.coverage_var = tk.IntVar(value=90)
        coverage_scale = tk.Scale(weighting_frame, from_=50, to=100, orient=tk.HORIZONTAL,
                                  variable=self.coverage_var,
                                  command=lambda val: self._on_coverage_changed(val))
        coverage_scale.grid(row=0, column=1, padx=2, pady=2)

        tk.Label(weighting_frame, text="Alpha Shape:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.alpha_var = tk.DoubleVar(value=1.0)
        alpha_scale = tk.Scale(weighting_frame, from_=0.1, to=5.0, resolution=0.1,
                               orient=tk.HORIZONTAL, variable=self.alpha_var,
                               command=lambda val: self._on_alpha_changed(val))
        alpha_scale.grid(row=1, column=1, padx=2, pady=2)

        tk.Label(weighting_frame, text="Scheme:").grid(row=2, column=0, padx=2, pady=2, sticky="w")
        self.weighting_scheme_var = tk.StringVar(value="Mixed")
        scheme_combo = ttk.Combobox(weighting_frame, textvariable=self.weighting_scheme_var,
                                    values=["Error-based", "Density-based", "Mixed"], state="readonly")
        scheme_combo.grid(row=2, column=1, padx=2, pady=2)

        self.show_main_clusters_var = tk.BooleanVar(value=False)
        show_clusters_chk = tk.Checkbutton(weighting_frame, text="Show Main Clusters",
                                           variable=self.show_main_clusters_var,
                                           command=self.update_plot)
        show_clusters_chk.grid(row=3, column=0, padx=2, pady=2, sticky="w")

        self.color_by_weight_var = tk.BooleanVar(value=False)
        color_by_weight_chk = tk.Checkbutton(weighting_frame, text="Color by Weight",
                                             variable=self.color_by_weight_var,
                                             command=self.update_plot)
        color_by_weight_chk.grid(row=3, column=1, padx=2, pady=2, sticky="w")

        # Separate buttons for clusters vs. weights
        recalc_clusters_btn = tk.Button(weighting_frame, text="Recalc Clusters",
                                        command=self._on_recalc_clusters)
        recalc_clusters_btn.grid(row=4, column=0, padx=2, pady=2)

        compute_wts_btn = tk.Button(weighting_frame, text="Compute Weights", command=self._on_compute_weights)
        compute_wts_btn.grid(row=4, column=1, padx=2, pady=2)

        export_wts_btn = tk.Button(weighting_frame, text="Export Weighted CSV", command=self._on_export_weights)
        export_wts_btn.grid(row=5, column=0, columnspan=2, padx=2, pady=2)

    # -------------------------------------
    # Cluster Recalculation (Async)
    # -------------------------------------
    def _on_recalc_clusters(self):
        """Launch BFS-based clustering + alpha-shape boundary building in a thread."""
        x_col = self.x_var.get()
        y_col = self.y_var.get()

        # Update coverage, alpha
        self.weight_manager.main_cluster_coverage = float(self.coverage_var.get()) / 100.0
        self.weight_manager.alpha_shape_alpha = float(self.alpha_var.get())

        # Callback to run after finishing BFS+boundary in worker thread
        def on_done():
            # This is called in the worker thread; schedule a GUI-safe callback
            self.after(0, self._clusters_done)

        # Start BFS + alpha-shape in background
        self.weight_manager.recalc_clusters_async(self.df, x_col, y_col, on_done=on_done)

    def _clusters_done(self):
        """Called after BFS + boundary thread is done."""
        self.last_labels = self.weight_manager.labels
        self.main_clusters = self.weight_manager.main_clusters
        self.boundaries = self.weight_manager.boundaries
        self.update_plot()

    # -------------------------------------
    # Weight Recalculation (Async)
    # -------------------------------------
    def _on_compute_weights(self):
        """Compute row-wise weights only (reuse existing clusters)."""
        scheme = self.weighting_scheme_var.get()
        error_col = None
        density_col = None

        if scheme in ["Error-based", "Mixed"]:
            s_col = self.student_col_var.get()
            error_col = f"{s_col}_abs_err"  # or rel_err

        if scheme in ["Density-based", "Mixed"]:
            x_col = self.x_var.get()
            y_col = self.y_var.get()
            bins = self._parse_bins(self.density_bins_var.get())
            self.df["__density_for_weight"] = self.weight_manager.compute_density(self.df, x_col, y_col, bins=bins)
            density_col = "__density_for_weight"

        # async callback
        def on_done():
            self.after(0, self._weights_done)

        # Just compute weights (no BFS, no boundaries)
        self.weight_manager.compute_weights_async(self.df, density_col, error_col, on_done=on_done)

    def _weights_done(self):
        """Called after weighting thread is done."""
        self.update_plot()



    # -------------------------------------
    # Other existing methods
    # -------------------------------------
    def populate_control_inputs_dropdowns(self):
        t_cols = self.config.teacher_control_columns or []
        if t_cols:
            self.teacher_col_dropdown["values"] = t_cols
            self.teacher_col_dropdown.current(0)
        else:
            self.teacher_col_dropdown["values"] = []
            self.teacher_col_dropdown.config(state="disabled")

        s_cols = self.config.student_control_columns or []
        if s_cols:
            self.student_col_dropdown["values"] = s_cols
            self.student_col_dropdown.current(0)
        else:
            self.student_col_dropdown["values"] = []
            self.student_col_dropdown.config(state="disabled")

    def apply_n_step_removal(self):
        n = self.n_steps_var.get()
        self.df = self.remove_n_starting_steps(self.original_df, n)
        self.update_plot()

    def remove_n_starting_steps(self, df, n):
        if n <= 0:
            return df
        return df.groupby("experiment_id", group_keys=False).apply(lambda g: g.iloc[n:])

    def apply_error_cap(self):
        try:
            new_cap = float(self.error_cap_var.get())
            self.config.relative_error_cap = new_cap
            self.original_df = self.processor.compute_errors(self.original_df)
            self.df = self.remove_n_starting_steps(self.original_df, self.n_steps_var.get())
            self.update_plot()
        except ValueError:
            messagebox.showwarning("Invalid Value", "Please enter a valid number for relative error cap.")

    def _on_coverage_changed(self, val):
        self.weight_manager.main_cluster_coverage = float(val) / 100.0

    def _on_alpha_changed(self, val):
        self.weight_manager.alpha_shape_alpha = float(val)

    def _on_export_weights(self):
        fname = filedialog.asksaveasfilename(defaultextension=".csv",
                                             initialfile="weighted_data.csv",
                                             title="Save Weighted CSV")
        if fname:
            self.df.to_csv(fname, index=False)
            messagebox.showinfo("Saved", f"Weights saved to {fname}")

    def update_plot(self):
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)
        df_plot = self.original_df.copy()

        n = self.n_steps_var.get()
        df_plot = df_plot.groupby("experiment_id", group_keys=False).apply(lambda g: g.iloc[n:])

        if self.normalize_var.get():
            df_plot = self._apply_normalization(df_plot)
            df_plot = self.processor.compute_errors(df_plot)

        plot_option = self.plot_var.get()

        # OLS vs. PLS vs. normal
        if self.use_ols_var.get():
            x_col, y_col = self._force_teacher_student_xy(df_plot)
            plot_type = "Scatter"
            do_ols = True
        elif self.use_pls_var.get():
            pls_target = self._get_pls_target_column(plot_option, df_plot)
            if pls_target is None:
                self.ax.set_title("Could not compute PLS target.")
                self.canvas.draw_idle()
                return
            df_plot["__target_for_pls"] = pls_target
            self._compute_pls_components(df_plot)
            x_col, y_col = "PLS_1", "PLS_2"
            plot_type = self.plot_type_var.get()
            do_ols = False
        else:
            x_col = self.x_var.get()
            y_col = self.y_var.get()
            plot_type = self.plot_type_var.get()
            do_ols = False

        if x_col not in df_plot.columns or y_col not in df_plot.columns:
            self.ax.set_title(f"X or Y not found: {x_col}, {y_col}")
            self.canvas.draw_idle()
            return

        df_filtered = df_plot.copy()
        control_filter_choice = self.control_filter_var.get()
        col_to_filter = self._map_control_filter_to_column(control_filter_choice, df_filtered, x_col, y_col)

        cf_min, cf_max = self._parse_range(self.filter_min_var.get(), self.filter_max_var.get())
        if col_to_filter and (cf_min is not None) and (cf_max is not None) and (cf_min < cf_max):
            df_filtered = df_filtered[
                (df_filtered[col_to_filter] >= cf_min) & (df_filtered[col_to_filter] <= cf_max)
            ]

        feat_col = self.feature_filter_var.get()
        if feat_col in df_filtered.columns:
            f_min, f_max = self._parse_range(self.feature_filter_min_var.get(),
                                             self.feature_filter_max_var.get())
            if f_min is not None and f_max is not None and f_min < f_max:
                df_filtered = df_filtered[
                    (df_filtered[feat_col] >= f_min) & (df_filtered[feat_col] <= f_max)
                ]

        if len(df_filtered) == 0:
            self.ax.set_title("No data after filtering")
            self.canvas.draw_idle()
            return

        x_vals = df_filtered[x_col].values
        y_vals = df_filtered[y_col].values

        # Choose color dimension
        c_data = None
        if plot_option != "Density":
            c_data = self._get_cdata_for_plot_option(plot_option, df_filtered)

        bins = self._parse_bins(self.density_bins_var.get())
        norm = mcolors.LogNorm() if self.log_scale_var.get() else None

        # If user wants color by weights, override c_data
        if self.color_by_weight_var.get() and "weights" in self.df.columns:
            c_data = self.df["weights"].values

        if plot_type == "Scatter":
            if plot_option == "Density" and not self.color_by_weight_var.get():
                self._plot_density_scatter(x_vals, y_vals, bins, norm)
            else:
                if c_data is not None:
                    sc = self.ax.scatter(x_vals, y_vals, c=c_data, cmap="viridis", s=5, norm=norm)
                    self.figure.colorbar(sc, ax=self.ax,
                                         label=("Weight" if self.color_by_weight_var.get() else plot_option))
            if do_ols:
                self._plot_ols_line(x_vals, y_vals)
        else:
            if plot_option == "Density" and not self.color_by_weight_var.get():
                self._plot_density_heatmap(x_vals, y_vals, bins, norm)
            else:
                if c_data is not None:
                    stat, x_edges, y_edges, _ = binned_statistic_2d(
                        x_vals, y_vals, c_data, statistic='mean', bins=bins
                    )
                    Xmesh, Ymesh = np.meshgrid(x_edges, y_edges)
                    im = self.ax.pcolormesh(Xmesh, Ymesh, stat.T,
                                            cmap='viridis', shading='auto', norm=norm)
                    self.figure.colorbar(
                        im, ax=self.ax,
                        label=("Weight" if self.color_by_weight_var.get() else f"{plot_option} (avg)")
                    )

        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        title_str = "Teacher vs. Student (OLS)" if do_ols else f"{plot_option} ({plot_type})"
        self.ax.set_title(title_str)

        # Draw cluster boundaries if requested
        if self.show_main_clusters_var.get() and self.boundaries:
            for lbl, poly in self.boundaries.items():
                exterior = getattr(poly, "exterior", None)
                if exterior:
                    xp, yp = exterior.xy
                    self.ax.plot(xp, yp, color="red", linewidth=1.5)

        self.canvas.draw_idle()

    def _apply_normalization(self, df):
        cols_to_scale = set()
        if self.config.state_columns:
            cols_to_scale.update(self.config.state_columns)
        if self.config.teacher_control_columns:
            cols_to_scale.update(self.config.teacher_control_columns)
        if self.config.student_control_columns:
            cols_to_scale.update(self.config.student_control_columns)

        cols_to_scale = [c for c in cols_to_scale if c in df.columns]
        if not cols_to_scale:
            return df

        if self.config.norm_file_path is not None:
            try:
                norm_df = pd.read_csv(self.config.norm_file_path, comment="#", index_col=0)
                if "min" not in norm_df.index or "max" not in norm_df.index:
                    messagebox.showwarning("Normalization File Error",
                                           f"Expected rows named 'min' and 'max' in {self.config.norm_file_path}")
                    return df
                min_vals = norm_df.loc["min"]
                max_vals = norm_df.loc["max"]
                for c in cols_to_scale:
                    if c in min_vals and c in max_vals:
                        cmin = min_vals[c]
                        cmax = max_vals[c]
                        rng = cmax - cmin
                        df[c] = 0.0 if abs(rng) < 1e-12 else 2*(df[c] - cmin)/rng - 1
            except Exception as e:
                messagebox.showwarning("Normalization File Error", f"Could not parse file: {e}")
        else:
            if not self.already_warned_about_local_norm:
                messagebox.showwarning("Normalization Warning",
                                       "No file found. Using dataset-based min/max.")
                self.already_warned_about_local_norm = True
            for c in cols_to_scale:
                cmin = df[c].min()
                cmax = df[c].max()
                rng = cmax - cmin
                df[c] = 0.0 if abs(rng) < 1e-12 else 2*(df[c] - cmin)/rng - 1
        return df

    def _force_teacher_student_xy(self, df):
        t = self.teacher_col_var.get()
        s = self.student_col_var.get()
        y_col = f"{s}_clipped" if self.clip_control_var.get() and f"{s}_clipped" in df.columns else s
        return t, y_col

    def _map_control_filter_to_column(self, choice, df, x_col, y_col):
        if choice in df.columns:
            return choice
        elif choice == "Absolute Error":
            s = self.student_col_var.get()
            return f"{s}_abs_err_clipped" if self.clip_control_var.get() else f"{s}_abs_err"
        elif choice == "Relative Error":
            s = self.student_col_var.get()
            return f"{s}_rel_err_clipped" if self.clip_control_var.get() else f"{s}_rel_err"
        elif choice == "Density":
            bins = self._parse_bins(self.density_bins_var.get())
            x_vals_temp = df[x_col].values
            y_vals_temp = df[y_col].values
            hist, xedges, yedges = np.histogram2d(x_vals_temp, y_vals_temp, bins=bins)
            density = np.zeros_like(x_vals_temp, dtype=float)
            x_bin_idx = np.digitize(x_vals_temp, xedges) - 1
            y_bin_idx = np.digitize(y_vals_temp, yedges) - 1
            x_bin_idx = np.clip(x_bin_idx, 0, bins - 1)
            y_bin_idx = np.clip(y_bin_idx, 0, bins - 1)
            for i in range(len(x_vals_temp)):
                density[i] = hist[x_bin_idx[i], y_bin_idx[i]]
            df["__density"] = density
            return "__density"
        return None

    def _compute_pls_components(self, df):
        x_cols = [col for col in self.config.state_columns if col in df.columns]
        if len(x_cols) == 0 or "__target_for_pls" not in df.columns:
            return
        X = df[x_cols].values
        Y = df["__target_for_pls"].values.reshape(-1, 1)
        pls = PLSRegression(n_components=2, scale=False)
        try:
            pls.fit(X, Y)
            scores = pls.transform(X)
            df["PLS_1"] = scores[:, 0]
            df["PLS_2"] = scores[:, 1]
        except Exception as e:
            messagebox.showwarning("PLS Error", f"Could not compute PLS: {e}")

    def _get_pls_target_column(self, plot_option, df):
        return self._get_cdata_for_plot_option(plot_option, df, for_pls=True)

    def _plot_ols_line(self, x_data, y_data):
        if len(x_data) < 2:
            return
        X_reshaped = x_data.reshape(-1, 1)
        model = LinearRegression().fit(X_reshaped, y_data)
        slope = model.coef_[0]
        intercept = model.intercept_
        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        y_fit = slope * x_fit + intercept
        self.ax.plot(x_fit, y_fit, color="red",
                     label=f"OLS: slope={slope:.3f}, intercept={intercept:.3f}")
        self.ax.legend()

    def _plot_density_scatter(self, x_vals, y_vals, bins, norm):
        hist, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=bins)
        c_data = np.zeros_like(x_vals, dtype=float)
        x_bin_idx = np.digitize(x_vals, xedges) - 1
        y_bin_idx = np.digitize(y_vals, yedges) - 1
        x_bin_idx = np.clip(x_bin_idx, 0, bins - 1)
        y_bin_idx = np.clip(y_bin_idx, 0, bins - 1)
        for i in range(len(x_vals)):
            c_data[i] = hist[x_bin_idx[i], y_bin_idx[i]]
        sc = self.ax.scatter(x_vals, y_vals, c=c_data, cmap="viridis", s=5, norm=norm)
        self.figure.colorbar(sc, ax=self.ax, label="Density (bin count)")

    def _plot_density_heatmap(self, x_vals, y_vals, bins, norm):
        counts, xedges, yedges, im = self.ax.hist2d(x_vals, y_vals, bins=bins, cmap="viridis", norm=norm)
        self.figure.colorbar(im, ax=self.ax, label="Density (bin count)")

    def _parse_bins(self, bins_str):
        try:
            val = int(bins_str)
            return val if val > 0 else 50
        except ValueError:
            return 50

    def _parse_range(self, s_min, s_max):
        val_min, val_max = None, None
        try:
            s_min, s_max = s_min.strip(), s_max.strip()
            if s_min != "":
                val_min = float(s_min)
            if s_max != "":
                val_max = float(s_max)
        except ValueError:
            return None, None
        return val_min, val_max

    def _get_cdata_for_plot_option(self, option, df, for_pls=False):
        def get_scol_clipped(scol, dframe):
            if self.clip_control_var.get():
                clipped = f"{scol}_clipped"
                return (dframe[clipped].values if clipped in dframe.columns
                        else dframe[scol].clip(-1, 1).values)
            else:
                return dframe[scol].values

        if option == "Teacher Control":
            t_col = self.teacher_col_var.get()
            return df[t_col].values if t_col in df.columns else None

        elif option == "Student Control":
            s_col = self.student_col_var.get()
            if s_col not in df.columns:
                return None
            return get_scol_clipped(s_col, df)

        elif option == "Absolute Error":
            s_col = self.student_col_var.get()
            col = f"{s_col}_abs_err_clipped" if self.clip_control_var.get() else f"{s_col}_abs_err"
            return df[col].values if col in df.columns else None

        elif option == "Relative Error":
            s_col = self.student_col_var.get()
            col = f"{s_col}_rel_err_clipped" if self.clip_control_var.get() else f"{s_col}_rel_err"
            return df[col].values if col in df.columns else None

        elif option == "Density":
            if len(self.config.state_columns) < 2:
                return None
            col1, col2 = self.config.state_columns[:2]
            if col1 not in df.columns or col2 not in df.columns:
                return None
            bins = self._parse_bins(self.density_bins_var.get())
            x_vals_temp = df[col1].values
            y_vals_temp = df[col2].values
            hist, xedges, yedges = np.histogram2d(x_vals_temp, y_vals_temp, bins=bins)
            density = np.zeros_like(x_vals_temp, dtype=float)
            x_bin_idx = np.digitize(x_vals_temp, xedges) - 1
            y_bin_idx = np.digitize(y_vals_temp, yedges) - 1
            x_bin_idx = np.clip(x_bin_idx, 0, bins - 1)
            y_bin_idx = np.clip(y_bin_idx, 0, bins - 1)
            for i in range(len(x_vals_temp)):
                density[i] = hist[x_bin_idx[i], y_bin_idx[i]]
            return density

        return None

    def sample_high_error(self):
        sampler = Sampler(self.config)
        sampled_df = sampler.sample_high_error(self.df, top_n=50)
        if sampled_df.empty:
            messagebox.showinfo("No Samples", "No error columns found.")
            return
        self.save_sample_to_csv(sampled_df, "high_error")

    def sample_poorly_represented(self):
        sampler = Sampler(self.config)
        sampled_df = sampler.sample_poorly_represented_regions(self.df, n=50)
        if sampled_df.empty:
            messagebox.showinfo("No Samples", "Dataframe empty.")
            return
        self.save_sample_to_csv(sampled_df, "low_density")

    def sample_uniform(self):
        sampler = Sampler(self.config)
        sampled_df = sampler.sample_uniform(self.df, n=50)
        if sampled_df.empty:
            messagebox.showinfo("No Samples", "Dataframe empty.")
            return
        self.save_sample_to_csv(sampled_df, "uniform")

    def save_sample_to_csv(self, sampled_df, sample_name="sample"):
        filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                initialfile=f"{sample_name}_sample.csv",
                                                title="Save sample")
        if filename:
            sampled_df.to_csv(filename, index=False)
            messagebox.showinfo("Saved", f"Sample saved to {filename}")
