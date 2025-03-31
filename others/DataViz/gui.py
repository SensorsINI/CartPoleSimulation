# gui.py
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression  # still used in BFS logic or weighting, if needed

from data_processor import DataProcessor
from sampler import Sampler
from weight_manager import WeightManager

# Import sub-frames
from gui_elements.frames_axes import AxesSelectionFrame
from gui_elements.frames_plot import PlotOptionsFrame, PlotTypeFrame
from gui_elements.frames_filter import (
    ControlFilterFrame, FeatureFilterFrame, DensitySettingsFrame,
    StepsRemovalFrame, ErrorSettingsFrame
)
from gui_elements.frames_weight import WeightingFrame, on_export_weights, on_coverage_changed, on_alpha_changed
from gui_elements.sampling import SamplingFrame

# Import the separate modules with big logic
from gui_elements.plot_logic import update_plot, apply_normalization, force_teacher_student_xy

matplotlib.use("TkAgg")


class MainApplication(tk.Tk):
    """Main Tkinter GUI, orchestrates sub-frames but delegates heavy logic to separate modules."""

    def __init__(self, config, df):
        super().__init__()
        self.config = config
        self.original_df = df
        self.df = df.copy()
        self.processor = DataProcessor(config)

        # WeightManager and cluster-related data
        self.weight_manager = WeightManager()
        self.last_labels = None
        self.last_main_labels = None
        self.main_clusters = None
        self.boundaries = None

        self.title("Robot State Data Analysis")
        self.geometry("1600x1200")
        self.lift()
        self.attributes("-topmost", True)
        self.after_idle(lambda: self.attributes("-topmost", False))

        # For completeness if using grid
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # If you have a "already_warned_about_local_norm" flag:
        self.already_warned_about_local_norm = False

        # --------------------------------------------------------------------------------
        # Shared Tkinter variables for all frames
        # --------------------------------------------------------------------------------
        self.x_var = tk.StringVar()
        self.y_var = tk.StringVar()
        self.teacher_col_var = tk.StringVar()
        self.student_col_var = tk.StringVar()
        self.use_pls_var = tk.BooleanVar(value=False)
        self.use_ols_var = tk.BooleanVar(value=False)
        self.normalize_var = tk.BooleanVar(value=False)
        self.plot_var = tk.StringVar(value="Teacher Control")
        self.plot_type_var = tk.StringVar(value="Scatter")
        self.clip_control_var = tk.BooleanVar(value=True)
        self.control_filter_var = tk.StringVar()
        self.filter_min_var = tk.StringVar(value="")
        self.filter_max_var = tk.StringVar(value="")
        self.feature_filter_var = tk.StringVar(value="")
        self.feature_filter_min_var = tk.StringVar(value="")
        self.feature_filter_max_var = tk.StringVar(value="")
        self.density_bins_var = tk.StringVar(value="50")
        self.log_scale_var = tk.BooleanVar(value=False)
        self.n_steps_var = tk.IntVar(value=0)
        self.error_cap_var = tk.StringVar(value=str(self.config.relative_error_cap))
        self.coverage_var = tk.IntVar(value=90)
        self.alpha_var = tk.DoubleVar(value=1.0)
        self.weighting_scheme_var = tk.StringVar(value="Mixed")
        self.show_main_clusters_var = tk.BooleanVar(value=False)
        self.color_by_weight_var = tk.BooleanVar(value=False)

        # Eps variable & color-by-cluster
        self.eps_var = tk.DoubleVar(value=0.5)
        self.color_by_cluster_var = tk.BooleanVar(value=False)

        # --------------------------------------------------------------------------------
        # Build layout & figure
        # --------------------------------------------------------------------------------
        self._create_layout()

        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Populate teacher/student, bind triggers, initial plot
        self._populate_control_inputs_dropdowns()
        self._bind_triggers()
        self.update_plot()

    def _create_layout(self):
        """Create scrollable left column for controls, plus right side plot."""
        self.left_side_frame = tk.Frame(self)
        self.left_side_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        # Fix the left pane width
        self.left_side_frame.pack_propagate(False)
        self.left_side_frame.configure(width=325)

        self.control_canvas = tk.Canvas(self.left_side_frame)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(
            self.left_side_frame, orient="vertical",
            command=self.control_canvas.yview
        )
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.control_frame = tk.Frame(self.control_canvas)
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor='nw')

        def _on_frame_configure(event):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        self.control_frame.bind("<Configure>", _on_frame_configure)

        def _on_mousewheel(event):
            if sys.platform == 'darwin':  # macOS
                # Sur macOS, event.delta donne déjà un incrément adapté
                self.control_canvas.yview_scroll(-1 * int(event.delta), "units")
            else:
                # Sous Windows, on divise par 120 pour obtenir le bon pas
                self.control_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        self.control_canvas.bind("<Enter>", lambda event: self.control_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.control_canvas.bind("<Leave>", lambda event: self.control_canvas.unbind_all("<MouseWheel>"))

        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Sub-frames
        self.axes_frame = AxesSelectionFrame(self.control_frame, self)
        self.axes_frame.pack(fill=tk.X, pady=5)

        self.plot_options_frame = PlotOptionsFrame(self.control_frame, self)
        self.plot_options_frame.pack(fill=tk.X, pady=5)

        self.plot_type_frame = PlotTypeFrame(self.control_frame, self)
        self.plot_type_frame.pack(fill=tk.X, pady=5)

        self.control_filter_frame = ControlFilterFrame(self.control_frame, self)
        self.control_filter_frame.pack(fill=tk.X, pady=5)

        self.feature_filter_frame = FeatureFilterFrame(self.control_frame, self)
        self.feature_filter_frame.pack(fill=tk.X, pady=5)

        self.density_frame = DensitySettingsFrame(self.control_frame, self)
        self.density_frame.pack(fill=tk.X, pady=5)

        self.sampling_frame = SamplingFrame(self.control_frame, self, sampler=Sampler)
        self.sampling_frame.pack(fill=tk.X, pady=5)

        self.steps_frame = StepsRemovalFrame(self.control_frame, self)
        self.steps_frame.pack(fill=tk.X, pady=5)

        self.error_frame = ErrorSettingsFrame(self.control_frame, self)
        self.error_frame.pack(fill=tk.X, pady=5)

        self.weighting_frame = WeightingFrame(self.control_frame, self)
        self.weighting_frame.pack(fill=tk.X, pady=5)

    def _bind_triggers(self):
        """Bind changes in tk.Variables so that the plot updates automatically."""
        # Axes changes
        self.x_var.trace_add("write", lambda *args: self.update_plot())
        self.y_var.trace_add("write", lambda *args: self.update_plot())

        # Teacher/Student
        self.teacher_col_var.trace_add("write", lambda *args: self.update_plot())
        self.student_col_var.trace_add("write", lambda *args: self.update_plot())

        # Plot options
        self.plot_var.trace_add("write", lambda *args: self.update_plot())
        self.plot_type_var.trace_add("write", lambda *args: self.update_plot())

        # Checkbuttons
        self.use_pls_var.trace_add("write", lambda *args: self.update_plot())
        self.use_ols_var.trace_add("write", lambda *args: self.update_plot())
        self.normalize_var.trace_add("write", lambda *args: self.update_plot())
        self.clip_control_var.trace_add("write", lambda *args: self.update_plot())

        # Filters
        self.control_filter_var.trace_add("write", lambda *args: self.update_plot())
        self.filter_min_var.trace_add("write", lambda *args: self.update_plot())
        self.filter_max_var.trace_add("write", lambda *args: self.update_plot())

        self.feature_filter_var.trace_add("write", lambda *args: self.update_plot())
        self.feature_filter_min_var.trace_add("write", lambda *args: self.update_plot())
        self.feature_filter_max_var.trace_add("write", lambda *args: self.update_plot())

        # Density
        self.density_bins_var.trace_add("write", lambda *args: self.update_plot())
        self.log_scale_var.trace_add("write", lambda *args: self.update_plot())

        # Steps
        self.n_steps_var.trace_add("write", lambda *args: self.apply_n_step_removal())

    # ----------------------------------------------------------------------------------
    # Basic Setup & Filtering Methods
    # ----------------------------------------------------------------------------------

    def _populate_control_inputs_dropdowns(self):
        """Populate teacher/student from config, if available."""
        t_cols = self.config.teacher_control_columns or []
        if t_cols:
            self.teacher_col_var.set(t_cols[0])

        s_cols = self.config.student_control_columns or []
        if s_cols:
            self.student_col_var.set(s_cols[0])

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

    # ----------------------------------------------------------------------------------
    # BFS + Weighting Callbacks (moved to weighting_logic.py, but we wrap them here)
    # ----------------------------------------------------------------------------------

    def _on_coverage_changed(self, val):
        """Called when coverage slider is changed."""
        on_coverage_changed(self, val)

    def _on_alpha_changed(self, val):
        """Called when alpha slider is changed."""
        on_alpha_changed(self, val)

    def _on_export_weights(self):
        """Called when user clicks 'Export Weighted CSV'."""
        on_export_weights(self)

    # ----------------------------------------------------------------------------------
    # Main Plot
    # ----------------------------------------------------------------------------------

    def update_plot(self):
        """Re-draw the entire plot (delegated to plot_logic.update_plot)."""
        update_plot(self)  # calls the big function from plot_logic.py

    def _parse_bins(self, bins_str):
        try:
            return int(bins_str)
        except ValueError:
            return 50
