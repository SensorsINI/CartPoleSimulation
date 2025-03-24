import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from typing import List, Optional
from tqdm import tqdm
from scipy.stats import binned_statistic_2d
import matplotlib.colors as mcolors

# For the OLS regression line
from sklearn.linear_model import LinearRegression

# For dimension reduction via PLS
from sklearn.cross_decomposition import PLSRegression

matplotlib.use("TkAgg")  # Use TkAgg backend for matplotlib


################################################################################
# Configuration
################################################################################

class Config:
    """
    Holds user-defined configuration for data loading and column names.
    """

    def __init__(self) -> None:
        # Path to the folder containing CSV files
        self.data_folder: str = r"./SI_Toolkit_ASF/Experiments/Train_NN_what_new/"

        # Column names for states (features for PLS)
        self.state_columns: List[str] = [
            "angle", "angleD", "angle_cos", "angle_sin",
            "position", "positionD", "target_position", "target_equilibrium"
        ]

        # Column names for the teacher's control input
        self.teacher_control_columns: List[str] = ["Q_calculated_offline"]

        # Column names for the student's (NN) control input
        self.student_control_columns: Optional[List[str]] = ["Q_calculated_offline_NN"]

        # The maximum relative error to allow before clipping
        self.relative_error_cap: float = 10.0

        # Number of starting steps to remove from each experiment (default 50)
        self.n_starting_steps: int = 50

        # Optional path to a normalization file (or None).
        self.norm_file_path: Optional[str] = './SI_Toolkit_ASF/Experiments/GRU-7IN-32H1-32H2-1OUT-3/NI_2024-08-17_22-25-35.csv'
        # Example:
        # self.norm_file_path = r"./my_normalization.csv"


################################################################################
# Data Loading
################################################################################

class DataLoader:
    """
    Responsible for loading CSV files into a single pandas DataFrame.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """
        Loads all CSV files from the `config.data_folder` directory and combines them.
        """
        folder_path = self.config.data_folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {folder_path}")

        df_list = []
        for i, file in enumerate(tqdm(csv_files, desc="Loading CSV files")):
            df_temp = pd.read_csv(file, comment="#")
            df_temp["experiment_id"] = i
            df_list.append(df_temp)

        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df


################################################################################
# Data Processing
################################################################################

class DataProcessor:
    """
    Computes absolute and relative errors between teacher and student columns,
    and handles capping of the relative error to avoid outliers.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def compute_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Precompute absolute and relative errors for each (teacher, student) pair
        if student controls exist. Rewrites or creates columns in-place.
        """
        if not self.config.student_control_columns:
            return df  # If no student columns, do nothing

        teacher_cols = self.config.teacher_control_columns
        student_cols = self.config.student_control_columns

        pair_count = min(len(teacher_cols), len(student_cols))
        teacher_cols = teacher_cols[:pair_count]
        student_cols = student_cols[:pair_count]

        for t_col, s_col in tqdm(zip(teacher_cols, student_cols), desc='Computing Errors'):
            if t_col not in df.columns or s_col not in df.columns:
                continue

            # Absolute error
            abs_err_col = f"{s_col}_abs_err"
            df[abs_err_col] = (df[t_col] - df[s_col]).abs()

            # Relative error
            rel_err_col = f"{s_col}_rel_err"
            epsilon = 1e-6
            df[rel_err_col] = df[abs_err_col] / (df[t_col].abs() + epsilon)
            df[rel_err_col] = df[rel_err_col].clip(upper=self.config.relative_error_cap)

            # Also store a clipped version of the student's control
            df[f"{s_col}_clipped"] = df[s_col].clip(-1, 1)

            # Clipped absolute error
            abs_err_col_clipped = f"{s_col}_abs_err_clipped"
            df[abs_err_col_clipped] = (df[t_col] - df[f"{s_col}_clipped"]).abs()

            # Clipped relative error
            rel_err_col_clipped = f"{s_col}_rel_err_clipped"
            df[rel_err_col_clipped] = (
                df[abs_err_col_clipped] / (df[t_col].abs() + epsilon)
            ).clip(upper=self.config.relative_error_cap)

        return df


################################################################################
# Sampler
################################################################################

class Sampler:
    """
    Samples states from the dataset in various ways to improve coverage
    or focus on trouble spots.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def sample_high_error(self, df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """
        Returns subset of rows with highest sum of absolute errors (sum across all student cols).
        """
        if not self.config.student_control_columns:
            return pd.DataFrame()

        abs_err_columns = [col for col in df.columns if col.endswith("_abs_err")]
        if not abs_err_columns:
            return pd.DataFrame()

        df_copy = df.copy()
        df_copy["total_abs_err"] = df_copy[abs_err_columns].sum(axis=1)

        df_sorted = df_copy.sort_values(by="total_abs_err", ascending=False)
        high_error_samples = df_sorted.head(top_n)

        return high_error_samples

    def sample_poorly_represented_regions(self, df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
        """
        Stub function: returns random sample.
        In principle, you'd define your own logic for "low density" regions.
        """
        if len(df) == 0:
            return pd.DataFrame()
        return df.sample(n=min(n, len(df)), replace=False)

    def sample_uniform(self, df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
        """
        Simple random sample from the dataset.
        """
        if len(df) == 0:
            return pd.DataFrame()
        return df.sample(n=min(n, len(df)), replace=False)


################################################################################
# Main GUI Application
################################################################################

class MainApplication(tk.Tk):
    """
    Main Tkinter application that provides a GUI for loading data, selecting
    axes, plotting, sampling, and performing dimension reduction with PLS or OLS,
    plus optional normalization of states & controls.
    """

    def __init__(self, config: Config, df: pd.DataFrame) -> None:
        super().__init__()

        self.config = config
        self.original_df = df  # Unmodified data (with precomputed errors)
        self.df = df.copy()    # Working copy (used for in-GUI transformations)

        # Keep a reference to our DataProcessor so we can re-compute errors after normalization
        self.processor = DataProcessor(config)

        self.title("Robot State Data Analysis")

        # Increase default window size
        self.geometry("1200x800")

        # Make the main window appear on top (at least once)
        self.lift()
        self.attributes("-topmost", True)
        self.after_idle(lambda: self.attributes("-topmost", False))

        # Make main window expandable
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # --- ADDED: one-time flag to warn about local normalization
        self.already_warned_about_local_norm = False

        # Create UI widgets
        self.create_widgets()

        # Setup matplotlib figure
        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Populate the teacher/student control dropdowns
        self.populate_control_inputs_dropdowns()

        # Populate X/Y dropdowns with state columns
        self.x_dropdown["values"] = self.config.state_columns
        self.y_dropdown["values"] = self.config.state_columns
        if len(self.config.state_columns) >= 2:
            self.x_dropdown.current(0)
            self.y_dropdown.current(1)

        # Populate feature filter dropdown with all columns
        self.feature_filter_dropdown["values"] = sorted(df.columns)
        if len(df.columns) > 0:
            self.feature_filter_dropdown.current(0)

        # Also populate the "Filter by Control Column" dropdown
        all_control_columns = []
        if self.config.teacher_control_columns:
            all_control_columns.extend(self.config.teacher_control_columns)
        if self.config.student_control_columns:
            all_control_columns.extend(self.config.student_control_columns)
        # Remove duplicates
        all_control_columns = list(dict.fromkeys(all_control_columns))
        # Add extra options for error and density filtering
        all_control_columns += ["Absolute Error", "Relative Error", "Density"]
        self.control_filter_dropdown["values"] = all_control_columns
        if len(all_control_columns) > 0:
            self.control_filter_dropdown.current(0)

        # Bind combobox changes to auto-update the plot
        self.x_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.y_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.teacher_col_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.student_col_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.plot_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.plot_type_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.feature_filter_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.control_filter_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        # Auto-update on text/checkbox changes
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

        # When user changes "Remove N steps" from the GUI
        self.n_steps_var.trace_add("write", lambda *args: self.apply_n_step_removal())

        # Initial plot
        self.update_plot()

    def create_widgets(self) -> None:
        """
        Create and position the Tkinter widgets (dropdowns, text entries, etc.).
        """
        # Left panel for controls
        self.control_frame = tk.Frame(self)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Right panel for plot
        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ========== Axes Selection, PLS Toggle, OLS Toggle ==========
        axes_frame = tk.LabelFrame(self.control_frame, text="Axes Selection")
        axes_frame.pack(fill=tk.X, pady=5)

        tk.Label(axes_frame, text="Select X-axis:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.x_var = tk.StringVar()
        self.x_dropdown = ttk.Combobox(
            axes_frame, textvariable=self.x_var, state="readonly", width=16
        )
        self.x_dropdown.grid(row=0, column=1, padx=2, pady=2)

        tk.Label(axes_frame, text="Select Y-axis:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.y_var = tk.StringVar()
        self.y_dropdown = ttk.Combobox(
            axes_frame, textvariable=self.y_var, state="readonly", width=16
        )
        self.y_dropdown.grid(row=1, column=1, padx=2, pady=2)

        # Checkbox to use PLS for dimension reduction
        self.use_pls_var = tk.BooleanVar(value=False)
        self.use_pls_check = tk.Checkbutton(
            axes_frame,
            text="Use PLS for X-Y (2D)",
            variable=self.use_pls_var
        )
        self.use_pls_check.grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky="w")

        # Checkbox to forcibly use OLS with Teacher on X, Student on Y
        self.use_ols_var = tk.BooleanVar(value=False)
        self.use_ols_check = tk.Checkbutton(
            axes_frame,
            text="Use OLS (Teacher vs. Student) for X-Y",
            variable=self.use_ols_var
        )
        self.use_ols_check.grid(row=3, column=0, columnspan=2, padx=2, pady=2, sticky="w")

        # Checkbox for normalization
        self.normalize_var = tk.BooleanVar(value=False)
        self.normalize_check = tk.Checkbutton(
            axes_frame,
            text="Normalize States/Controls [-1..1]",
            variable=self.normalize_var
        )
        self.normalize_check.grid(row=4, column=0, columnspan=2, padx=2, pady=2, sticky="w")

        # ========== Teacher/Student Control Selection ==========
        ctrl_frame = tk.LabelFrame(self.control_frame, text="Teacher/Student Control")
        ctrl_frame.pack(fill=tk.X, pady=5)

        tk.Label(ctrl_frame, text="Teacher Control:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.teacher_col_var = tk.StringVar()
        self.teacher_col_dropdown = ttk.Combobox(
            ctrl_frame, textvariable=self.teacher_col_var, state="readonly", width=16
        )
        self.teacher_col_dropdown.grid(row=0, column=1, padx=2, pady=2)

        tk.Label(ctrl_frame, text="Student Control:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.student_col_var = tk.StringVar()
        self.student_col_dropdown = ttk.Combobox(
            ctrl_frame, textvariable=self.student_col_var, state="readonly", width=16
        )
        self.student_col_dropdown.grid(row=1, column=1, padx=2, pady=2)

        # ========== Plot Options ==========
        plotopts_frame = tk.LabelFrame(self.control_frame, text="Plot Options")
        plotopts_frame.pack(fill=tk.X, pady=5)

        tk.Label(plotopts_frame, text="Plot Option:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.plot_var = tk.StringVar()
        self.plot_options = [
            "Teacher Control",
            "Student Control",
            "Absolute Error",
            "Relative Error",
            "Density"
        ]
        self.plot_dropdown = ttk.Combobox(
            plotopts_frame,
            textvariable=self.plot_var,
            values=self.plot_options,
            state="readonly",
            width=16
        )
        self.plot_dropdown.current(0)
        self.plot_dropdown.grid(row=0, column=1, padx=2, pady=2)

        tk.Label(plotopts_frame, text="Plot Type:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.plot_type_var = tk.StringVar(value="Scatter")
        self.plot_type_dropdown = ttk.Combobox(
            plotopts_frame,
            textvariable=self.plot_type_var,
            values=["Scatter", "Heatmap"],
            state="readonly",
            width=16
        )
        self.plot_type_dropdown.current(0)
        self.plot_type_dropdown.grid(row=1, column=1, padx=2, pady=2)

        # Checkbox to toggle clipping for student's control
        self.clip_control_var = tk.BooleanVar(value=True)
        self.clip_control_check = tk.Checkbutton(
            plotopts_frame,
            text="Clip Student Control [-1,1]",
            variable=self.clip_control_var
        )
        self.clip_control_check.grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky="w")

        # ========== Control Input Filter ==========
        filter_frame = tk.LabelFrame(self.control_frame, text="Control Input Filter")
        filter_frame.pack(pady=5, fill=tk.X)

        tk.Label(filter_frame, text="Filter by:").pack(pady=2)
        self.control_filter_var = tk.StringVar()
        self.control_filter_dropdown = ttk.Combobox(
            filter_frame, textvariable=self.control_filter_var, state="readonly", width=20
        )
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

        # ========== Additional Feature Filter ==========
        feature_filter_frame = tk.LabelFrame(self.control_frame, text="Additional Feature Filter")
        feature_filter_frame.pack(pady=5, fill=tk.X)

        tk.Label(feature_filter_frame, text="Feature:").pack(side=tk.LEFT, padx=2)
        self.feature_filter_var = tk.StringVar(value="")
        self.feature_filter_dropdown = ttk.Combobox(
            feature_filter_frame, textvariable=self.feature_filter_var, state="readonly", width=12
        )
        self.feature_filter_dropdown.pack(side=tk.LEFT, padx=2)

        tk.Label(feature_filter_frame, text="Min:").pack(side=tk.LEFT, padx=2)
        self.feature_filter_min_var = tk.StringVar(value="")
        self.feature_filter_min_entry = tk.Entry(
            feature_filter_frame, textvariable=self.feature_filter_min_var, width=8
        )
        self.feature_filter_min_entry.pack(side=tk.LEFT, padx=2)

        tk.Label(feature_filter_frame, text="Max:").pack(side=tk.LEFT, padx=2)
        self.feature_filter_max_entry = tk.Entry(
            feature_filter_frame, textvariable=self.feature_filter_max_var, width=8
        )
        self.feature_filter_max_entry.pack(side=tk.LEFT, padx=2)

        # ========== Density Settings ==========
        density_frame = tk.LabelFrame(self.control_frame, text="Density Settings")
        density_frame.pack(pady=5, fill=tk.X)

        row_density = tk.Frame(density_frame)
        row_density.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(row_density, text="Bins:").pack(side=tk.LEFT, padx=2)
        self.density_bins_var = tk.StringVar(value="50")
        self.density_bins_entry = tk.Entry(
            row_density, textvariable=self.density_bins_var, width=6
        )
        self.density_bins_entry.pack(side=tk.LEFT, padx=2)

        self.log_scale_var = tk.BooleanVar(value=False)
        self.log_scale_check = tk.Checkbutton(
            density_frame,
            text="Log Scale",
            variable=self.log_scale_var
        )
        self.log_scale_check.pack(pady=2)

        # ========== Sampling Buttons (one row) ==========
        sampling_frame = tk.LabelFrame(self.control_frame, text="Sampling")
        sampling_frame.pack(pady=5, fill=tk.X)

        btn_high_err = tk.Button(sampling_frame, text="High Error",
                                 command=self.sample_high_error)
        btn_high_err.pack(side=tk.LEFT, padx=2, pady=2)

        btn_low_density = tk.Button(sampling_frame, text="Low Density",
                                    command=self.sample_poorly_represented)
        btn_low_density.pack(side=tk.LEFT, padx=2, pady=2)

        btn_uniform = tk.Button(sampling_frame, text="Uniform",
                                command=self.sample_uniform)
        btn_uniform.pack(side=tk.LEFT, padx=2, pady=2)

        # ========== Remove N Steps ==========
        steps_frame = tk.Frame(self.control_frame)
        steps_frame.pack(fill=tk.X, pady=5)

        tk.Label(steps_frame, text="Remove N steps:").pack(side=tk.LEFT, padx=2)
        self.n_steps_var = tk.IntVar(value=0)
        self.n_steps_entry = tk.Entry(steps_frame, textvariable=self.n_steps_var, width=8)
        self.n_steps_entry.pack(side=tk.LEFT, padx=2)

        # ========== Error Settings (NEW) ==========
        error_frame = tk.LabelFrame(self.control_frame, text="Error Settings")
        error_frame.pack(pady=5, fill=tk.X)

        tk.Label(error_frame, text="Relative Error Cap:").pack(side=tk.LEFT, padx=2)
        self.error_cap_var = tk.StringVar(value=str(self.config.relative_error_cap))
        self.error_cap_entry = tk.Entry(error_frame, textvariable=self.error_cap_var, width=8)
        self.error_cap_entry.pack(side=tk.LEFT, padx=2)

        apply_cap_button = tk.Button(
            error_frame, text="Apply", command=self.apply_error_cap
        )
        apply_cap_button.pack(side=tk.LEFT, padx=2)

    def populate_control_inputs_dropdowns(self) -> None:
        """
        Populate the teacher and student control dropdowns, if available.
        """
        # Teacher control
        teacher_cols = self.config.teacher_control_columns or []
        if teacher_cols:
            self.teacher_col_dropdown["values"] = teacher_cols
            self.teacher_col_dropdown.current(0)
        else:
            self.teacher_col_dropdown["values"] = []
            self.teacher_col_dropdown.config(state="disabled")

        # Student control
        student_cols = self.config.student_control_columns or []
        if student_cols:
            self.student_col_dropdown["values"] = student_cols
            self.student_col_dropdown.current(0)
        else:
            self.student_col_dropdown["values"] = []
            self.student_col_dropdown.config(state="disabled")

    def apply_n_step_removal(self):
        """
        Re-apply n-step removal to the entire dataset for each experiment.
        """
        n = self.n_steps_var.get()
        self.df = self.remove_n_starting_steps(self.original_df, n)
        self.update_plot()

    def remove_n_starting_steps(self, df, n):
        """
        Removes the first n rows from each experiment's data, grouped by experiment_id.
        """
        if n <= 0:
            return df
        return df.groupby("experiment_id", group_keys=False).apply(lambda g: g.iloc[n:])

    # ========== NEW METHOD to apply a changed Relative Error Cap ==========
    def apply_error_cap(self):
        """
        Reads the new relative error cap from the GUI, updates config,
        recomputes errors on the original DataFrame, and refreshes the plot.
        """
        try:
            new_cap = float(self.error_cap_var.get())
            # Update our config
            self.config.relative_error_cap = new_cap
            # Recompute errors in original_df
            self.original_df = self.processor.compute_errors(self.original_df)
            # Also re-apply n-step removal to refresh self.df
            self.df = self.remove_n_starting_steps(self.original_df, self.n_steps_var.get())
            self.update_plot()
        except ValueError:
            messagebox.showwarning("Invalid Value", "Please enter a valid number for relative error cap.")

    ###########################################################################
    # Main update_plot Method
    ###########################################################################
    def update_plot(self) -> None:
        """
        Refresh the plot based on user selections, applying:
          1) revert to original data
          2) remove N steps
          3) optionally normalize states/controls
          3b) recompute errors if normalized
          4) proceed with PLS/OLS or normal X/Y
          5) filter
          6) plot
        """
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)

        # 1) Start from original data
        df_plot = self.original_df.copy()

        # 2) Remove N steps per experiment
        n = self.n_steps_var.get()
        df_plot = df_plot.groupby("experiment_id", group_keys=False).apply(lambda g: g.iloc[n:])

        # 3) Optionally normalize states/controls
        if self.normalize_var.get():
            df_plot = self._apply_normalization(df_plot)
            # 3b) Recompute errors to reflect scaled teacher/student controls
            df_plot = self.processor.compute_errors(df_plot)

        # 4) Decide if OLS or PLS or normal
        plot_option = self.plot_var.get()
        if self.use_ols_var.get():
            x_col, y_col = self._force_teacher_student_xy(df_plot)
            plot_type = "Scatter"
            do_ols = True
        elif self.use_pls_var.get():
            pls_target = self._get_pls_target_column(plot_option, df_plot)
            if pls_target is None:
                self.ax.set_title("Could not compute PLS target. Check columns.")
                self.canvas.draw_idle()
                return
            df_plot["__target_for_pls"] = pls_target

            # Confirm we are applying PLS on the final dataset (which may be normalized)
            self._compute_pls_components(df_plot)
            x_col, y_col = "PLS_1", "PLS_2"
            plot_type = self.plot_type_var.get()
            do_ols = False
        else:
            x_col = self.x_var.get()
            y_col = self.y_var.get()
            plot_type = self.plot_type_var.get()
            do_ols = False

        # Check columns exist
        if x_col not in df_plot.columns or y_col not in df_plot.columns:
            self.ax.set_title(f"X or Y column not found: {x_col}, {y_col}.")
            self.canvas.draw_idle()
            return

        # 5) Filter
        df_filtered = df_plot.copy()

        # Filter by control or derived columns
        control_filter_choice = self.control_filter_var.get()
        col_to_filter = self._map_control_filter_to_column(control_filter_choice, df_filtered, x_col, y_col)
        cf_min, cf_max = self._parse_range(self.filter_min_var.get(), self.filter_max_var.get())
        if col_to_filter and (cf_min is not None) and (cf_max is not None) and (cf_min < cf_max):
            df_filtered = df_filtered[
                (df_filtered[col_to_filter] >= cf_min) & (df_filtered[col_to_filter] <= cf_max)
            ]

        # Additional feature-based filter
        feat_col = self.feature_filter_var.get()
        if feat_col in df_filtered.columns:
            f_min, f_max = self._parse_range(
                self.feature_filter_min_var.get(),
                self.feature_filter_max_var.get()
            )
            if f_min is not None and f_max is not None and f_min < f_max:
                df_filtered = df_filtered[
                    (df_filtered[feat_col] >= f_min) & (df_filtered[feat_col] <= f_max)
                ]

        if len(df_filtered) == 0:
            self.ax.set_title("No data to plot after filtering")
            self.canvas.draw_idle()
            return

        # Extract final X, Y
        x_vals = df_filtered[x_col].values
        y_vals = df_filtered[y_col].values

        # 6) Determine color dimension from "Plot Option"
        c_data = None
        if plot_option == "Density":
            pass
        else:
            c_data = self._get_cdata_for_plot_option(plot_option, df_filtered)

        bins = self._parse_bins(self.density_bins_var.get())
        norm = mcolors.LogNorm() if self.log_scale_var.get() else None

        # 7) Plot
        if plot_type == "Scatter":
            if plot_option == "Density":
                self._plot_density_scatter(x_vals, y_vals, bins, norm)
            else:
                if c_data is None:
                    self.ax.set_title(f"No valid data for {plot_option}.")
                else:
                    sc = self.ax.scatter(x_vals, y_vals, c=c_data, cmap="viridis", s=5, norm=norm)
                    self.figure.colorbar(sc, ax=self.ax, label=plot_option)
            if do_ols:
                self._plot_ols_line(x_vals, y_vals)

        else:  # "Heatmap"
            if plot_option == "Density":
                self._plot_density_heatmap(x_vals, y_vals, bins, norm)
            else:
                if c_data is not None:
                    stat, x_edges, y_edges, _ = binned_statistic_2d(
                        x_vals, y_vals, c_data, statistic='mean', bins=bins
                    )
                    Xmesh, Ymesh = np.meshgrid(x_edges, y_edges)
                    im = self.ax.pcolormesh(Xmesh, Ymesh, stat.T, cmap='viridis', shading='auto', norm=norm)
                    self.figure.colorbar(im, ax=self.ax, label=f"{plot_option} (avg)")
                else:
                    self.ax.set_title(f"No valid data for {plot_option}.")

        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        title_str = "Teacher vs. Student (OLS)" if do_ols else f"{plot_option} ({plot_type})"
        self.ax.set_title(title_str)
        self.canvas.draw_idle()

    ###########################################################################
    # Normalization Method
    ###########################################################################
    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale each relevant column (states, teacher, student) to [-1,1].
        If self.config.norm_file_path is not None, read min/max from that file;
        otherwise compute min/max from df directly.

        The file is expected to have rows named "min" and "max" (plus columns matching
        the relevant columns). We ignore any column not in that file or not in df.
        """
        cols_to_scale = []
        if self.config.state_columns:
            cols_to_scale.extend(self.config.state_columns)
        if self.config.teacher_control_columns:
            cols_to_scale.extend(self.config.teacher_control_columns)
        if self.config.student_control_columns:
            cols_to_scale.extend(self.config.student_control_columns)

        # Remove duplicates
        cols_to_scale = list(dict.fromkeys(cols_to_scale))
        # Keep only those actually in df
        cols_to_scale = [c for c in cols_to_scale if c in df.columns]

        if not cols_to_scale:
            return df

        if self.config.norm_file_path is not None:
            # Load the CSV with row index "min" and "max"
            try:
                norm_df = pd.read_csv(self.config.norm_file_path, comment="#", index_col=0)
                if "min" not in norm_df.index or "max" not in norm_df.index:
                    messagebox.showwarning(
                        "Normalization File Error",
                        f"Expected rows named 'min' and 'max' in {self.config.norm_file_path}"
                    )
                    return df

                min_vals = norm_df.loc["min"]
                max_vals = norm_df.loc["max"]

                for c in cols_to_scale:
                    if c in min_vals and c in max_vals:
                        cmin = min_vals[c]
                        cmax = max_vals[c]
                        rng = cmax - cmin
                        if abs(rng) < 1e-12:
                            df[c] = 0.0
                        else:
                            df[c] = 2 * (df[c] - cmin) / rng - 1
                    else:
                        # If not in the norm file, skip
                        pass

            except Exception as e:
                messagebox.showwarning("Normalization File Error",
                                       f"Could not parse {self.config.norm_file_path}: {e}")
                return df
        else:
            # Warn once if using dataset-based min/max
            if not self.already_warned_about_local_norm:
                messagebox.showwarning(
                    "Normalization Warning",
                    "No normalization file found. Using dataset-based min/max for scaling."
                )
                self.already_warned_about_local_norm = True

            # Compute min/max from the data
            for c in cols_to_scale:
                cmin = df[c].min()
                cmax = df[c].max()
                rng = cmax - cmin
                if abs(rng) < 1e-12:
                    df[c] = 0.0
                else:
                    df[c] = 2 * (df[c] - cmin) / rng - 1

        return df

    ###########################################################################
    # Force Teacher/Student for OLS
    ###########################################################################
    def _force_teacher_student_xy(self, df: pd.DataFrame) -> (str, str):
        """
        If "Use OLS" is checked, override X,Y with teacher vs. student.
        If "Clip" is on, we use the clipped column if present.
        """
        teacher_sel = self.teacher_col_var.get()
        student_sel = self.student_col_var.get()

        if self.clip_control_var.get():
            clipped_col = f"{student_sel}_clipped"
            if clipped_col in df.columns:
                y_col = clipped_col
            else:
                y_col = student_sel
        else:
            y_col = student_sel

        x_col = teacher_sel
        return x_col, y_col

    ###########################################################################
    # Map Filter Choice to Column
    ###########################################################################
    def _map_control_filter_to_column(self,
                                      choice: str,
                                      df: pd.DataFrame,
                                      x_col: str,
                                      y_col: str) -> Optional[str]:
        """
        Given the user's selection in the "Control Input Filter" combobox,
        decide which column in df to filter on.
        """
        if choice in df.columns:
            return choice
        elif choice == "Absolute Error":
            s_col = self.student_col_var.get()
            return f"{s_col}_abs_err_clipped" if self.clip_control_var.get() else f"{s_col}_abs_err"
        elif choice == "Relative Error":
            s_col = self.student_col_var.get()
            return f"{s_col}_rel_err_clipped" if self.clip_control_var.get() else f"{s_col}_rel_err"
        elif choice == "Density":
            # We compute density on x_col,y_col
            x_vals_temp = df[x_col].values
            y_vals_temp = df[y_col].values
            bins = self._parse_bins(self.density_bins_var.get())
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

    ###########################################################################
    # PLS: compute and get target
    ###########################################################################
    def _compute_pls_components(self, df: pd.DataFrame):
        """
        Compute PLS(2 components) from the state columns (X) to the data in __target_for_pls (Y).
        Store the result in df['PLS_1'] and df['PLS_2'].

        *** Use scale=False so that your own normalization (if any) is preserved.
        """
        x_cols = [col for col in self.config.state_columns if col in df.columns]
        if len(x_cols) == 0:
            return
        X = df[x_cols].values

        if "__target_for_pls" not in df.columns:
            return
        Y = df["__target_for_pls"].values.reshape(-1, 1)

        pls = PLSRegression(n_components=2, scale=False)
        try:
            pls.fit(X, Y)
            scores = pls.transform(X)  # shape (n_samples, 2)
            df["PLS_1"] = scores[:, 0]
            df["PLS_2"] = scores[:, 1]
        except Exception as e:
            messagebox.showwarning("PLS Error", f"Could not compute PLS: {e}")

    def _get_pls_target_column(self, plot_option: str, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Determine which column in df to use as the target (Y) for PLS.
        """
        return self._get_cdata_for_plot_option(plot_option, df, for_pls=True)

    ###########################################################################
    # OLS Plot
    ###########################################################################
    def _plot_ols_line(self, x_data: np.ndarray, y_data: np.ndarray):
        """
        Plots a best-fit line (OLS) for x_data vs. y_data in red.
        """
        if len(x_data) < 2:
            return

        X_reshaped = x_data.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_reshaped, y_data)
        slope = model.coef_[0]
        intercept = model.intercept_

        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        y_fit = slope * x_fit + intercept
        self.ax.plot(x_fit, y_fit, color="red",
                     label=f"OLS: slope={slope:.3f}, intercept={intercept:.3f}")
        self.ax.legend()

    ###########################################################################
    # Density Plots
    ###########################################################################
    def _plot_density_scatter(self, x_vals, y_vals, bins, norm):
        """
        Scatter mode for 'Density': color each point by the local bin count.
        """
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
        """
        Heatmap mode for 'Density': standard 2D histogram (bin counts) as an image.
        """
        counts, xedges, yedges, im = self.ax.hist2d(
            x_vals, y_vals, bins=bins, cmap="viridis", norm=norm
        )
        self.figure.colorbar(im, ax=self.ax, label="Density (bin count)")

    ###########################################################################
    # Color/Data Helpers
    ###########################################################################
    def _parse_bins(self, bins_str: str) -> int:
        try:
            val = int(bins_str)
            if val < 1:
                return 50
            return val
        except ValueError:
            return 50

    def _parse_range(self, s_min: str, s_max: str) -> (Optional[float], Optional[float]):
        val_min, val_max = None, None
        try:
            s_min = s_min.strip()
            s_max = s_max.strip()
            if s_min != "":
                val_min = float(s_min)
            if s_max != "":
                val_max = float(s_max)
        except ValueError:
            return None, None
        return val_min, val_max

    def _get_cdata_for_plot_option(self,
                                   plot_option: str,
                                   df: pd.DataFrame,
                                   for_pls: bool=False) -> Optional[np.ndarray]:
        """
        Determine the color dimension (or PLS target) based on the selected plot option.
        If for_pls=True, we typically want some 1D data to serve as the "Y" in PLS.
        """

        def get_student_col_base_or_clipped(scol: str, dframe: pd.DataFrame) -> np.ndarray:
            if self.clip_control_var.get():
                clipped_col = f"{scol}_clipped"
                if clipped_col in dframe.columns:
                    return dframe[clipped_col].values
                else:
                    return dframe[scol].clip(-1, 1).values
            else:
                return dframe[scol].values

        if plot_option == "Teacher Control":
            t_col = self.teacher_col_var.get()
            if t_col in df.columns:
                return df[t_col].values
            else:
                return None

        elif plot_option == "Student Control":
            s_col = self.student_col_var.get()
            if s_col not in df.columns:
                return None
            return get_student_col_base_or_clipped(s_col, df)

        elif plot_option == "Absolute Error":
            s_col = self.student_col_var.get()
            if self.clip_control_var.get():
                abs_err_col = f"{s_col}_abs_err_clipped"
            else:
                abs_err_col = f"{s_col}_abs_err"
            if abs_err_col in df.columns:
                return df[abs_err_col].values
            else:
                return None

        elif plot_option == "Relative Error":
            s_col = self.student_col_var.get()
            if self.clip_control_var.get():
                rel_err_col = f"{s_col}_rel_err_clipped"
            else:
                rel_err_col = f"{s_col}_rel_err"
            if rel_err_col in df.columns:
                return df[rel_err_col].values
            else:
                return None

        elif plot_option == "Density":
            # If for_pls=True and user picks "Density" as PLS target, that means
            # we compute the 2D histogram of the first two state columns (somewhat unusual).
            if len(self.config.state_columns) < 2:
                return None
            col1 = self.config.state_columns[0]
            col2 = self.config.state_columns[1]
            if (col1 not in df.columns) or (col2 not in df.columns):
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

    ###########################################################################
    # Sampling Methods (buttons)
    ###########################################################################
    def sample_high_error(self) -> None:
        sampler = Sampler(self.config)
        sampled_df = sampler.sample_high_error(self.df, top_n=50)
        if sampled_df.empty:
            messagebox.showinfo("No Samples",
                                "No student columns or error columns found; cannot sample by error.")
            return
        self.save_sample_to_csv(sampled_df, sample_name="high_error")

    def sample_poorly_represented(self) -> None:
        sampler = Sampler(self.config)
        sampled_df = sampler.sample_poorly_represented_regions(self.df, n=50)
        if sampled_df.empty:
            messagebox.showinfo("No Samples", "The dataframe is empty, cannot sample.")
            return
        self.save_sample_to_csv(sampled_df, sample_name="low_density")

    def sample_uniform(self) -> None:
        sampler = Sampler(self.config)
        sampled_df = sampler.sample_uniform(self.df, n=50)
        if sampled_df.empty:
            messagebox.showinfo("No Samples", "The dataframe is empty, cannot sample.")
            return
        self.save_sample_to_csv(sampled_df, sample_name="uniform")

    def save_sample_to_csv(self, sampled_df: pd.DataFrame, sample_name: str = "sample") -> None:
        """
        Prompt the user with a file dialog to save the sampled DataFrame to CSV.
        """
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"{sample_name}_sample.csv",
            title="Save sample to CSV"
        )
        if filename:
            sampled_df.to_csv(filename, index=False)
            messagebox.showinfo("Saved", f"Sample saved to {filename}")


################################################################################
# Main entry point
################################################################################

def main() -> None:
    """
    Main entry point for the application:
    1. Build config
    2. Load data
    3. Precompute errors
    4. Launch the GUI.
    """
    config = Config()
    # If you want a specific normalization file, uncomment and specify:
    # config.norm_file_path = r"./my_normalization_file.csv"

    data_loader = DataLoader(config)
    df = data_loader.load_data()

    # Precompute unnormalized errors
    processor = DataProcessor(config)
    df = processor.compute_errors(df)

    app = MainApplication(config, df)
    app.mainloop()


if __name__ == "__main__":
    main()
