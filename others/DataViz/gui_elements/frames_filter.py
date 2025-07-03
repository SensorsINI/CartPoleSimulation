# gui_elements/frames_filter.py

import tkinter as tk
from tkinter import ttk


class ControlFilterFrame(tk.LabelFrame):
    """Frame for control input filter (Min/Max)."""
    def __init__(self, parent, main_app, **kwargs):
        super().__init__(parent, text="Control Input Filter", **kwargs)
        self.main_app = main_app

        tk.Label(self, text="Filter by:").pack(pady=2)
        cf_dd = ttk.Combobox(
            self,
            textvariable=main_app.control_filter_var,
            state="readonly",
            width=20
        )
        cf_dd.pack(pady=2)

        # --------------------------------------------------------------------
        # Populate the combobox with teacher + student columns + special items
        # --------------------------------------------------------------------
        all_ctrl_cols = []
        # Add teacher columns
        if self.main_app.config.teacher_control_columns:
            all_ctrl_cols.extend(self.main_app.config.teacher_control_columns)
        # Add student columns
        if self.main_app.config.student_control_columns:
            all_ctrl_cols.extend(self.main_app.config.student_control_columns)
        # Remove duplicates (keep order)
        all_ctrl_cols = list(dict.fromkeys(all_ctrl_cols))

        # Add special filter options
        all_ctrl_cols += ["Absolute Error", "Relative Error", "Density"]

        cf_dd["values"] = all_ctrl_cols
        # If there's something, pick first as default
        if len(all_ctrl_cols) > 0:
            self.main_app.control_filter_var.set(all_ctrl_cols[0])

        row_cf = tk.Frame(self)
        row_cf.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(row_cf, text="Min:").pack(side=tk.LEFT, padx=2)
        tk.Entry(row_cf, textvariable=self.main_app.filter_min_var, width=8).pack(side=tk.LEFT, padx=2)

        tk.Label(row_cf, text="Max:").pack(side=tk.LEFT, padx=2)
        tk.Entry(row_cf, textvariable=self.main_app.filter_max_var, width=8).pack(side=tk.LEFT, padx=2)


class FeatureFilterFrame(tk.LabelFrame):
    """Frame for additional feature filter (dropdown + min/max)."""
    def __init__(self, parent, main_app, **kwargs):
        super().__init__(parent, text="Additional Feature Filter", **kwargs)
        self.main_app = main_app

        # First row - Combobox
        row1 = tk.Frame(self)
        row1.pack(side=tk.TOP, fill=tk.X, pady=2)

        tk.Label(row1, text="Feature:").pack(side=tk.LEFT, padx=2)
        feat_dd = ttk.Combobox(
            row1,
            textvariable=main_app.feature_filter_var,
            state="readonly",
            width=12
        )
        feat_dd.pack(side=tk.LEFT, padx=2)

        all_cols = list(self.main_app.df.columns)
        feat_dd["values"] = all_cols
        if len(all_cols) > 0:
            self.main_app.feature_filter_var.set(all_cols[0])

        # Second row - Min and Max entries
        row2 = tk.Frame(self)
        row2.pack(side=tk.TOP, fill=tk.X, pady=2)

        tk.Label(row2, text="Min:").pack(side=tk.LEFT, padx=2)
        tk.Entry(
            row2,
            textvariable=self.main_app.feature_filter_min_var,
            width=4
        ).pack(side=tk.LEFT, padx=2)

        tk.Label(row2, text="Max:").pack(side=tk.LEFT, padx=2)
        tk.Entry(
            row2,
            textvariable=self.main_app.feature_filter_max_var,
            width=4
        ).pack(side=tk.LEFT, padx=2)


class DensitySettingsFrame(tk.LabelFrame):
    """Frame for density bin settings + log scale check."""
    def __init__(self, parent, main_app, **kwargs):
        super().__init__(parent, text="Density Settings", **kwargs)
        self.main_app = main_app

        row_density = tk.Frame(self)
        row_density.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(row_density, text="Bins:").pack(side=tk.LEFT, padx=2)
        tk.Entry(row_density, textvariable=main_app.density_bins_var, width=6).pack(side=tk.LEFT, padx=2)

        self.log_scale_check = tk.Checkbutton(
            self,
            text="Log Scale",
            variable=main_app.log_scale_var
        )
        self.log_scale_check.pack(pady=2)


class StepsRemovalFrame(tk.Frame):
    """Frame for removing N steps from start of each experiment."""
    def __init__(self, parent, main_app, **kwargs):
        super().__init__(parent, **kwargs)
        self.main_app = main_app

        tk.Label(self, text="Remove N steps:").pack(side=tk.LEFT, padx=2)
        tk.Entry(self, textvariable=self.main_app.n_steps_var, width=8).pack(side=tk.LEFT, padx=2)


class ErrorSettingsFrame(tk.LabelFrame):
    """Frame for setting relative error cap."""
    def __init__(self, parent, main_app, **kwargs):
        super().__init__(parent, text="Error Settings", **kwargs)
        self.main_app = main_app

        tk.Label(self, text="Relative Error Cap:").pack(side=tk.LEFT, padx=2)
        tk.Entry(self, textvariable=self.main_app.error_cap_var, width=8).pack(side=tk.LEFT, padx=2)

        apply_cap_button = tk.Button(self, text="Apply", command=self.main_app.apply_error_cap)
        apply_cap_button.pack(side=tk.LEFT, padx=2)
