# gui_elements/frames_axes.py

import tkinter as tk
from tkinter import ttk


class AxesSelectionFrame(tk.LabelFrame):
    """
    A sub-frame to choose X and Y axes, plus checkboxes for PLS/OLS/Normalization.
    Populates x_dd and y_dd with state_columns from main_app.config.
    """
    def __init__(self, parent, main_app, **kwargs):
        super().__init__(parent, text="Axes Selection", **kwargs)
        self.main_app = main_app

        tk.Label(self, text="Select X-axis:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        x_dd = ttk.Combobox(
            self,
            textvariable=main_app.x_var,
            state="readonly",
            width=16
        )
        x_dd.grid(row=0, column=1, padx=2, pady=2)

        tk.Label(self, text="Select Y-axis:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        y_dd = ttk.Combobox(
            self,
            textvariable=main_app.y_var,
            state="readonly",
            width=16
        )
        y_dd.grid(row=1, column=1, padx=2, pady=2)

        # Populate the X/Y dropdowns from config.state_columns:
        state_cols = main_app.config.state_columns or []
        x_dd["values"] = state_cols
        y_dd["values"] = state_cols

        # If there are at least two columns, pick them by default
        if len(state_cols) >= 2:
            main_app.x_var.set(state_cols[0])
            main_app.y_var.set(state_cols[1])
        elif len(state_cols) == 1:
            # If there's only one, set both X and Y to that one (or leave Y blank)
            main_app.x_var.set(state_cols[0])
            main_app.y_var.set(state_cols[0])

        self.use_pls_check = tk.Checkbutton(
            self,
            text="Use PLS for X-Y (2D)",
            variable=main_app.use_pls_var
        )
        self.use_pls_check.grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky="w")

        self.use_ols_check = tk.Checkbutton(
            self,
            text="Use OLS (Teacher vs. Student)",
            variable=main_app.use_ols_var
        )
        self.use_ols_check.grid(row=3, column=0, columnspan=2, padx=2, pady=2, sticky="w")

        self.normalize_check = tk.Checkbutton(
            self,
            text="Normalize [-1..1]",
            variable=main_app.normalize_var
        )
        self.normalize_check.grid(row=4, column=0, columnspan=2, padx=2, pady=2, sticky="w")
