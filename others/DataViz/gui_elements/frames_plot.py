# gui_elements/frames_plot.py

import tkinter as tk
from tkinter import ttk


class PlotOptionsFrame(tk.LabelFrame):
    """Dropdown for 'Teacher Control', 'Student Control', 'Absolute Error', etc."""
    def __init__(self, parent, main_app, **kwargs):
        super().__init__(parent, text="Plot Options", **kwargs)
        self.main_app = main_app

        tk.Label(self, text="Plot Option:").grid(row=0, column=0, padx=2, pady=2, sticky="w")

        plot_dd = ttk.Combobox(
            self,
            textvariable=main_app.plot_var,
            values=["Teacher Control", "Student Control", "Absolute Error", "Relative Error", "Density"],
            state="readonly",
            width=16
        )
        plot_dd.grid(row=0, column=1, padx=2, pady=2)


class PlotTypeFrame(tk.LabelFrame):
    """Dropdown for Scatter vs. Heatmap, plus Clip Student check."""
    def __init__(self, parent, main_app, **kwargs):
        super().__init__(parent, text="Plot Type", **kwargs)
        self.main_app = main_app

        tk.Label(self, text="Plot Type:").grid(row=0, column=0, padx=2, pady=2, sticky="w")

        pt_dd = ttk.Combobox(
            self,
            textvariable=main_app.plot_type_var,
            values=["Scatter", "Heatmap"],
            state="readonly",
            width=16
        )
        pt_dd.grid(row=0, column=1, padx=2, pady=2)

        self.clip_control_check = tk.Checkbutton(
            self,
            text="Clip Student [-1,1]",
            variable=main_app.clip_control_var
        )
        self.clip_control_check.grid(row=1, column=0, columnspan=2, padx=2, pady=2, sticky="w")
