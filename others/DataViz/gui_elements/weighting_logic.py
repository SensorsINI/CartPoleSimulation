# gui_elements/weighting_logic.py

from tkinter import filedialog, messagebox
import pandas as pd

def on_coverage_changed(main_app, val):
    """Called when coverage slider changes."""
    main_app.weight_manager.main_cluster_coverage = float(val) / 100.0

def on_alpha_changed(main_app, val):
    """Called when alpha slider changes."""
    main_app.weight_manager.alpha_shape_alpha = float(val)

def on_export_weights(main_app):
    """Called when user exports the weighted CSV."""
    fname = filedialog.asksaveasfilename(
        defaultextension=".csv",
        initialfile="weighted_data.csv",
        title="Save Weighted CSV"
    )
    if fname:
        main_app.df.to_csv(fname, index=False)
        messagebox.showinfo("Saved", f"Weights saved to {fname}")
