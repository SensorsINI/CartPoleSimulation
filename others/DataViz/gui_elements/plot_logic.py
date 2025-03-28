# gui_elements/plot_logic.py

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from scipy.stats import binned_statistic_2d
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from tkinter import messagebox


def update_plot(main_app):
    """
    Your big method that re-draws the plot.
    Takes 'main_app' as an argument so we can access main_app.df,
    main_app.x_var, main_app.ax, etc.
    """
    main_app.figure.clf()
    main_app.ax = main_app.figure.add_subplot(111)

    df_plot = main_app.original_df.copy()
    # Apply N-step removal
    n = main_app.n_steps_var.get()
    df_plot = df_plot.groupby("experiment_id", group_keys=False).apply(lambda g: g.iloc[n:])

    # Possibly normalize
    if main_app.normalize_var.get():
        df_plot = apply_normalization(main_app, df_plot)
        df_plot = main_app.processor.compute_errors(df_plot)

    plot_option = main_app.plot_var.get()

    # Decide OLS vs. PLS vs. normal XY
    if main_app.use_ols_var.get():
        x_col, y_col = force_teacher_student_xy(main_app, df_plot)
        plot_type = "Scatter"
        do_ols = True
    elif main_app.use_pls_var.get():
        pls_target = _get_pls_target_column(main_app, plot_option, df_plot)
        if pls_target is None:
            main_app.ax.set_title("Could not compute PLS target.")
            main_app.canvas.draw_idle()
            return
        df_plot["__target_for_pls"] = pls_target
        _compute_pls_components(main_app, df_plot)
        x_col, y_col = "PLS_1", "PLS_2"
        plot_type = main_app.plot_type_var.get()
        do_ols = False
    else:
        x_col = main_app.x_var.get()
        y_col = main_app.y_var.get()
        plot_type = main_app.plot_type_var.get()
        do_ols = False

    # Check columns exist
    if x_col not in df_plot.columns or y_col not in df_plot.columns:
        main_app.ax.set_title(f"X or Y not found: {x_col}, {y_col}")
        main_app.canvas.draw_idle()
        return

    # Filter user-specified control or feature columns
    df_filtered = _apply_all_filters(main_app, df_plot, x_col, y_col)
    if len(df_filtered) == 0:
        main_app.ax.set_title("No data after filtering")
        main_app.canvas.draw_idle()
        return

    x_vals = df_filtered[x_col].values
    y_vals = df_filtered[y_col].values

    # Decide coloring
    bins = _parse_bins(main_app, main_app.density_bins_var.get())
    norm = mcolors.LogNorm() if main_app.log_scale_var.get() else None

    c_data = None
    if plot_option != "Density":
        c_data = _get_cdata_for_plot_option(main_app, plot_option, df_filtered)

    if main_app.color_by_cluster_var.get():
        if "cluster_label" in df_filtered.columns:
            c_data = df_filtered["cluster_label"].values
        else:
            print("Warning: color_by_cluster is True but no 'cluster_label' in df_filtered.")

    # If user wants color by weights, but not cluster
    elif main_app.color_by_weight_var.get() and "weights" in main_app.df.columns:
        c_data = main_app.df["weights"].values

    # Scatter vs. Heatmap
    if plot_type == "Scatter":
        if (plot_option == "Density") and not main_app.color_by_cluster_var.get() \
                                       and not main_app.color_by_weight_var.get():
            _plot_density_scatter(main_app, x_vals, y_vals, bins, norm)
        else:
            if c_data is not None:
                sc = main_app.ax.scatter(
                    x_vals, y_vals,
                    c=c_data,
                    cmap="viridis",  # you can choose a discrete colormap if you prefer
                    s=5,
                    norm=norm
                )
                main_app.figure.colorbar(
                    sc, ax=main_app.ax,
                    label=_choose_colorbar_label(main_app, plot_option)
                )
        if do_ols:
            _plot_ols_line(main_app, x_vals, y_vals)

    else:  # "Heatmap"
        if (plot_option == "Density") and not main_app.color_by_weight_var.get():
            _plot_density_heatmap(main_app, x_vals, y_vals, bins, norm)
        else:
            if c_data is not None:
                stat, x_edges, y_edges, _ = binned_statistic_2d(
                    x_vals, y_vals, c_data, statistic='mean', bins=bins
                )
                Xmesh, Ymesh = np.meshgrid(x_edges, y_edges)
                im = main_app.ax.pcolormesh(
                    Xmesh, Ymesh, stat.T,
                    cmap='viridis', shading='auto', norm=norm
                )
                main_app.figure.colorbar(
                    im, ax=main_app.ax,
                    label=_choose_colorbar_label(main_app, plot_option, heatmap=True)
                )

    # Title, axes
    main_app.ax.set_xlabel(x_col)
    main_app.ax.set_ylabel(y_col)
    title_str = "Teacher vs. Student (OLS)" if do_ols else f"{plot_option} ({plot_type})"
    main_app.ax.set_title(title_str)

    # Possibly draw cluster boundaries
    if main_app.show_main_clusters_var.get() and main_app.boundaries:
        for lbl, poly in main_app.boundaries.items():
            exterior = getattr(poly, "exterior", None)
            if exterior:
                xp, yp = exterior.xy
                main_app.ax.plot(xp, yp, color="red", linewidth=1.5)

    main_app.canvas.draw_idle()


# ------------------------------------------------------------------------------
# Normalization, BFS weighting, and other helper functions
# ------------------------------------------------------------------------------

def apply_normalization(main_app, df):
    """
    Moved from main_app._apply_normalization => a separate function
    """
    cols_to_scale = set()
    if main_app.config.state_columns:
        cols_to_scale.update(main_app.config.state_columns)
    if main_app.config.teacher_control_columns:
        cols_to_scale.update(main_app.config.teacher_control_columns)
    if main_app.config.student_control_columns:
        cols_to_scale.update(main_app.config.student_control_columns)

    cols_to_scale = [c for c in cols_to_scale if c in df.columns]
    if not cols_to_scale:
        return df

    if main_app.config.norm_file_path is not None:
        try:
            norm_df = pd.read_csv(main_app.config.norm_file_path, comment="#", index_col=0)
            if "min" not in norm_df.index or "max" not in norm_df.index:
                messagebox.showwarning(
                    "Normalization File Error",
                    f"Expected rows named 'min' and 'max' in {main_app.config.norm_file_path}"
                )
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
        if not main_app.already_warned_about_local_norm:
            messagebox.showwarning("Normalization Warning",
                                   "No norm file found. Using dataset-based min/max for scaling.")
            main_app.already_warned_about_local_norm = True
        for c in cols_to_scale:
            cmin = df[c].min()
            cmax = df[c].max()
            rng = cmax - cmin
            df[c] = 0.0 if abs(rng) < 1e-12 else 2*(df[c] - cmin)/rng - 1
    return df

def force_teacher_student_xy(main_app, df):
    """
    Moved from main_app._force_teacher_student_xy => a separate function
    """
    t = main_app.teacher_col_var.get()
    s = main_app.student_col_var.get()
    if main_app.clip_control_var.get() and f"{s}_clipped" in df.columns:
        y_col = f"{s}_clipped"
    else:
        y_col = s
    return t, y_col

# Apply filters
def _apply_all_filters(main_app, df, x_col, y_col):
    """
    Moved from main_app._apply_all_filters => a private function
    """
    df_filtered = df.copy()

    # Control filter
    choice = main_app.control_filter_var.get()
    col_to_filter = _map_control_filter_to_column(main_app, choice, df_filtered, x_col, y_col)
    cf_min, cf_max = _parse_range(main_app, main_app.filter_min_var.get(), main_app.filter_max_var.get())
    if col_to_filter and cf_min is not None and cf_max is not None and cf_min < cf_max:
        df_filtered = df_filtered[
            (df_filtered[col_to_filter] >= cf_min) & (df_filtered[col_to_filter] <= cf_max)
        ]

    # Feature filter
    feat_col = main_app.feature_filter_var.get()
    if feat_col in df_filtered.columns:
        f_min, f_max = _parse_range(main_app, main_app.feature_filter_min_var.get(),
                                    main_app.feature_filter_max_var.get())
        if f_min is not None and f_max is not None and f_min < f_max:
            df_filtered = df_filtered[
                (df_filtered[feat_col] >= f_min) & (df_filtered[feat_col] <= f_max)
            ]

    return df_filtered

def _map_control_filter_to_column(main_app, choice, df, x_col, y_col):
    """
    Moved from main_app._map_control_filter_to_column => a private function
    """
    if choice in df.columns:
        return choice
    elif choice == "Absolute Error":
        s = main_app.student_col_var.get()
        return f"{s}_abs_err_clipped" if main_app.clip_control_var.get() else f"{s}_abs_err"
    elif choice == "Relative Error":
        s = main_app.student_col_var.get()
        return f"{s}_rel_err_clipped" if main_app.clip_control_var.get() else f"{s}_rel_err"
    elif choice == "Density":
        bins = _parse_bins(main_app, main_app.density_bins_var.get())
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

# PLS
def _get_pls_target_column(main_app, plot_option, df):
    return _get_cdata_for_plot_option(main_app, plot_option, df, for_pls=True)

def _compute_pls_components(main_app, df):
    x_cols = [col for col in main_app.config.state_columns if col in df.columns]
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

# OLS line
def _plot_ols_line(main_app, x_data, y_data):
    if len(x_data) < 2:
        return
    X_reshaped = x_data.reshape(-1, 1)
    model = LinearRegression().fit(X_reshaped, y_data)
    slope = model.coef_[0]
    intercept = model.intercept_
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = slope * x_fit + intercept
    main_app.ax.plot(x_fit, y_fit, color="red",
                     label=f"OLS: slope={slope:.3f}, intercept={intercept:.3f}")
    main_app.ax.legend()

# Plot densities
def _plot_density_scatter(main_app, x_vals, y_vals, bins, norm):
    hist, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=bins)
    c_data = np.zeros_like(x_vals, dtype=float)
    x_bin_idx = np.digitize(x_vals, xedges) - 1
    y_bin_idx = np.digitize(y_vals, yedges) - 1
    x_bin_idx = np.clip(x_bin_idx, 0, bins - 1)
    y_bin_idx = np.clip(y_bin_idx, 0, bins - 1)
    for i in range(len(x_vals)):
        c_data[i] = hist[x_bin_idx[i], y_bin_idx[i]]
    sc = main_app.ax.scatter(x_vals, y_vals, c=c_data, cmap="viridis", s=5, norm=norm)
    main_app.figure.colorbar(sc, ax=main_app.ax, label="Density (bin count)")

def _plot_density_heatmap(main_app, x_vals, y_vals, bins, norm):
    counts, xedges, yedges, im = main_app.ax.hist2d(
        x_vals, y_vals, bins=bins, cmap="viridis", norm=norm
    )
    main_app.figure.colorbar(im, ax=main_app.ax, label="Density (bin count)")

# cdata for color dimension
def _get_cdata_for_plot_option(main_app, option, df, for_pls=False):
    def get_scol_clipped(scol, dframe):
        if main_app.clip_control_var.get():
            clipped = f"{scol}_clipped"
            if clipped in dframe.columns:
                return dframe[clipped].values
            else:
                return dframe[scol].clip(-1,1).values
        else:
            return dframe[scol].values

    if option == "Teacher Control":
        t_col = main_app.teacher_col_var.get()
        return df[t_col].values if t_col in df.columns else None
    elif option == "Student Control":
        s_col = main_app.student_col_var.get()
        if s_col not in df.columns:
            return None
        return get_scol_clipped(s_col, df)
    elif option == "Absolute Error":
        s_col = main_app.student_col_var.get()
        col = f"{s_col}_abs_err_clipped" if main_app.clip_control_var.get() else f"{s_col}_abs_err"
        return df[col].values if col in df.columns else None
    elif option == "Relative Error":
        s_col = main_app.student_col_var.get()
        col = f"{s_col}_rel_err_clipped" if main_app.clip_control_var.get() else f"{s_col}_rel_err"
        return df[col].values if col in df.columns else None
    elif option == "Density":
        if len(main_app.config.state_columns) < 2:
            return None
        col1, col2 = main_app.config.state_columns[:2]
        if col1 not in df.columns or col2 not in df.columns:
            return None
        bins = _parse_bins(main_app, main_app.density_bins_var.get())
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

def _parse_bins(main_app, bins_str):
    """Used to parse the density_bins_var from the user"""
    try:
        val = int(bins_str)
        return val if val > 0 else 50
    except ValueError:
        return 50

def _parse_range(main_app, s_min, s_max):
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

def _choose_colorbar_label(main_app, plot_option, heatmap=False):
    if main_app.color_by_cluster_var.get():
        return "Cluster Label"
    elif main_app.color_by_weight_var.get():
        return "Weight"
    else:
        if heatmap:
            return f"{plot_option} (avg)"
        else:
            return plot_option
