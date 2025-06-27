import matplotlib
matplotlib.use('MacOSX')

import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import csv

from tqdm import tqdm

from scipy.interpolate import (
    CubicSpline,
    PchipInterpolator,
    interp1d
)

########################################################################
# User Configuration
########################################################################
MODE = 0  # 1 or 2 for testing, 0 for batch processing

# Path to your CSV
input_file = "./MPCswingups/CPP_swing_up-0.csv"
output_file = "./example_single_upsampled.csv"

time_column = "time"
columns_to_interpolate = [
    "angle", "angle_sin", "angle_cos",
    "angleD", "position", "positionD",
    "target_position", "Q", "Q_ccrc"
]

apply_denoising = False
desired_dt = 0.01  # new time step (e.g. 1 ms)

# Columns that are known to be step-like; we do zero-order hold interpolation
step_columns = {"target_position"}

########################################################################
# Wavelet Denoising
########################################################################
def wavelet_denoise(
    data,
    wavelet='db4',
    threshold_method='hard',
    threshold_scale=0.7
):
    """
    Denoises a time series using wavelet thresholding.
    """
    coeff = pywt.wavedec(data, wavelet)
    # Robust noise estimation using the median absolute deviation of last detail coeffs
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = threshold_scale * sigma * np.sqrt(2 * np.log(len(data)))

    denoised_coeff = [coeff[0]]
    for detail in coeff[1:]:
        denoised_detail = pywt.threshold(
            detail, value=uthresh, mode=threshold_method
        )
        denoised_coeff.append(denoised_detail)

    denoised_signal = pywt.waverec(denoised_coeff, wavelet)
    # Truncate if wavelet transform returns length off by 1
    return denoised_signal[:len(data)]


########################################################################
# General Interpolation Function (by dt) with Different Methods
########################################################################
def upsample_signal_by_dt(time, signal, dt, method="pchip"):
    """
    Resamples 'signal' at a new uniform time grid from time[0] to time[-1]
    in increments of dt, using one of several interpolation methods.

    :param time: 1D array of original time points (sorted).
    :param signal: 1D array of signal values (same length as time).
    :param dt: desired time step in the new uniform grid.
    :param method: one of {"pchip", "cubic_spline", "linear", "previous"}.
    :return: (time_new, signal_new)
    """
    t_min = time[0]
    t_max = time[-1]
    # Build new uniform time
    t_new = np.arange(t_min, t_max + 0.5*dt, dt)
    # Drop last point if it slightly exceeds t_max
    if t_new[-1] > t_max + 1e-12:
        t_new = t_new[:-1]

    # Choose interpolation approach
    if method == "cubic_spline":
        interpolator = CubicSpline(time, signal)
    elif method == "pchip":
        interpolator = PchipInterpolator(time, signal)
    elif method == "linear":
        interpolator = interp1d(
            time, signal, kind='linear', bounds_error=False,
            fill_value="extrapolate"
        )
    elif method == "previous":  # zero-order hold
        interpolator = interp1d(
            time, signal, kind='previous', bounds_error=False,
            fill_value=(signal[0], signal[-1])
        )
    else:
        raise ValueError(f"Unknown interpolation method '{method}'")

    signal_new = interpolator(t_new)
    return t_new, signal_new


########################################################################
# Main Processing Logic
########################################################################
def process_single_csv(
    input_csv_path,
    output_csv_path,
    time_column,
    columns_to_interpolate,
    desired_dt=0.001,
    wavelet='db4',
    threshold_method='hard',
    threshold_scale=0.7,
    plot_result=False,
    apply_denoising=True,
    step_columns=None
):
    """
    Process a single CSV to upsample specified columns at a uniform dt.
     1) Preserve comment lines (#).
     2) Read header/data. Identify time_column -> float array of times.
     3) [Optional] wavelet denoising on each column to be interpolated
        (angle is always unwrapped first).
     4) For each column in columns_to_interpolate:
        - pick an interpolation method:
            * "previous" if column is in step_columns
            * "cubic_spline" for 'angle'
            * else "pchip" by default
        - upsample to a new uniform time grid [time[0], time[-1]] with step = desired_dt
     5) If 'angle' in columns_to_interpolate, re-wrap to [-pi, pi], then derive angle_sin/cos
     6) Columns not in columns_to_interpolate: zero-order hold from old data to new time grid.
     7) Write out to a CSV with the same column order. Comments on top.
     8) [Optional] generate a plot for each upsampled column:
         * Original data: blue line + blue dots
         * Upsampled data: red line + red dots
    """
    if step_columns is None:
        step_columns = set()  # no step columns by default

    # 1) read lines, separate comment lines
    with open(input_csv_path, 'r', newline='') as f:
        lines = f.readlines()

    comment_lines = []
    data_lines = []
    for line in lines:
        if line.strip().startswith('#'):
            comment_lines.append(line.rstrip('\n'))
        else:
            data_lines.append(line.rstrip('\n'))

    # 2) parse CSV header
    reader = csv.reader(data_lines)
    header = next(reader)
    col_index = {c: i for i, c in enumerate(header)}

    if time_column not in col_index:
        raise ValueError(f"Time column '{time_column}' not found in header.")

    data_rows = [row for row in reader if len(row) == len(header)]
    time_idx = col_index[time_column]

    # float array for original time
    time_data = np.array([float(r[time_idx]) for r in data_rows], dtype=float)
    time_data = time_data - np.min(time_data)  # normalize time to start from 0

    # ensure sorted time
    if not np.all(time_data[1:] >= time_data[:-1]):
        sort_idx = np.argsort(time_data)
        time_data = time_data[sort_idx]
        data_rows = [data_rows[i] for i in sort_idx]

    # gather columns to process
    signals_to_process = {}
    for col in columns_to_interpolate:
        if col not in col_index:
            raise ValueError(f"Column '{col}' not found in header.")
        idx = col_index[col]
        col_data = np.array([float(r[idx]) for r in data_rows], dtype=float)
        signals_to_process[col] = col_data

    # 3) wavelet denoise if requested, unwrap angle first
    denoised_signals = {}
    for col in columns_to_interpolate:
        arr = signals_to_process[col]
        if col == 'angle':
            # unwrap angle
            arr = np.unwrap(arr)
        if apply_denoising and col not in ['angle_sin', 'angle_cos']:
            arr = wavelet_denoise(
                arr,
                wavelet=wavelet,
                threshold_method=threshold_method,
                threshold_scale=threshold_scale
            )
        denoised_signals[col] = arr

    # 4) build upsampled signals
    # pick a single uniform time grid from time_data[0]..time_data[-1]
    t_min, t_max = time_data[0], time_data[-1]
    time_new = np.arange(t_min, t_max + 0.5*desired_dt, desired_dt)
    if time_new[-1] > t_max + 1e-12:
        time_new = time_new[:-1]
    upsampled_signals = {}

    for col in columns_to_interpolate:
        if col in ('angle_sin', 'angle_cos'):
            # skip for now, we derive from angle later
            continue

        # choose interpolation method
        if col == 'angle':
            method = "cubic_spline"  # or "pchip", your choice
        elif col in step_columns:
            method = "previous"
        else:
            method = "pchip"

        # do the interpolation
        signal_old = denoised_signals[col]
        interpolator = None
        if method == "cubic_spline":
            interpolator = CubicSpline(time_data, signal_old)
        elif method == "pchip":
            interpolator = PchipInterpolator(time_data, signal_old)
        elif method == "linear":
            interpolator = interp1d(
                time_data, signal_old, kind='linear', fill_value="extrapolate"
            )
        elif method == "previous":
            interpolator = interp1d(
                time_data, signal_old, kind='previous', bounds_error=False,
                fill_value=(signal_old[0], signal_old[-1])
            )
        else:
            raise ValueError(f"Unknown method '{method}'")

        sig_new = interpolator(time_new)
        upsampled_signals[col] = sig_new

    # 5) angle re-wrap, then derive angle_sin/cos
    if 'angle' in columns_to_interpolate:
        angle_wrapped = (upsampled_signals['angle'] + np.pi) % (2*np.pi) - np.pi
        upsampled_signals['angle'] = angle_wrapped
        # derive sin, cos
        angle_sin = np.sin(angle_wrapped)
        angle_cos = np.cos(angle_wrapped)
        upsampled_signals['angle_sin'] = angle_sin
        upsampled_signals['angle_cos'] = angle_cos

    # 6) build final table
    new_length = len(time_new)
    upsampled_table = [[""] * len(header) for _ in range(new_length)]

    # fill new time column
    for i in range(new_length):
        upsampled_table[i][time_idx] = f"{time_new[i]:.6f}"

    # fill columns we upsampled
    for col in columns_to_interpolate:
        if col not in upsampled_signals:
            continue
        idx = col_index[col]
        svals = upsampled_signals[col]
        for i in range(new_length):
            upsampled_table[i][idx] = f"{svals[i]:.6f}"

    # for others, do zero-order hold from old data
    for col, idx in col_index.items():
        if col == time_column or col in columns_to_interpolate:
            continue
        old_values = [r[idx] for r in data_rows]  # strings
        # do zero-order hold by stepping through time_new
        new_col_vals = []
        i_old = 0
        for t_new_val in time_new:
            while (i_old < len(time_data) - 1) and (time_data[i_old+1] <= t_new_val):
                i_old += 1
            new_col_vals.append(old_values[i_old])

        for i in range(new_length):
            upsampled_table[i][idx] = new_col_vals[i]

    # 7) write CSV
    with open(output_csv_path, 'w', newline='') as fout:
        # comments
        for cmt in comment_lines:
            fout.write(cmt + "\n")
        writer = csv.writer(fout)
        writer.writerow(header)
        for row in upsampled_table:
            writer.writerow(row)

    # 8) Optionally plot each upsampled column
    if plot_result:
        for col in columns_to_interpolate:
            if col not in upsampled_signals:
                continue

            # original data
            original_y = signals_to_process[col]
            # final upsampled data
            upsampled_y = upsampled_signals[col]

            plt.figure(figsize=(10, 5))

            # Plot original as blue line + blue dots
            plt.plot(
                time_data, original_y, '-o',
                color='blue', linewidth=1, markersize=4,
                label='Original'
            )

            # Plot new as red line + red dots
            plt.plot(
                time_new, upsampled_y, '-o',
                color='red', linewidth=1, markersize=3,
                label='Upsampled'
            )

            plt.xlabel(time_column)
            plt.ylabel(col)
            if apply_denoising:
                title_str = f"{col} (Denoised + Upsampled, dt={desired_dt})"
            else:
                title_str = f"{col} (Upsampled, dt={desired_dt})"
            plt.title(title_str)
            plt.legend()
            plt.tight_layout()
            plt.show()


########################################################################
# MAIN
########################################################################
if __name__ == "__main__":
    if MODE == 1:
        # Example: small synthetic data
        N = 20
        time_data = np.linspace(0, 2*np.pi, N)
        # example step-like data
        signal_step = np.where(time_data < np.pi, 0.0, 1.0)
        # example smooth data
        signal_sin = np.sin(time_data)

        # put them in columns, write to a test CSV, then read from it
        import tempfile
        test_csv = "test_data.csv"
        with open(test_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "step_col", "sin_col"])
            for i in range(N):
                writer.writerow([
                    f"{time_data[i]:.6f}",
                    f"{signal_step[i]:.6f}",
                    f"{signal_sin[i]:.6f}"
                ])

        # process it
        process_single_csv(
            input_csv_path=test_csv,
            output_csv_path="test_data_upsampled.csv",
            time_column="time",
            columns_to_interpolate=["step_col", "sin_col"],
            desired_dt=0.1,
            plot_result=True,
            apply_denoising=False,
            step_columns={"step_col"}  # do zero-order hold for step_col
        )

    elif MODE == 2:
        # Process a real CSV with your desired dt
        process_single_csv(
            input_csv_path=input_file,
            output_csv_path=output_file,
            time_column=time_column,
            columns_to_interpolate=columns_to_interpolate,
            desired_dt=desired_dt,
            wavelet='db4',
            threshold_method='hard',
            threshold_scale=0.7,
            plot_result=True,          # show plots
            apply_denoising=apply_denoising,
            step_columns=step_columns  # columns that need step interpolation
        )

    elif MODE == 0:
        # Process ALL CSV in a folder, no plotting
        source_folder = "./MPCSwingUps_11_04_2025/"
        target_folder = "./MPCSwingUps_11_04_2025_100Hz/"
        os.makedirs(target_folder, exist_ok=True)

        for file_name in tqdm(os.listdir(source_folder)):
            if file_name.lower().endswith(".csv"):
                input_path = os.path.join(source_folder, file_name)
                output_path = os.path.join(target_folder, file_name)

                process_single_csv(
                    input_csv_path=input_path,
                    output_csv_path=output_path,
                    time_column=time_column,
                    columns_to_interpolate=columns_to_interpolate,
                    desired_dt=0.01,
                    wavelet='db4',
                    threshold_method='hard',
                    threshold_scale=0.7,
                    plot_result=False,
                    apply_denoising=apply_denoising,
                    step_columns=step_columns  # example
                )
