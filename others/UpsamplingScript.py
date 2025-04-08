import matplotlib

matplotlib.use('MacOSX')

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import CubicSpline
import os
import csv

MODE = 2  # 1 and 2 for testing, 0 for processing the data.
# -------------------------------------------------------------------------
# Process a SINGLE CSV file, upsampling specified columns,
# then optionally display the result in a plot (similar style to MODE 1).
# Each column_to_interpolate will get its own figure.
# -------------------------------------------------------------------------
input_file = "../MPCswingups/CPP_swing_up-0.csv"  # CHANGE to your input CSV
output_file = "example_single_upsampled.csv"  # Where to write upsampled result
time_column = "time"  # CHANGE to your actual time column name
columns_to_interpolate = ["angle", "angle_sin", "angle_cos"]  # CHANGE to actual columns to interpolate


def wavelet_denoise(data, wavelet='db4', threshold_method='hard', threshold_scale=0.7):
    """
    Denoises a time series using wavelet thresholding.
    """
    coeff = pywt.wavedec(data, wavelet)
    sigma = np.median(np.abs(coeff[-1])) / 0.6745  # Robust noise estimation using MAD
    uthresh = threshold_scale * sigma * np.sqrt(2 * np.log(len(data)))  # Scaled universal threshold
    denoised_coeff = [coeff[0]]
    for detail in coeff[1:]:
        denoised_detail = pywt.threshold(detail, value=uthresh, mode=threshold_method)
        denoised_coeff.append(denoised_detail)
    denoised_signal = pywt.waverec(denoised_coeff, wavelet)
    return denoised_signal


def upsample_signal(time, signal, factor=2):
    """
    Upsamples the input signal by a given factor using cubic spline interpolation.
    """
    new_length = len(time) * factor - (factor - 1)
    time_new = np.linspace(time[0], time[-1], new_length)
    cs = CubicSpline(time, signal)
    signal_new = cs(time_new)
    return time_new, signal_new


def process_single_csv(
        input_csv_path,
        output_csv_path,
        time_column,
        columns_to_interpolate,
        wavelet='db4',
        threshold_method='hard',
        threshold_scale=0.7,
        factor=2,
        plot_result=False
):
    """
    Process a single CSV file:
      1. Preserve leading comment lines (#).
      2. Read data; find the time column and specified columns to interpolate.
      3. Wavelet-denoise then upsample those columns by factor=2.
      4. For other columns, fill new rows with the previous row's value.
      5. Write result to output_csv_path (keeping comments).
      6. Optionally plot the results in the same style as MODE 1 (if plot_result=True).
    """
    # 1) Read lines and separate comments
    with open(input_csv_path, 'r', newline='') as f:
        lines = f.readlines()

    comment_lines = []
    data_lines = []
    for line in lines:
        if line.strip().startswith('#'):
            comment_lines.append(line.rstrip('\n'))
        else:
            data_lines.append(line.rstrip('\n'))

    # 2) Parse CSV header
    reader = csv.reader(data_lines)
    header = next(reader)  # first non-comment line is the header
    # Create a dictionary {column_name -> index_in_csv}
    col_index = {col_name: i for i, col_name in enumerate(header)}

    # Read data into a list of lists
    data_rows = [row for row in reader if len(row) == len(header)]

    # Convert columns to float arrays
    if time_column not in col_index:
        raise ValueError(f"Time column '{time_column}' not found in CSV header.")

    time_idx = col_index[time_column]
    time_data = np.array([float(r[time_idx]) for r in data_rows], dtype=float)

    signals_to_process = {}
    for col in columns_to_interpolate:
        if col not in col_index:
            raise ValueError(f"Column '{col}' not found in CSV header.")
        cidx = col_index[col]
        col_data = np.array([float(r[cidx]) for r in data_rows], dtype=float)
        signals_to_process[col] = col_data

    # 3) Wavelet-denoise each signal (for "angle", unwrap before denoising)
    denoised_signals = {}
    for col in columns_to_interpolate:
        if col == 'angle':
            d = wavelet_denoise(np.unwrap(signals_to_process[col]),
                                wavelet=wavelet,
                                threshold_method=threshold_method,
                                threshold_scale=threshold_scale)
        else:
            d = wavelet_denoise(signals_to_process[col],
                                wavelet=wavelet,
                                threshold_method=threshold_method,
                                threshold_scale=threshold_scale)
        # Truncate if wavelet reconstruction extends length.
        denoised_signals[col] = d[:len(time_data)]

    new_length = len(time_data) * factor - (factor - 1)
    # 3b) Upsample the time column based on first signal
    first_col = columns_to_interpolate[0]
    time_upsampled, _ = upsample_signal(time_data, denoised_signals[first_col], factor=factor)

    # 3c) Upsample signals for columns EXCLUDING "angle_sin" and "angle_cos"
    # We'll derive sin and cos from the upsampled angle.
    upsampled_signals = {}
    for col in columns_to_interpolate:
        if col in ['angle_sin', 'angle_cos']:
            # Skip these; they will be calculated later.
            continue
        # Get the upsampled signal using cubic spline interpolation.
        _, sig_upsampled = upsample_signal(time_data, denoised_signals[col], factor=factor)
        if col != 'angle':
            # For non-angle columns, preserve original points.
            augmented_col = np.copy(sig_upsampled)
            original_indices = np.arange(0, len(augmented_col), factor)
            augmented_col[original_indices] = signals_to_process[col]
            upsampled_signals[col] = augmented_col
        else:
            # For the angle column, do not overwrite with the original (wrapped) values.
            upsampled_signals[col] = sig_upsampled

    # Special treatment for angle interpolation:
    # Recompute the angle from its denoised and unwrapped version, then rewrap.
    if 'angle' in columns_to_interpolate:
        if 'angle' not in denoised_signals:
            raise ValueError("Column 'angle' not found in denoised signals.")
        # Upsample the unwrapped denoised angle.
        _, angle_interpolated = upsample_signal(time_data, denoised_signals['angle'], factor=factor)
        # Rewrap to the range [-π, π].
        wrapped_angle = (angle_interpolated + np.pi) % (2 * np.pi) - np.pi
        # Update the upsampled angle in the dictionary.
        upsampled_signals['angle'] = wrapped_angle
        # Now, derive sin and cos from the rewrapped upsampled angle.
        sin_from_angle = np.sin(wrapped_angle)
        cos_from_angle = np.cos(wrapped_angle)
        # Insert the derived values into the upsampled signals dictionary.
        upsampled_signals['angle_sin'] = sin_from_angle
        upsampled_signals['angle_cos'] = cos_from_angle

    # 4) Build the upsampled table for CSV output.
    upsampled_table = [[""] * len(header) for _ in range(new_length)]

    # Fill time column
    for i in range(new_length):
        upsampled_table[i][time_idx] = f"{time_upsampled[i]:.6f}"

    # Fill the columns we upsampled (including derived sin and cos from angle).
    for col in columns_to_interpolate:
        cidx = col_index[col]
        # Use the computed value from upsampled_signals if available.
        if col in upsampled_signals:
            col_vals = upsampled_signals[col]
            for i in range(new_length):
                upsampled_table[i][cidx] = f"{col_vals[i]:.6f}"
        else:
            # In case any column was not processed (should not occur)
            pass

    # 5) For columns not interpolated (and not time), replicate the previous row's value.
    for col, idx in col_index.items():
        if col == time_column or col in columns_to_interpolate:
            continue
        old_values = [r[idx] for r in data_rows]  # strings from original
        new_col_vals = [None] * new_length
        original_indices = np.arange(0, new_length, factor)
        for j, orig_idx in enumerate(original_indices):
            new_col_vals[orig_idx] = old_values[j]
        # Fill gaps.
        for i in range(new_length):
            if new_col_vals[i] is None:
                new_col_vals[i] = new_col_vals[i - 1]
        for i in range(new_length):
            upsampled_table[i][idx] = new_col_vals[i]

    # 6) Write result to output_csv_path.
    with open(output_csv_path, 'w', newline='') as fout:
        for cmt_line in comment_lines:
            fout.write(cmt_line + "\n")

        writer = csv.writer(fout)
        writer.writerow(header)
        for row in upsampled_table:
            writer.writerow(row)

    # 7) Optionally plot results (each column in its own figure).
    if plot_result:
        for col in columns_to_interpolate:
            if col not in upsampled_signals:
                continue
            augmented_col = upsampled_signals[col]
            original_indices = np.arange(0, len(augmented_col), factor)
            new_indices = np.setdiff1d(np.arange(len(augmented_col)), original_indices)

            plt.figure(figsize=(10, 5))
            # Plot full augmented curve in gray.
            plt.plot(time_upsampled, augmented_col, label="Augmented Signal",
                     color='gray', alpha=0.5)
            # Plot original points in blue.
            plt.scatter(time_upsampled[original_indices], augmented_col[original_indices],
                        color='blue', label="Original Points", s=20)
            # Plot new points in red.
            plt.scatter(time_upsampled[new_indices], augmented_col[new_indices],
                        color='red', label="New Points", s=10)
            plt.xlabel(time_column)
            plt.ylabel(col)
            plt.title(f"Column: {col} (Wavelet + Upsampling)")
            plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":

    if MODE == 1:
        # -------------------------------------------------------------------------
        # Original example from your code - unchanged.
        # -------------------------------------------------------------------------
        np.random.seed(42)  # For reproducibility.
        N = 1000  # Number of data points.
        t = np.linspace(0, 10, N)
        noise_level = 0.3
        original_signal = np.sin(2 * np.pi * t) + noise_level * np.random.randn(N)

        denoised_signal = wavelet_denoise(original_signal, threshold_method='hard', threshold_scale=0.9)
        denoised_signal = denoised_signal[:N]

        factor = 2
        t_upsampled, signal_upsampled = upsample_signal(t, denoised_signal, factor=factor)

        # Combine original points.
        augmented_signal = np.copy(signal_upsampled)
        original_indices = np.arange(0, len(t_upsampled), factor)
        augmented_signal[original_indices] = original_signal
        new_indices = np.setdiff1d(np.arange(len(t_upsampled)), original_indices)

        # Plot.
        plt.figure(figsize=(12, 6))
        plt.plot(t_upsampled, augmented_signal, label="Augmented Signal", color='gray', alpha=0.5)
        plt.scatter(t_upsampled[original_indices], augmented_signal[original_indices],
                    color='blue', label="Original Points", s=20)
        plt.scatter(t_upsampled[new_indices], augmented_signal[new_indices],
                    color='red', label="New Points", s=10)
        plt.xlabel("Time")
        plt.ylabel("Signal Value")
        plt.title("Augmented Dataset: Preserving Original Points, Augmenting with Interpolated Ones")
        plt.legend()
        plt.show()

    elif MODE == 2:
        # -------------------------------------------------------------------------
        # Process a single CSV file and plot in Mode 1 style (multiple columns, each in its own figure).
        # -------------------------------------------------------------------------
        process_single_csv(
            input_csv_path=input_file,
            output_csv_path=output_file,
            time_column=time_column,
            columns_to_interpolate=columns_to_interpolate,
            wavelet='db4',
            threshold_method='hard',
            threshold_scale=0.7,
            factor=2,
            plot_result=True  # Display final result.
        )

    elif MODE == 0:
        # -------------------------------------------------------------------------
        # Process ALL CSV files in a folder, write upsampled CSVs to target folder.
        # No plotting is done.
        # -------------------------------------------------------------------------
        source_folder = "source_data"  # CHANGE to your source folder
        target_folder = "target_data"  # CHANGE to your target folder
        os.makedirs(target_folder, exist_ok=True)

        time_column = "Time"  # CHANGE as needed
        columns_to_interpolate = ["Signal"]  # CHANGE as needed

        for file_name in os.listdir(source_folder):
            if file_name.lower().endswith(".csv"):
                input_path = os.path.join(source_folder, file_name)
                output_path = os.path.join(target_folder, file_name)

                process_single_csv(
                    input_csv_path=input_path,
                    output_csv_path=output_path,
                    time_column=time_column,
                    columns_to_interpolate=columns_to_interpolate,
                    wavelet='db4',
                    threshold_method='hard',
                    threshold_scale=0.7,
                    factor=2,
                    plot_result=False  # No plotting in batch mode.
                )
