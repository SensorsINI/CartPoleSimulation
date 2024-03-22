import os
import pandas as pd
import numpy as np
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles
from CartPole.cartpole_parameters import TrackHalfLength

# Settings
encoder_precision = 2*TrackHalfLength/4705.0
ADC_precision = (1.0 / 4096.0)
measurement_interval = 1.0e-3  # s
add_noise = 1  # +/- quantised units
path_to_data = './SI_Toolkit_ASF/'
quantized_folder = path_to_data + '_quantized'

features_precision_dict = {
    'angle': ADC_precision,
    'angleD': ADC_precision / measurement_interval,
    'position': encoder_precision,
    'positionD': encoder_precision / measurement_interval,
}

# Ensure the quantized directory exists
if not os.path.exists(quantized_folder):
    os.makedirs(quantized_folder)

paths_to_datafiles = get_paths_to_datafiles(path_to_data)
dfs = load_data(paths_to_datafiles)


def quantize_with_noise(value, quantization_value):
    noise_options = range(-add_noise, add_noise + 1)
    noise = np.random.choice(noise_options) * quantization_value
    return np.round(value / quantization_value) * quantization_value + noise


for path, df in zip(paths_to_datafiles, dfs):
    for key, quantization_value in features_precision_dict.items():
        if key in df.columns:
            df[key] = df[key].apply(quantize_with_noise, args=(quantization_value,))

    new_path = os.path.join(quantized_folder, os.path.relpath(path, start=path_to_data))
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    with open(path, 'r') as original_file, open(new_path, 'w') as new_file:
        # Copy over the comments from the original file
        for line in original_file:
            if line.startswith('#'):
                new_file.write(line)
            else:
                break  # Stop reading after comments
        # Add a custom line indicating this file is a quantized version
        new_file.write('# This is a quantized version of the original file with added noise\n')

        # Now write the modified DataFrame
        df.to_csv(new_file, index=False, line_terminator='\n')

