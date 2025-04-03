import os
import shutil
import numpy as np

def split_csv_files(path_source,
                    path_target=None,
                    train_fraction=0.7,
                    validation_fraction=0.2,
                    random_seed=42,
                    files_to_log=None):
    """
    In `path_target` (equal to `path_source` if `path_target` is None),
    create folders: "All", "Train", "Validate", "Test", "Logs".

    Steps:
    1. If `path_source` does not exist, create it.
    2. Create sub-folders in `path_target`: "All", "Train", "Validate", "Test", "Logs".
    3. Gather all CSV files in `path_source` (non-recursively).
    4. Copy each CSV to `path_target/All` (but do NOT remove from `path_source` yet).
    5. Use NumPy to create a reproducible, exact split of the CSV list:
       - N_train = floor(len(CSVs) * train_fraction)
       - N_val   = floor(len(CSVs) * validation_fraction)
       - N_test  = remaining files
       Assign files to Train, Validate, and Test by randomly shuffling the indices.
    6. Copy files from "All" into the respective Train/Validate/Test sub-folders.
    7. ONLY NOW remove the CSV files from the `path_source`.
    8. For each file in `files_to_log`, copy it into `path_target/Logs` if it exists;
       otherwise, print a warning.
    9. The script uses `random_seed` for reproducible shuffling.

    :param path_source: Source folder containing CSV files (will be created if missing).
    :param path_target: Target folder (defaults to `path_source` if None).
    :param train_fraction: Fraction of CSV files to put into the "Train" folder.
    :param validation_fraction: Fraction of CSV files to put into the "Validate" folder.
    :param random_seed: Random seed for reproducible split.
    :param files_to_log: List of file paths to copy into the "Logs" folder.
    """

    if files_to_log is None:
        files_to_log = []

    # If path_target is not provided, use path_source
    if path_target is None:
        path_target = path_source

    # 1. Ensure path_source exists
    os.makedirs(path_source, exist_ok=True)

    # 2. Create the necessary sub-folders in path_target
    folders = ["All", "Train", "Validate", "Test", "Logs"]
    for folder in folders:
        os.makedirs(os.path.join(path_target, folder), exist_ok=True)

    # Gather CSV files in path_source (non-recursively)
    csv_files = [
        f for f in os.listdir(path_source)
        if f.lower().endswith(".csv") and os.path.isfile(os.path.join(path_source, f))
    ]

    # 3 & 4. Copy each CSV to "All" (do NOT remove from source yet)
    for csv_file in csv_files:
        src_path = os.path.join(path_source, csv_file)
        dest_all_path = os.path.join(path_target, "All", csv_file)
        shutil.copy2(src_path, dest_all_path)

    # 5. Create a reproducible, exact split using NumPy
    np.random.seed(random_seed)
    n_total = len(csv_files)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * validation_fraction)
    n_test = n_total - n_train - n_val

    # Safety check
    if n_test < 0:
        raise ValueError("Train fraction plus validation fraction cannot exceed 1.")

    # Shuffle indices
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # 6. Copy from "All" to Train/Validate/Test
    for i in train_indices:
        csv_file = csv_files[i]
        src = os.path.join(path_target, "All", csv_file)
        dst = os.path.join(path_target, "Train", csv_file)
        shutil.copy2(src, dst)

    for i in val_indices:
        csv_file = csv_files[i]
        src = os.path.join(path_target, "All", csv_file)
        dst = os.path.join(path_target, "Validate", csv_file)
        shutil.copy2(src, dst)

    for i in test_indices:
        csv_file = csv_files[i]
        src = os.path.join(path_target, "All", csv_file)
        dst = os.path.join(path_target, "Test", csv_file)
        shutil.copy2(src, dst)

    # 7. Copy log files, warning if missing
    for log_file in files_to_log:
        if os.path.isfile(log_file):
            shutil.copy2(log_file, os.path.join(path_target, "Logs", os.path.basename(log_file)))
        elif os.path.isdir(log_file):
            dest_dir = os.path.join(path_target, "Logs", os.path.basename(log_file))
            shutil.copytree(log_file, dest_dir)
        else:
            print(f"Warning: log file or directory '{log_file}' not found; skipping.")

    # 8. Now that everything is safely copied, remove CSV files from source
    for csv_file in csv_files:
        os.remove(os.path.join(path_source, csv_file))

if __name__ == "__main__":
    # Example usage:
    split_csv_files(
        path_source="Experiment_Recordings/",
        path_target="./SI_Toolkit_ASF/Experiments/Experiment_3_04_2025/Recordings",
        train_fraction=0.9,
        validation_fraction=0.05,
        random_seed=42,
        files_to_log=[
            "./cartpole_physical_parameters.yml",
            "./config_data_gen.yml",
            "./SI_Toolkit_ASF/config_predictors.yml",
            "./Control_Toolkit_ASF/config_controllers.yml",
            "./Control_Toolkit_ASF/config_cost_function.yml",
            "./Control_Toolkit_ASF/config_optimizers.yml",
            "./Control_Toolkit_ASF/Cost_Functions/CartPole",
        ]
    )
