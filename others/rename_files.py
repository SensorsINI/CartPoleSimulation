import os

STARTSWITH = "Experiment"
ENDSWITH   = ".csv"


def collect_matching_files(root, recursive=False):
    # Return a sorted list of absolute paths matching our pattern
    if recursive:
        file_paths = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.startswith(STARTSWITH) and f.endswith(ENDSWITH):
                    file_paths.append(os.path.join(dirpath, f))
        file_paths.sort()
        return file_paths

    # non-recursive
    return sorted(
        os.path.join(root, f)
        for f in os.listdir(root)
        if f.startswith(STARTSWITH) and f.endswith(ENDSWITH)
    )

def staged_rename(paths):
    # Step 1: Rename all target files to temporary unique names
    temp_paths = []
    for i, old_path in enumerate(paths, start=1):
        dirpath   = os.path.dirname(old_path)
        temp_name = f"temp_rename_{i:04d}.tmp"
        temp_path = os.path.join(dirpath, temp_name)
        print(f"Renaming {old_path} -> {temp_path}")
        os.rename(old_path, temp_path)
        temp_paths.append(temp_path)

    # Step 2: Rename from temporary names to final desired names
    for i, temp_path in enumerate(temp_paths, start=1):
        dirpath  = os.path.dirname(temp_path)
        new_name = f"{STARTSWITH}-{i:03d}{ENDSWITH}"
        new_path = os.path.join(dirpath, new_name)
        print(f"Renaming {temp_path} -> {new_path}")
        os.rename(temp_path, new_path)

def rename_files_preserve_order(directory, recursive=False):
    startswith = STARTSWITH
    endswith   = ENDSWITH

    # Gather all matching files (flat or recursive)
    file_paths = collect_matching_files(directory, recursive)
    if not file_paths:
        mode = "recursively " if recursive else ""
        print(f"No files {mode}starting with {startswith} and ending with {endswith} found in {directory}.")
        return

    # Debug: Print the list of files to be renamed
    print("Files to be renamed:")
    for path in file_paths:
        print(path)

    staged_rename(file_paths)
    print("Renaming completed successfully.")

# Example usage
if __name__ == "__main__":
    target_directory = "./SI_Toolkit_ASF/Experiments/Data_5_05_2025_sets_1_2"
    RECURSIVE = True

    # Verify the directory exists
    if not os.path.isdir(target_directory):
        print(f"Error: The directory '{target_directory}' does not exist.")
    else:
        rename_files_preserve_order(target_directory, recursive=RECURSIVE)
