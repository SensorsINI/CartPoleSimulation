import os

def rename_files_preserve_order(directory):
    # Get a sorted list of all files in the directory
    files = sorted(os.listdir(directory))

    # Filter files that match the pattern
    files_to_rename = [f for f in files if f.startswith("Experiment-") and f.endswith(".csv")]

    # Debug: Print the list of files to be renamed
    print("Files to be renamed:")
    for f in files_to_rename:
        print(f)
    
    # Step 1: Rename all target files to temporary unique names
    temp_names = []
    for i, old_name in enumerate(files_to_rename, start=1):
        old_path = os.path.join(directory, old_name)
        temp_name = f"temp_rename_{i:04d}.tmp"
        temp_path = os.path.join(directory, temp_name)
        print(f"Renaming {old_path} -> {temp_path}")
        os.rename(old_path, temp_path)
        temp_names.append(temp_name)

    # Step 2: Rename from temporary names to final desired names
    for i, temp_name in enumerate(temp_names, start=1):
        temp_path = os.path.join(directory, temp_name)
        new_name = f"Experiment-{i:03d}.csv"
        new_path = os.path.join(directory, new_name)
        print(f"Renaming {temp_path} -> {new_path}")
        os.rename(temp_path, new_path)

    print("Renaming completed successfully.")

# Example usage
if __name__ == "__main__":
    # Provide the absolute or relative path to your directory
    target_directory = "./Experiment_Recordings/"
    
    # Verify the directory exists
    if not os.path.isdir(target_directory):
        print(f"Error: The directory '{target_directory}' does not exist.")
    else:
        rename_files_preserve_order(target_directory)

