import os
from SI_Toolkit.load_and_normalize import get_paths_to_datafiles, load_data
import numpy as np

path_to_folder = 'SI_Toolkit_ASF/Experiments/Pretrained-RNN-1-Derivative/Recordings/Train'

paths_to_recordings = get_paths_to_datafiles(path_to_folder)

dfs = load_data(list_of_paths_to_datafiles=paths_to_recordings)

corrupted_files = []
corrupted_files_indices = []

for i in range(len(dfs)):
    df = dfs[i]
    D_angleD = df['D_angleD']
    if np.any(D_angleD < -50):
        corrupted_files.append(paths_to_recordings[i])
        corrupted_files_indices.append(i)
    # if np.all(abs(angle[len(angle)//2:])>np.pi/2.0):
    #     corrupted_files.append(paths_to_recordings[i])

print('__________________________________________________________________________')
print('Found corrupted files. The paths are:')
for file in corrupted_files:
    print(file)
print(corrupted_files)

print()
idx = corrupted_files_indices[0]
file_name = corrupted_files[0]
df = dfs[idx]
D_angleD = df['D_angleD']
indices_corrupted = np.where(D_angleD < -50)
print('File {} there is corrupted at line {}'.format(file_name, indices_corrupted))

print('Altogether there are {} corrupted files'.format(len(corrupted_files)))

# Delete corrupted files
for file in corrupted_files:
    os.remove(file)
print('Removed all corrupted_files')