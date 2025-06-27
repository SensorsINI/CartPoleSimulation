from types import SimpleNamespace
import csv

import os
import glob
import pandas as pd

def get_full_paths_to_csvs(default_locations='', csv_names=None):
    """
    This super cool function takes as the argument
    default locations where the csv files are normally find
    and
    the names of csv files.
    It returns the list of absolute paths to the csv files.

    It is probably mostly useful for loading single csv file at a variable location.

    Both the default_locations and csv_names are graciously allowed to be a string
    if there is only one default location respectively one csv file.
    But they can always be list of strings with one or more elements.

    csv_names can be either the name of a file (with or without ".csv" suffix - isn't it delightful?)
    or the absolute or relative path to it.
    csv_names is None, '' or [] the path to the most recent files over default locations will be returned

    If there are two csv with name = csv_names at two places listed as default_location,
    and only name, not the path is specified,
    the exception will be raised notifying user about the problem. I find it also fantastic.
    """

    file_paths = []

    # If not already a list pack default location into a list
    if not isinstance(default_locations, list):
        default_locations = [default_locations]

    # If empty, load the most recent file from ANY(!) of the default locations
    if csv_names is None or csv_names == '' or csv_names == []:
        if default_locations[0] != [] and (default_locations[0] is not None):
            # get the latest file from the default location
            try:
                list_of_files = []
                for default_location in default_locations:
                    list_of_files.extend(glob.glob(default_location + '/*.csv'))
                file_paths = [max(list_of_files, key=os.path.getctime)]
            except FileNotFoundError:
                print('Cannot load: No experiment recording found in data folders: {}'.format(default_locations))
        else:
            raise Exception('Cannot load: Tried loading most recent recording, but no default locations specified')

    else:
        # If not already a list csv_names location into a list
        if not isinstance(csv_names, list):
            csv_names = [csv_names]

        for filename in csv_names:

            if filename[-4:] != '.csv':
                filename += '.csv'

            # check if file found in DATA_FOLDER_NAME or at local starting point
            if os.path.isfile(filename):
                file_path = [filename]
            elif default_locations is None or default_locations == [] or default_locations == '':
                file_path = []
                print(
                    'Cannot load: There is no experiment recording file with name {} in root and no default location is specified'.format(
                        filename, default_locations))
            else:
                file_path = []
                one_file_already_found = False
                for default_location in default_locations:
                    file_path_trial = os.path.join(default_location, filename)
                    if os.path.isfile(file_path_trial):
                        if one_file_already_found:
                            raise Exception('There is more than one csv file with specified name at default location')
                        file_path.append(file_path_trial)
                        one_file_already_found = True
                if not file_path:
                    print(
                        'Cannot load: There is no experiment recording file with name {} at local folder or in {}'.format(
                            filename, default_locations))

            file_paths.extend(file_path)

    return file_paths


# load csv file with experiment recording (e.g. for replay)
def load_csv_recording(file_path):
    if isinstance(file_path, list):
        file_path = file_path[0]

    # Get race recording
    print('Loading file {}'.format(file_path))
    try:
        data: pd.DataFrame = pd.read_csv(file_path, comment='#')  # skip comment lines starting with #
    except Exception as e:
        print('Cannot load: Caught {} trying to read CSV file {}'.format(e, file_path))
        return False

    # Change to float32 wherever numeric column
    cols = data.columns
    data[cols] = data[cols].apply(pd.to_numeric, errors='ignore', downcast='float')

    return data

def load_cartpole_parameters(dataset_path):
    p = SimpleNamespace()
    line_count = 0
    # region Get information about the pretrained network from the associated txt file
    with open(dataset_path, newline='') as f:
        reader = csv.reader(f)
        updated_features = 0
        for line in reader:
            line = line[0]
            line_count += 1
            if line[:len('# m: ')] == '# m: ':
                p.m = float(line[len('# m: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# M: ')] == '# M: ':
                p.M = float(line[len('# M: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# L: ')] == '# L: ':
                p.L = float(line[len('# L: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# u_max: ')] == '# u_max: ':
                p.u_max = float(line[len('# u_max: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# M_fric: ')] == '# M_fric: ':
                p.M_fric = float(line[len('# M_fric: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# J_fric: ')] == '# J_fric: ':
                p.J_fric = float(line[len('# J_fric: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# v_max: ')] == '# v_max: ':
                p.v_max = float(line[len('# v_max: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# TrackHalfLength: ')] == '# TrackHalfLength: ':
                p.TrackHalfLength = float(line[len('# TrackHalfLength: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# controlNoise: ')] == '# controlNoise: ':
                p.controlNoise = float(line[len('# controlNoise: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# controlNoiseCorrelation: ')] == '# controlNoiseCorrelation: ':
                p.controlNoiseCorrelation = float(line[len('# controlNoiseCorrelation: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# g: ')] == '# g: ':
                p.g = float(line[len('# g: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# k: ')] == '# k: ':
                p.k = float(line[len('# k: '):].rstrip("\n"))
                updated_features += 1
                continue

            if line_count >= 25:  # Avoid going through the whole csv file
                break

    return p
