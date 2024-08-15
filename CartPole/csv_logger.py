import os
import csv

import numpy as np

from git import Repo
from datetime import datetime



def create_csv_file(csv_name, keys, path_to_experiment_recordings=None,
                    title='', header=[]):
    csv_filepath = _create_csv_file_path(csv_name, path_to_experiment_recordings)

    print('Saving to the file: {}'.format(csv_filepath))
    # Write the .csv file
    with open(csv_filepath, "a", newline='') as outfile:
        writer = csv.writer(outfile)

        writer.writerow(['# ' + title])

        repo = Repo(search_parent_directories=True)
        git_revision = repo.head.object.hexsha
        writer.writerow(['# ' + 'Done with git-revision: {}'
                        .format(git_revision)])

        writer.writerow(['#'])
        for line in header:
            writer.writerow(['# ' + line])
        writer.writerow(keys)

    return csv_filepath

def save_data_to_csv_file(csv_filepath, dict_history, rounding_decimals, mode='save online'):
    if mode == 'save online':

        # Save this dict
        with open(csv_filepath, "a", newline='') as outfile:
            writer = csv.writer(outfile)
            if rounding_decimals == np.inf:
                pass
            else:
                dict_history = {key: np.around(value, rounding_decimals)
                                     for key, value in dict_history.items()}
            writer.writerows(zip(*dict_history.values()))

    elif mode == 'save offline':
        # Round data to a set precision
        with open(csv_filepath, "a", newline='') as outfile:
            writer = csv.writer(outfile)
            if rounding_decimals == np.inf:
                pass
            else:
                dict_history = {key: np.around(value, rounding_decimals)
                                     for key, value in dict_history.items()}
            writer.writerows(zip(*dict_history.values()))
        # Another possibility to save data.
        # DF_history = pd.DataFrame.from_dict(dict_history).round(rounding_decimals)
        # DF_history.to_csv(self.csv_filepath, index=False, header=False, mode='a') # Mode (a)ppend


def _create_csv_file_path(csv_name, path_to_experiment_recordings=None):
    if path_to_experiment_recordings is not None:
        # Make folder to save data (if not yet existing)
        try:
            os.makedirs(path_to_experiment_recordings)
        except FileExistsError:
            pass

    # Check if the csv_name has .csv at the end otherwise add it
    if csv_name[-4:] != '.csv':
        csv_name += '.csv'

    csv_filepath = os.path.join(path_to_experiment_recordings, csv_name)

    # If file with a given name exists,
    # append increasing index to the end if the name until you find not used name (do not overwrite)
    file_index = 1
    filepath_new = csv_filepath
    while True:
        if os.path.isfile(filepath_new):
            filepath_new = csv_filepath[:-4]
        else:
            csv_filepath = filepath_new
            break
        filepath_new = filepath_new + '-' + str(file_index) + '.csv'
        file_index += 1

    csv_filepath = csv_filepath

    return csv_filepath

# Specific for Simulated Cartpole


def create_csv_file_name(controller='', controller_name='', optimizer_name='',
                         prefix='CPS', with_date=True, title=''):

    if with_date:
        date = str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S'))
    else:
        date = ''

    if controller_name == '':
        name_controller = ''
    elif controller is not None and hasattr(controller, "has_optimizer") and controller.has_optimizer:
        name_controller = '_' + controller_name + '_' + optimizer_name
    else:
        name_controller = '_' + controller_name

    if title != '':
        title = '_' + title

    csv_filename = prefix + title + name_controller + date + '.csv'

    return csv_filename


def create_csv_title():
    title = (f"This is CartPole simulation from {datetime.now().strftime('%d.%m.%Y')}" +
             f" at time {datetime.now().strftime('%H:%M:%S')}")
    return title


def create_csv_header(cps, length_of_experiment):
    header = [
        f"Length of experiment: {str(length_of_experiment)} s",

        f"",
        f"Time intervals dt:",
        f"Simulation: {str(cps.dt_simulation)} s",
        f"Controller update: {str(cps.dt_controller)} s",
        f"Saving: {str(cps.dt_save)} s",
        f"",

        f"Controller: {cps.controller_name}",
    ]

    if cps.optimizer_name:
        header.append(f"MPC Optimizer: {cps.optimizer_name}")

    header.append(
        f""
        f"Parameters:"
    )
    for param_name in cps.cpe.params.__dict__:
        parameter = getattr(cps.cpe.params, param_name)
        if isinstance(parameter, dict):
            dict_string = ' '.join(f"{key}: {value}; " for key, value in parameter.items())
            header.append(param_name + ': ' + str(dict_string))
        else:
            if param_name != 'lib':
                header.append(param_name + ': ' + str(parameter))
    header.append(f"")

    header.append(f"Data:")

    return header

