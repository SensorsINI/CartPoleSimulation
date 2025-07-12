from SI_Toolkit.data_preprocessing import transform_dataset

import numpy as np
from CartPole.state_utilities import STATE_VARIABLES, CONTROL_INPUTS
from others.globals_and_utils import MockSpace

import argparse

def args_fun():
    parser = argparse.ArgumentParser(description='Generate CartPole data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--secondary_experiment_index', default=-1, type=int,
                        help='Additional index to the experiment folder (ML Pipeline mode) or file (otherwise) name. -1 to skip.')


    args = parser.parse_args()

    if args.secondary_experiment_index == -1:
        args.secondary_experiment_index = None

    digits = 3
    if args.secondary_experiment_index is not None:
        formatted_index = f"{args.secondary_experiment_index:0{digits}d}"
    else:
        formatted_index = None

    return formatted_index

formatted_index = args_fun()
if formatted_index is not None:
    get_files_from = f'./Experiment_Recordings/Experiment-{formatted_index}.csv'
else:
    get_files_from = 'SI_Toolkit_ASF/Experiments/CPP_MPCswingups_before_23_04_2025/Recordings/XTest'

save_files_to = './SI_Toolkit_ASF/Experiments/CPP_MPCswingups_before_23_04_2025/Recordings/XTest'

variables_dict = {
    "action_components": CONTROL_INPUTS,
    "state_components": STATE_VARIABLES,
    "environment_attributes_dict": {  # keys are names used by ODE, values the csv column names
    },
}

if __name__ == '__main__':
    transform_dataset(get_files_from, save_files_to, transformation='calculate_cartpole_ode_along_trajectories',
                      variables_dict=variables_dict,
                      )
