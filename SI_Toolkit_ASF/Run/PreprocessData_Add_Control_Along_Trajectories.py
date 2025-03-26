"""
This script calculate the control signal along the prerecorded trajectories.
There need to be no relationship between the controller with which the trajectories were recorded
and the controller which is used here to calculate the control signal.

In Pycharm to get the progress bars display correctly you need to set
"Emulate terminal in output console" in the run configuration.

"""


from SI_Toolkit.data_preprocessing import transform_dataset

import numpy as np
from CartPole.state_utilities import STATE_VARIABLES
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
    get_files_from = f'./Experiment_Recordings/Experiment_26_03_2025/raw/Experiment-{formatted_index}.csv'
else:
    get_files_from = 'SI_Toolkit_ASF/Run/df_to_transform/informed_gru_2.csv'

save_files_to = './Experiment_Recordings/Experiment_26_03_2025/Q_calculated_offline_mpc'

controller = {
    "controller_name": "mpc",
    "optimizer_name": 'rpgd-tf',
    "environment_name": "CartPole",
    "action_space": MockSpace(-1.0, 1.0, (1,), np.float32),
    "state_components": STATE_VARIABLES,
    "environment_attributes_dict": {  # keys are names used by controller, values the csv column names
        "target_position": "target_position",
        "target_equilibrium": "target_equilibrium",
        # "L": "L_differentiate_",
        # "m_pole": "m_pole_differentiate_",
        "Q_ccrc": "Q_applied_-1",
    },
}

controller_output_variable_name = 'Q_calculated_offline_mpc'

if __name__ == '__main__':
    transform_dataset(get_files_from, save_files_to, transformation='add_control_along_trajectories',
                      controller_config=controller, controller_output_variable_name=controller_output_variable_name,
                      integration_num_evals=4,
                      save_output_only=False,
                      )
