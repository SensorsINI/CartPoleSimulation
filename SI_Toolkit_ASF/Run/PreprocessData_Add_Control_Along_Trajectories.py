from SI_Toolkit.data_preprocessing import transform_dataset

import numpy as np
from CartPole.state_utilities import STATE_VARIABLES
from others.globals_and_utils import MockSpace

get_files_from = 'SI_Toolkit_ASF/Experiments/Pretrained-RNN-1-Derivative/Recordings/Train'
save_files_to = get_files_from

controller = {
    "controller_name": "mpc",
    "optimizer_name": "rpgd-tf",
    "dt_controller": 0.02,
    "environment_name": "CartPole",
    "action_space": MockSpace(-1.0, 1.0, (1,), np.float32),
    "state_components": STATE_VARIABLES,
    "environment_attributes_list": ["target_position", "target_equilibrium", "L"]
}

controller_output_variable_name = 'Q_calculated_MPC'

transform_dataset(get_files_from, save_files_to, transformation='add_control_along_trajectories',
                  controller=controller, controller_output_variable_name=controller_output_variable_name)
