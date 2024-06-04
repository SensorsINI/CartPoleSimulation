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

get_files_from = 'SI_Toolkit_ASF/Experiments/Experiment-29-30-quant-10-shiftD-1/Recordings/Train'
save_files_to = 'SI_Toolkit_ASF/Experiments/Experiment-29-30-quant-10-shiftD-1/Recordings/Train-neural'

controller = {
    "controller_name": "neural-imitator",
    "optimizer_name": None,
    "dt_controller": 0.02,
    "environment_name": "CartPole",
    "action_space": MockSpace(-1.0, 1.0, (1,), np.float32),
    "state_components": STATE_VARIABLES,
    "environment_attributes_list": {  # keys are names used by controller, values the csv column names
        "target_position": "target_position",
        "target_equilibrium": "target_equilibrium",
        "L": "L",
    },
}

controller_output_variable_name = 'Q_calculated_neural'

transform_dataset(get_files_from, save_files_to, transformation='add_control_along_trajectories',
                  controller=controller, controller_output_variable_name=controller_output_variable_name)
