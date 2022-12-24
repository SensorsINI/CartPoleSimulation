"""
This is a linear-quadratic regulator
It assumes that the input relation is u = Q*u_max (no fancy motor model) !
"""

from SI_Toolkit.computation_library import NumpyLibrary
import numpy as np
from Control_Toolkit.Controllers import template_controller

from Control_Toolkit.Optimizers import template_optimizer
from CartPole.state_utilities import cartpole_state_varname_to_index
from others.globals_and_utils import load_config

config = load_config("config.yml")

class controller_pid(template_controller):
    _computation_library = NumpyLibrary
    
    def __init__(self):
        self.P_angle = config["controller"]["pid"]["P_angle"]
        self.I_angle = config["controller"]["pid"]["I_angle"]
        self.D_angle = config["controller"]["pid"]["D_angle"]
        self.P_position = config["controller"]["pid"]["P_position"]
        self.I_position = config["controller"]["pid"]["I_position"]
        self.D_position = config["controller"]["pid"]["D_position"]

    def step(self, state: np.ndarray, target_position: np.ndarray, time=None):
        error = -np.array([
            [state[cartpole_state_varname_to_index('position')] - target_position],
            [state[cartpole_state_varname_to_index('positionD')]],
            [state[cartpole_state_varname_to_index('angle')]],
            [state[cartpole_state_varname_to_index('angleD')]]
        ])

        positionCMD = self.P_position * error[0, :] + self.D_position * error[1]
        angleCMD = self.P_angle * error[2] + self.D_angle * error[3]

        N = np.asscalar(angleCMD + positionCMD)
        Q = N

        if Q > 1.0:
            Q = 1.0
        elif Q < -1.0:
            Q = -1.0
        else:
            pass

        return N
