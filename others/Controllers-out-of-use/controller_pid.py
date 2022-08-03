"""
This is a linear-quadratic regulator
It assumes that the input relation is u = Q*u_max (no fancy motor model) !
"""

import os
import scipy
import numpy as np

from Control_Toolkit.Controllers import template_controller
from CartPole.state_utilities import create_cartpole_state, cartpole_state_varname_to_index
from CartPole.cartpole_model import u_max, s0
from CartPole.cartpole_jacobian import cartpole_jacobian

import yaml
config = yaml.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "config.yml"), "r"), Loader=yaml.FullLoader)

class controller_pid(template_controller):
    def __init__(self):
        self.P_angle = config["controller"]["pid"]["P_angle"]
        self.I_angle = config["controller"]["pid"]["I_angle"]
        self.D_angle = config["controller"]["pid"]["D_angle"]
        self.P_position = config["controller"]["pid"]["P_position"]
        self.I_position = config["controller"]["pid"]["I_position"]
        self.D_position = config["controller"]["pid"]["D_position"]

    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        error = -np.array([
            [s[cartpole_state_varname_to_index('position')] - target_position],
            [s[cartpole_state_varname_to_index('positionD')]],
            [s[cartpole_state_varname_to_index('angle')]],
            [s[cartpole_state_varname_to_index('angleD')]]
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
