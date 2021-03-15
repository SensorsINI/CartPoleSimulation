"""
Small general mathematical functions.
This file was necessary to make CartPole module self-contained.
"""

from math import fmod

import numpy as np


# Wraps the angle into range [-π, π]
def wrap_angle_rad(angle):
    Modulo = fmod(angle, 2 * np.pi)  # positive modulo
    if Modulo < -np.pi:
        angle = Modulo + 2 * np.pi
    elif Modulo > np.pi:
        angle = Modulo - 2 * np.pi
    else:
        angle = Modulo
    return angle


STATE_VARIABLES = np.sort(['angle', 'angleD', 'angleDD', 'position', 'positionD', 'positionDD', 'angle_cos', 'angle_sin'])

def create_cartpole_state(angle: float=0.0, angleD: float=0.0, angleDD: float=0.0, position: float=0.0, positionD: float=0.0, positionDD: float=0.0) -> np.ndarray:
    """
    Constructor of cartpole state from named arguments. The order of variables is fixed in STATE_VARIABLES.

    :param angle: Pole angle. 0 means pole is upright. Clockwise angle rotation is defined as negative.
    :param angleD: Angular velocity of pole.
    :param angleDD: Angular acceleration of pole.
    :param position: Horizontal position of pole.
    :param positionD: Horizontal velocity of pole. Cart movement to the right is positive.
    :param positionDD: Horizontal acceleration of pole.

    :returns: A numpy.ndarray with values filled in order set by STATE_VARIABLES
    """
    angle_cos = np.cos(angle)
    angle_sin = np.sin(angle)
    s = np.zeros_like(STATE_VARIABLES, dtype=float)
    for i, v in enumerate(STATE_VARIABLES):
        s[i] = eval(v)
    return s


def cartpole_state_varname_to_index(variable_name: str) -> int:
    return np.where(STATE_VARIABLES == variable_name)[0][0]


def cartpole_state_index_to_varname(index: int) -> str:
    return STATE_VARIABLES[index]


# Test functions
# s = create_cartpole_state(angle=46.2, angleD=12.1, angleDD=-33.5, position=2.3, positionD=-19.77, positionDD=3.42)
# s[cartpole_state_varname_to_index('positionD')] = -14.9
# cartpole_state_index_to_varname(4)
# print(s)