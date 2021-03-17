"""
Small general mathematical functions.
This file was necessary to make CartPole module self-contained.
"""

from math import fmod
from types import SimpleNamespace

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

def create_cartpole_state(state: dict={}) -> np.ndarray:
    """
    Constructor of cartpole state from named arguments. The order of variables is fixed in STATE_VARIABLES.

    Input parameters are passed as a dict with the following possible keys. Other keys are ignored.
    Unset key-value pairs are initialized to 0.
    
    :param angle: Pole angle. 0 means pole is upright. Clockwise angle rotation is defined as negative.
    :param angleD: Angular velocity of pole.
    :param angleDD: Angular acceleration of pole.
    :param position: Horizontal position of pole.
    :param positionD: Horizontal velocity of pole. Cart movement to the right is positive.
    :param positionDD: Horizontal acceleration of pole.

    :returns: A numpy.ndarray with values filled in order set by STATE_VARIABLES
    """
    state['angle_cos'] = np.cos(state['angle']) if 'angle' in state.keys() else np.cos(0.0)
    state['angle_sin'] = np.sin(state['angle']) if 'angle' in state.keys() else np.sin(0.0)

    s = np.zeros_like(STATE_VARIABLES, dtype=float)
    for i, v in enumerate(STATE_VARIABLES):
        s[i] = state.get(v) if v in state.keys() else s[i]
    return s


def conditional_decorator(dec, cond):
    def decorator(func):
        return dec(func) if cond else func
    return decorator


def cartpole_state_varname_to_index(variable_name: str) -> int:
    return np.where(STATE_VARIABLES == variable_name)[0][0]


def cartpole_state_index_to_varname(index: int) -> str:
    return STATE_VARIABLES[index]


def cartpole_state_namespace_to_vector(s_namespace: SimpleNamespace) -> np.ndarray:
    s_array = np.zeros_like(STATE_VARIABLES, dtype=float)
    for a in STATE_VARIABLES:
        s_array[cartpole_state_varname_to_index(a)] = getattr(s_namespace, a, s_array[cartpole_state_varname_to_index(a)])
    return s_array


def cartpole_state_vector_to_namespace(s_vector: np.ndarray) -> SimpleNamespace:
    s_namespace = SimpleNamespace()
    for i, a in enumerate(STATE_VARIABLES):
        setattr(s_namespace, a, s_vector[i])
    return s_namespace


# # Test functions
# s = create_cartpole_state(dict(angleD=12.1, angleDD=-33.5, position=2.3, positionD=-19.77, positionDD=3.42))
# s[cartpole_state_varname_to_index('positionD')] = -14.9
# cartpole_state_index_to_varname(4)

# sn = SimpleNamespace()
# sn.position=23.55
# sn.angleDD=4.11
# sn.eew = -1.22
# q = cartpole_state_namespace_to_vector(sn)
# v = cartpole_state_vector_to_namespace(q)

# print(s)