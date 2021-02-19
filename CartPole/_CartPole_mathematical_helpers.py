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
