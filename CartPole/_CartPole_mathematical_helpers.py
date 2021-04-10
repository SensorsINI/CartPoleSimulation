"""
Small general mathematical functions.
This file was necessary to make CartPole module self-contained.
"""

from math import fmod
import numpy as np


# Wraps the angle into range [-π, π]
def wrap_angle_rad(angle: float) -> float:
    Modulo = fmod(angle, 2 * np.pi)  # positive modulo
    if Modulo < -np.pi:
        angle = Modulo + 2 * np.pi
    elif Modulo > np.pi:
        angle = Modulo - 2 * np.pi
    else:
        angle = Modulo
    return angle


def wrap_angle_rad_inplace(angle: np.ndarray) -> None:
    Modulo = np.fmod(angle, 2 * np.pi)  # positive modulo
    neg_wrap, pos_wrap = Modulo < -np.pi, Modulo > np.pi
    angle[neg_wrap] = Modulo[neg_wrap] + 2 * np.pi
    angle[pos_wrap] = Modulo[pos_wrap] - 2 * np.pi
    angle[~(neg_wrap | pos_wrap)] = Modulo[~(neg_wrap | pos_wrap)]


def conditional_decorator(dec, cond):
    def decorator(func):
        return dec(func) if cond else func
    return decorator


