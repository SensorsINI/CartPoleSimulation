import numpy as np
from numpy.random import SFC64, Generator

from datetime import datetime

from CartPole.state_utilities import STATE_VARIABLES, \
    ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX

from tqdm import trange

from CartPole._CartPole_mathematical_helpers import wrap_angle_rad

sigma_Q = 0.1

sigma_angle = 0.2
sigma_position = 0.1

# For iir noise propagation mode sigma_angleD and sigma_positionD
# Are calculated from sigma_angle and sigma_position
# The values below are overwritten in such case
sigma_angleD = 0.2
sigma_positionD = 0.1

# Parameters for IIR noise propagation
dt_derivative = 0.005  # time interval used to calculate derivative
angle_smoothing = 0.8
position_smoothing = 0.9

NOISE_MODE = 'noise_off'

# NOISE_MODE = 'preset'
# NOISE_MODE = 'iir'
# NOISE_MODE = 'no_derivatives'


def _noise_iir_factor(smoothing_factor):
    return smoothing_factor / np.sqrt(1 - ((1 - smoothing_factor) ** 2))


def _noise_derivative(dt):
    return 2.0 / dt


class NoiseAdder:
    def __init__(self):

        SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 77.0)
        self.rng_noise_adder = Generator(SFC64(SEED))

        self.noise_mode = NOISE_MODE

        self.sigma_Q = sigma_Q

        self.sigma_angle = sigma_angle
        self.sigma_position = sigma_position

        if self.noise_mode == 'iir':
            self.sigma_angleD = self.sigma_angle * _noise_iir_factor(angle_smoothing) * _noise_derivative(dt_derivative)
            self.sigma_positionD = self.sigma_position * _noise_iir_factor(position_smoothing) * _noise_derivative(
                dt_derivative)
        else:
            self.sigma_angleD = sigma_angleD
            self.sigma_positionD = sigma_positionD
            
    def add_noise_to_measurement(self, s, copy=True):

        if copy == True:
            s_noisy = np.copy(s)
        else:
            s_noisy = s

        if self.noise_mode == 'noise_off':
            pass
        else:
            s_noisy[ANGLE_IDX] += self.sigma_angle * self.rng_noise_adder.standard_normal(dtype=np.float32)
            s_noisy[ANGLE_IDX] = wrap_angle_rad(s_noisy[ANGLE_IDX])
            s_noisy[ANGLE_COS_IDX] = np.cos(s_noisy[ANGLE_IDX])
            s_noisy[ANGLE_SIN_IDX] = np.sin(s_noisy[ANGLE_IDX])
            s_noisy[POSITION_IDX] += self.sigma_position * self.rng_noise_adder.standard_normal(dtype=np.float32)
            if self.noise_mode == 'no_derivatives':
                s_noisy[ANGLED_IDX] = 0.0
                s_noisy[POSITIOND_IDX] = 0.0
            else:
                s_noisy[ANGLED_IDX] += self.sigma_angleD * self.rng_noise_adder.standard_normal(dtype=np.float32)
                s_noisy[POSITIOND_IDX] += self.sigma_positionD * self.rng_noise_adder.standard_normal(dtype=np.float32)

        return s_noisy

    def add_noise_to_control_input(self, Q):

        Q_noisy = np.copy(Q)
        Q_noisy += self.sigma_Q * self.rng_noise_adder.standard_normal(dtype=np.float32)

        return Q_noisy


if __name__ == '__main__':
    from CartPole.state_utilities import create_cartpole_state

    NoiseAdderInstance = NoiseAdder()
    s = create_cartpole_state()

    for i in trange(10):
        s_noisy = NoiseAdderInstance.add_noise_to_measurement(s)
        print(s_noisy)
        s += 1
    for i in trange(10):
        s_noisy = NoiseAdderInstance.add_noise_to_measurement(s)
        print(s_noisy)
        s += 1