import numpy as np
from numpy.random import SFC64, Generator

from datetime import datetime

from CartPole.state_utilities import STATE_VARIABLES, \
    ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX

from tqdm import trange

from CartPole._CartPole_mathematical_helpers import wrap_angle_rad

import yaml


def _noise_iir_factor(smoothing_factor):
    return smoothing_factor / np.sqrt(1 - ((1 - smoothing_factor) ** 2))

# Parameters for IIR noise propagation
dt_derivative = 0.02  # time interval used to calculate derivative
angle_smoothing = 1.0
position_smoothing = 0.2


def _noise_derivative(dt):
    return 2.0 / dt


sigma_Q = 0.1

# Trial to calculate the std from physical cartpole
# ANGLE_ADC_RANGE = 4095  # Range of angle values #
# ANGLE_NORMALIZATION_FACTOR = 2 * np.pi / ANGLE_ADC_RANGE
#
# TRACK_LENGTH = 0.396  # Total usable track length in meters
# POSITION_ENCODER_RANGE = 4660  # This is an empirical approximation
# POSITION_NORMALIZATION_FACTOR = TRACK_LENGTH/POSITION_ENCODER_RANGE
#
# sigma_angle_raw = 14.864 # Angle noise variance as measured by Asude
# sigma_angle = _noise_iir_factor(angle_smoothing)*sigma_angle_raw*ANGLE_NORMALIZATION_FACTOR
# sigma_position_raw = 1.0 * POSITION_NORMALIZATION_FACTOR
# sigma_position = _noise_iir_factor(position_smoothing)*sigma_position_raw
#
# sigma_angleD = _noise_derivative(dt_derivative)*sigma_angle
# sigma_positionD = _noise_derivative(dt_derivative)*sigma_position

try:
    config = yaml.load(open("CartPoleSimulation/config.yml", "r"), Loader=yaml.FullLoader)
except FileNotFoundError:
    config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

sigma_angle = config["cartpole"]["noise"]["sigma_angle"]
sigma_position = config["cartpole"]["noise"]["sigma_position"]

sigma_angleD = config["cartpole"]["noise"]["sigma_angleD"]
sigma_positionD = config["cartpole"]["noise"]["sigma_positionD"]

NOISE_MODE = config["cartpole"]["noise"]["noise_mode"]

class NoiseAdder:
    def __init__(self):

        global sigma_angle, sigma_position, sigma_angleD, sigma_positionD

        SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 77.0)
        self.rng_noise_adder = Generator(SFC64(SEED))

        self.noise_mode = NOISE_MODE

        self.sigma_Q = sigma_Q

    def add_noise_to_measurement(self, s, copy=True):

        if copy == True:
            s_noisy = np.copy(s)
        else:
            s_noisy = s

        if self.noise_mode == 'OFF':
            pass
        else:
            s_noisy[ANGLE_IDX] += sigma_angle * self.rng_noise_adder.standard_normal(dtype=np.float32)
            s_noisy[ANGLE_IDX] = wrap_angle_rad(s_noisy[ANGLE_IDX])

            s_noisy[ANGLE_COS_IDX] = np.cos(s_noisy[ANGLE_IDX])
            s_noisy[ANGLE_SIN_IDX] = np.sin(s_noisy[ANGLE_IDX])

            s_noisy[POSITION_IDX] += sigma_position * self.rng_noise_adder.standard_normal(dtype=np.float32)


            s_noisy[ANGLED_IDX] += sigma_angleD * self.rng_noise_adder.standard_normal(dtype=np.float32)
            s_noisy[POSITIOND_IDX] += sigma_positionD * self.rng_noise_adder.standard_normal(dtype=np.float32)

        return s_noisy

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