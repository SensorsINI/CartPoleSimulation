from datetime import datetime
import numpy as np
from numpy.random import SFC64, Generator

from CartPole.cartpole_model import Q2u


def add_noise(dataset, noise_level):
    SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)
    rng_preprocess_brunton = Generator(SFC64(SEED))
    dataset['angle_noisy'] *= (1 + noise_level[0] * rng_preprocess_brunton.standard_normal(
        size=(len(dataset['angle_noisy'])), dtype=np.float32))
    dataset['angle_cos_noisy'] = np.cos(dataset['angle_noisy'])
    dataset['angle_sin_noisy'] = np.sin(dataset['angle_noisy'])
    dataset['angleD_noisy'] *= (1 + noise_level[1] * rng_preprocess_brunton.standard_normal(
        size=(len(dataset['angleD_noisy'])), dtype=np.float32))
    dataset['position_noisy'] *= (1 + noise_level[2] * rng_preprocess_brunton.standard_normal(
        size=(len(dataset['position_noisy'])), dtype=np.float32))
    dataset['positionD_noisy'] *= (1 + noise_level[3] * rng_preprocess_brunton.standard_normal(
        size=(len(dataset['positionD_noisy'])), dtype=np.float32))
    dataset['Q_noisy'] = (1 + noise_level[4] * rng_preprocess_brunton.standard_normal(
        size=(len(dataset['Q'])), dtype=np.float32))
    dataset['u_noisy'] = Q2u(dataset['Q_noisy'])