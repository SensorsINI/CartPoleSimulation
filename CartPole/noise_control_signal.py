import numpy as np
from scipy.stats import truncnorm

from CartPole.cartpole_parameters import controlBias, controlDisturbance, controlDisturbance_mode

def add_control_noise(Q_calculated, rng):

    if controlDisturbance_mode == 'OFF':
        Q_applied = Q_calculated
    elif controlDisturbance_mode == 'additive':
        Q_applied = Q_calculated + controlDisturbance * rng.standard_normal(
            size=np.shape(Q_calculated), dtype=np.float32) + controlBias
    elif controlDisturbance_mode == 'truncnorm':
        scale = controlDisturbance
        loc = Q_calculated + controlBias
        Q_applied = truncnorm.rvs((-1.0 - loc) / scale, (1.0 - loc) / scale, loc=loc,
                                  scale=scale, random_state=rng)
        Q_applied = np.cast[np.float32](Q_applied)
    else:
        raise ValueError('controlDisturbance_mode with value {} not valid'.format(controlDisturbance_mode))

    return Q_applied
