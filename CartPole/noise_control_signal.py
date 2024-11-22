import numpy as np
from scipy.stats import truncnorm


def add_control_noise(
        Q_calculated,
        rng,
        ControlDisturbanceMode=None,
        ControlDisturbanceMult=None,
        ControlDisturbanceAdd=None):

    if ControlDisturbanceMode == 'OFF':
        Q_applied = Q_calculated
    elif ControlDisturbanceMode == 'additive':
        Q_applied = Q_calculated + ControlDisturbanceMult * rng.standard_normal(
            size=np.shape(Q_calculated), dtype=np.float32) + ControlDisturbanceAdd
    elif ControlDisturbanceMode == 'truncnorm':
        scale = ControlDisturbanceMult
        loc = Q_calculated + ControlDisturbanceAdd
        Q_applied = truncnorm.rvs((-1.0 - loc) / scale, (1.0 - loc) / scale, loc=loc,
                                  scale=scale, random_state=rng)
        Q_applied = np.cast[np.float32](Q_applied)
    else:
        raise ValueError('controlDisturbance_mode with value {} not valid'.format(ControlDisturbanceMode))

    return Q_applied
