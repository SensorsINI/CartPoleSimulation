import numpy as np
from scipy.stats import truncnorm

from CartPole.correlated_noise_source import CorrelatedNoiseGenerator


class ControlNoiseGenerator:
    def __init__(self,
                 rng,
                 controlNoise_Mode=None,
                 controlNoise_Scale=None,
                 controlNoise_Bias=None,
                 controlNoise_Correlation=None,
                 dt=None,
                 initial_state=0.0,
                 ):
        self.rng = rng
        self.controlNoise_Mode = controlNoise_Mode
        self.controlNoise_Scale = controlNoise_Scale
        self.controlNoise_Bias = controlNoise_Bias
        self.controlNoise_Correlation = controlNoise_Correlation
        self.dt = dt
        self.initial_state = initial_state

        self.correlated_noise_generator = CorrelatedNoiseGenerator(
            rng=self.rng,
            bias=self.controlNoise_Bias,
            sigma=self.controlNoise_Scale,
            a=self.controlNoise_Correlation,
            dt=self.dt,
            initial_state=self.initial_state,
        )


    def add_control_noise(
            self,
            Q_calculated,
    ):
        if self.controlNoise_Mode == 'OFF':
            Q_applied = Q_calculated
        elif self.controlNoise_Mode == 'gaussian':

            Q_applied = Q_calculated + self.correlated_noise_generator.sample(size=np.shape(Q_calculated))

        elif self.controlNoise_Mode == 'truncnorm':
            scale = self.controlNoise_Scale
            loc = Q_calculated + self.controlNoise_Bias
            Q_applied = truncnorm.rvs((-1.0 - loc) / scale, (1.0 - loc) / scale, loc=loc,
                                      scale=scale, random_state=self.rng)
            Q_applied = np.cast[np.float32](Q_applied)
        else:
            raise ValueError('controlNoiseMode with value {} not valid'.format(self.controlNoise_Mode))

        return Q_applied

    def reset(self, noise_initial_state=0.0, dt=None):
        """
        Reset the internal state and time step of the correlated noise generator.

        Parameters:
            noise_initial_state (float): The new initial state for the noise process.
            dt (float): The new time step. If None, the current dt is retained.
        """
        self.correlated_noise_generator.reset(noise_initial_state=noise_initial_state, dt=dt)
