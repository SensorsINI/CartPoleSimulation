from random import random

import numpy as np


class ParameterUpdater:

    def __init__(self, parameter_update_config):

        self.mode = parameter_update_config['mode']
        self.change_every_x_seconds = parameter_update_config['change_every_x_seconds']
        self.reset_every_x_seconds = parameter_update_config['reset_every_x_seconds']
        self.increment = parameter_update_config['increment']

        self.time_of_the_last_change = 0.0
        self.time_of_the_last_reset = 0.0

    def update_parameter(
            self,
            current_value,
            time_now,
    ):
        if self.change_every_x_seconds and time_now - self.time_of_the_last_change < self.change_every_x_seconds:
            new_parameter_value = current_value
        elif self.reset_every_x_seconds and self.mode != 'constant' and time_now - self.time_of_the_last_reset >= self.reset_every_x_seconds:
            self.time_of_the_last_reset = time_now
            new_parameter_value = 0.0
        else:
            self.time_of_the_last_change = time_now

            if self.mode == 'constant':
                increment = 0.0
            elif self.mode == 'random walk':
                increment = (1.0 if random() < 0.5 else -1.0) * self.increment
            elif self.mode == 'increase':
                self.increment *= 1.000
                increment = self.increment
            elif self.mode == 'random':
                new_parameter_value = np.random.uniform(-np.pi, np.pi)
                return new_parameter_value
            else:
                raise ValueError('mode with value {} not valid'.format(self.mode))

            new_parameter_value = current_value + increment

        return new_parameter_value
