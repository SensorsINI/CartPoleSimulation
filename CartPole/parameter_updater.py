from random import random

import numpy as np


class ParameterUpdater:

    def __init__(self, parameter_update_config):

        self.init_value = parameter_update_config['init_value']
        if self.init_value == 'random': self.init_value = np.random.uniform(*parameter_update_config['range_random'])

        self.mode = parameter_update_config['mode']

        self.change_every_x_seconds = parameter_update_config['change_every_x_seconds']
        self.reset_every_x_seconds = parameter_update_config['reset_every_x_seconds']
        if self.change_every_x_seconds == 'inf': self.change_every_x_seconds = np.inf
        if self.reset_every_x_seconds == 'inf': self.reset_every_x_seconds = np.inf

        self.increment = parameter_update_config['increment']
        self.range_random = parameter_update_config['range_random']
        self.range_clip = parameter_update_config['range_clip']

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
            new_parameter_value = self.init_value
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
                new_parameter_value = np.random.uniform(*self.range_random)
                return new_parameter_value
            else:
                raise ValueError('mode with value {} not valid'.format(self.mode))

            new_parameter_value = current_value + increment

            if self.range_clip:
                new_parameter_value = np.clip(new_parameter_value, *self.range_clip)

        return new_parameter_value


class ParameterJointUpdater:
    def __init__(self, parameter_update_config_1, parameter_update_config_2, mode=1):

        self.mode = mode

        self.change_every_x_seconds_1 = parameter_update_config_1['change_every_x_seconds']
        self.change_every_x_seconds_2 = parameter_update_config_2['change_every_x_seconds']

        self.range_clip_1 = parameter_update_config_1['range_clip']
        self.range_clip_2 = parameter_update_config_2['range_clip']

        # Internal state to track current phase and time in mode 2
        self.current_phase = 0  # 0: param1 up, 1: param2 up, 2: param1 down, 3: param2 down
        self.last_update_time = 0  # Time when the last phase started

    def update_parameter(
            self,
            time_now,
    ):

        if self.mode == 1:
            b_1 = (self.range_clip_1[1]+self.range_clip_1[0])/2
            b_2 = (self.range_clip_2[1]+self.range_clip_2[0])/2

            A_1 = b_1 - self.range_clip_1[0]
            A_2 = b_2 - self.range_clip_2[0]

            T1 = self.change_every_x_seconds_1
            T2 = self.change_every_x_seconds_2

            new_parameter_value_1 = b_1 + A_1*np.sin((time_now/T1)*2*np.pi)
            new_parameter_value_2 = b_2 + A_2*np.cos((time_now/T2)*2*np.pi)
        elif self.mode == 2:

            # Determine the phase duration based on the configured change times
            phase_duration = self.change_every_x_seconds_1

            # Check if we need to transition to the next phase
            if time_now - self.last_update_time > phase_duration:
                self.current_phase = (self.current_phase + 1) % 4
                self.last_update_time = time_now

            # Compute parameter values based on the current phase
            if self.current_phase == 0:  # param1 goes min to max, param2 stays at min
                progress = (time_now - self.last_update_time) / phase_duration
                new_parameter_value_1 = self.range_clip_1[0] + progress * (self.range_clip_1[1] - self.range_clip_1[0])
                new_parameter_value_2 = self.range_clip_2[0]
            elif self.current_phase == 1:  # param2 goes min to max, param1 stays at max
                progress = (time_now - self.last_update_time) / phase_duration
                new_parameter_value_1 = self.range_clip_1[1]
                new_parameter_value_2 = self.range_clip_2[0] + progress * (self.range_clip_2[1] - self.range_clip_2[0])
            elif self.current_phase == 2:  # param1 goes max to min, param2 stays at max
                progress = (time_now - self.last_update_time) / phase_duration
                new_parameter_value_1 = self.range_clip_1[1] - progress * (self.range_clip_1[1] - self.range_clip_1[0])
                new_parameter_value_2 = self.range_clip_2[1]
            elif self.current_phase == 3:  # param2 goes max to min, param1 stays at min
                progress = (time_now - self.last_update_time) / phase_duration
                new_parameter_value_1 = self.range_clip_1[0]
                new_parameter_value_2 = self.range_clip_2[1] - progress * (self.range_clip_2[1] - self.range_clip_2[0])


        return new_parameter_value_1, new_parameter_value_2
