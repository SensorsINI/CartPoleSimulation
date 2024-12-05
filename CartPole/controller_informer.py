import numpy as np
# mode: # 'OFF', 'ON', 'switching_random', 'switching_regular'


class ControllerInformer:
    def __init__(self, config):
        self.mode = config['mode']
        self.change_to_on_after_x_seconds_off = config['change_to_on_after_x_seconds_off']
        self.change_to_off_after_x_seconds_on = config['change_to_off_after_x_seconds_on']

        self.change_to_on_after_x_seconds_off_random = np.random.uniform(0, config['change_to_on_after_x_seconds_off'])
        self.change_to_off_after_x_seconds_on_random = np.random.uniform(0, config['change_to_off_after_x_seconds_on'])

        self.time_last_switch_to_on = 0.0
        self.time_last_switch_to_off = 0.0

        self.value_to_return = 'default'

    def get_parameters(self, true_value, default_value, time_now):

        if self.mode == 'OFF':
            self.value_to_return = 'default'
        elif self.mode == 'ON':
            self.value_to_return = 'true'
        elif self.mode == 'switching_regular':
            if self.value_to_return == 'default':
                if time_now - self.time_last_switch_to_off >= self.change_to_on_after_x_seconds_off:
                    self.value_to_return = 'true'
                    self.time_last_switch_to_on = time_now
            elif self.value_to_return == 'true':
                if time_now - self.time_last_switch_to_on >= self.change_to_off_after_x_seconds_on:
                    self.value_to_return = 'default'
                    self.time_last_switch_to_off = time_now
        elif self.mode == 'switching_random':
            if self.value_to_return == 'default':
                if time_now - self.time_last_switch_to_off >= self.change_to_on_after_x_seconds_off_random:
                    self.value_to_return = 'true'
                    self.time_last_switch_to_on = time_now
                    self.change_to_on_after_x_seconds_off_random = np.random.uniform(0, self.change_to_on_after_x_seconds_off)
            elif self.value_to_return == 'true':
                if time_now - self.time_last_switch_to_on >= self.change_to_off_after_x_seconds_on_random:
                    self.value_to_return = 'default'
                    self.time_last_switch_to_off = time_now
                    self.change_to_off_after_x_seconds_on_random = np.random.uniform(0, self.change_to_off_after_x_seconds_on)


        if self.value_to_return == 'default':
            return default_value
        elif self.value_to_return == 'true':
            return true_value
        else:
            raise ValueError('value_to_return must be either "default" or "true"')
