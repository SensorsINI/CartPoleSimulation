import numpy as np


class TargetGenerator:
    def __init__(self):

        self.current_target = None

        self.tolerance = 0.015      # 10%

        self.x_lower_bound = -0.15
        self.x_upper_bound = 0.15

        self.counter_wait_for_success = 0
        self.counter_wait_if_success = 0

        dt = 0.03
        self.counter_wait_for_success_max = int(8.0/dt)
        self.counter_wait_if_success_max = int(0.5/dt)

    def generate_random_target_position(self):
        target_x = np.random.uniform(self.x_lower_bound, self.x_upper_bound)
        return target_x


    def step(self, pos):

        if self.current_target is None:
            self.current_target = self.generate_random_target_position()
            return self.current_target




        if np.linalg.norm(pos - self.current_target) < self.tolerance:
            self.counter_wait_if_success += 1
        else:
            self.counter_wait_if_success = 0
            self.counter_wait_for_success_max += 1


        if (
                self.counter_wait_if_success > self.counter_wait_if_success_max
                or
                self.counter_wait_for_success > self.counter_wait_for_success_max
        ):
            self.current_target = self.generate_random_target_position()
            self.counter_wait_if_success = 0
            self.counter_wait_for_success = 0
            print('New target position is {}, current position is {}'.format(self.current_target, pos))

        return self.current_target



