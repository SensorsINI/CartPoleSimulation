import numpy as np
from CartPole.state_utilities import ANGLE_IDX

TARGET_TIME_UP = 3.0  # s
TARGET_TIME_DOWN = 4.0

TARGET_ANGLE_UP = np.pi/5.0
TARGET_ANGLE_DOWN = 4.0*np.pi/5.0


class CheckStabilized:
    
    def __init__(self, dt, pole_position_init='down'):
        self.samples_stabilized_min = TARGET_TIME_UP/dt  # Gives for how many calls of this function the pole needs to be upright to return true
        self.samples_down_min = TARGET_TIME_DOWN/dt  # Due to noise in sensor reading you can wrongly read that pole is in lower half plane. This should account for that.
    
        self.pole_position = pole_position_init
        self.pole_position_now = pole_position_init

        self.counter = 0
        
    def check(self, s):
        
        if abs(s[ANGLE_IDX]) < TARGET_ANGLE_UP:
            self.pole_position_now = 'up'
        elif abs(s[ANGLE_IDX]) > TARGET_ANGLE_DOWN:
            self.pole_position_now = 'down'
        
        if self.pole_position == self.pole_position_now:
            self.counter = 0
        else:
            if self.pole_position == 'up':
                if self.counter == self.samples_down_min:
                    self.pole_position = self.pole_position_now
            else:
                if self.counter == self.samples_stabilized_min:
                    self.pole_position = self.pole_position_now
            self.counter += 1

        if self.pole_position == 'up':
            return True
        else:
            return False
