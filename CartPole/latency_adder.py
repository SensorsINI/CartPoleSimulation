import numpy as np
from numpy.random import SFC64, Generator

from datetime import datetime

from CartPole.state_utilities import STATE_VARIABLES, \
    ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX

from tqdm import trange

MAX_LATENCY_LEN = 50  # Total size of latency buffer.

class LatencyAdder():
    def __init__(self,
                 latency=0.0,
                 dt_sampling=0.002,
                 ):

        self.dt_sampling = dt_sampling
        self.latency = latency

        self.latency_len = None
        self.latency_len_int = None
        self.latency_len_fraction = None
        self.max_latency = None
        self.set_latency(latency)
        self.latency_buffer_len = MAX_LATENCY_LEN+2
        self.latency_buffer = np.zeros((self.latency_buffer_len, len(STATE_VARIABLES)))
        self.latency_buffer[:, ANGLE_COS_IDX] = 1.0

        self.latency_buffer_current_index = 0

        # The below array allows for fast lookup where is the value laying a predifined
        self.latency_buffer_lookup = np.zeros((self.latency_buffer_len, self.latency_buffer_len), dtype=np.short)
        for i in range(self.latency_buffer_len):
            for j in range(self.latency_buffer_len):
                self.latency_buffer_lookup[i, j] = self.access_past_value(i, j)

    def add_current_state_to_latency_buffer(self, s):
        """
        Adds a current state s to the circular buffer
        and shift the index pointing to the next position in the buffer to be filled
        (the oldest position)
        """
        self.latency_buffer[self.latency_buffer_current_index, :] = s

        if self.latency_buffer_current_index == self.latency_buffer_len-1:
            self.latency_buffer_current_index = 0
        else:
            self.latency_buffer_current_index += 1

    def access_past_value(self, latency_buffer_current_index, i):
        """
        i gives how many steps in the past lays the requested state
        the function than returns the index in the circular buffer where this state can be found
        """
        if i < 0:
            raise('i must be positive!')

        index = latency_buffer_current_index-1-i
        if index < 0:
            index += self.latency_buffer_len
            if index < 0:
                raise('Requested point to far in the past - not more in the buffer')

        return index

    def get_delayed_state(self, i):
        return self.latency_buffer_lookup(self.latency_buffer_current_index, i)

    def get_interpolated_delayed_state(self):
        s1 = self.latency_buffer[self.latency_buffer_lookup[self.latency_buffer_current_index, self.latency_len_int], :]
        s2 = self.latency_buffer[self.latency_buffer_lookup[self.latency_buffer_current_index, self.latency_len_int+1], :]

        return s1+self.latency_len_fraction*(s2-s1)
            
    def set_latency(self, latency):
        self.latency = latency
        self.latency_len = latency/self.dt_sampling
        self.latency_len_int = int(self.latency_len)
        self.latency_len_fraction = self.latency_len-self.latency_len_int
        self.max_latency = MAX_LATENCY_LEN*self.dt_sampling

if __name__ == '__main__':
    from CartPole.state_utilities import create_cartpole_state

    LatencyAdderInstance = LatencyAdder()
    s = create_cartpole_state()

    LatencyAdderInstance.set_latency(0.01)
    for i in trange(10):
        LatencyAdderInstance.add_current_state_to_latency_buffer(s)
        s_delayed = LatencyAdderInstance.get_interpolated_delayed_state()
        print(s_delayed)
        s+=1

    LatencyAdderInstance.set_latency(0.002)
    for i in trange(10):
        LatencyAdderInstance.add_current_state_to_latency_buffer(s)
        s_delayed = LatencyAdderInstance.get_interpolated_delayed_state()
        print(s_delayed)
        s+=1

