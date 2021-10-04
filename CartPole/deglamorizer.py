import numpy as np

class Deglamorizer():
    def __init__(self,
                 latency=0.002,
                 simulation_dt=0.002,
                 ):

        self.latency = latency
        self.latency_buffer_len = int(latency/simulation_dt)
        self.latency_buffer = None

    def add_noise_and_latency_to_measurement(self, s):
        ...