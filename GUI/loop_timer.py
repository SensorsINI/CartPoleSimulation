import numpy as np

from timeit import default_timer as timer
from time import sleep

from collections import deque

import warnings

class loop_timer():
    """ Simple game loop timer that sleeps for leftover time (if any) at end of each iteration"""
    LOG_INTERVAL_SEC = 10
    NUM_SAMPLES = 1000

    def __init__(self, rate_hz: float = None, dt_target: float = None, do_diagnostics: bool = False) -> None:
        """ Make a new loop_timer, specifying the target frame rate in Hz or time interval dt_target in seconds

        :param rate_hz: the target loop rate in Hz. The rate can be changed anytime by modifying rate_hz.
        :param dt_target: the time interval dt in seconds. It can be changed anytime by modifying dt.
        :returns: new instance of loop_timer
        """

        self.rate_hz = None
        self.dt_target = None

        if (rate_hz is not None) and (dt_target is not None):
            raise Exception('You should provide either rate_hz OR dt, not both!')
        elif (rate_hz is None) and (dt_target is None):
            raise Exception('You must provide either rate_hz or dt!')
        elif (rate_hz is None) and (dt_target is not None):
            self.dt_target = dt_target  # rate_hz set automatically
        elif (self.rate_hz is not None) and (self.dt_target is None):
            self.rate_hz = rate_hz  # dt_target set automatically

        self.first_call_done = False

        self.last_iteration_start_time = 0

        self.do_diagnostics = do_diagnostics
        self.last_log_time = 0
        self.circ_buffer_dt = deque(iterable=np.zeros(self.NUM_SAMPLES), maxlen=self.NUM_SAMPLES)
        self.circ_buffer_leftover = deque(iterable=np.zeros(self.NUM_SAMPLES), maxlen=self.NUM_SAMPLES)
        self.circ_buffer_dt_real = deque(iterable=np.zeros(50), maxlen=50)

    @property
    def rate_hz(self):
        return self._rate_hz

    @property
    def dt_target(self):
        return self._dt_target

    @rate_hz.setter
    def rate_hz(self, new_rate):
        if new_rate is None:
            self._rate_hz = None
        elif new_rate > 0.0:
            self._rate_hz = float(new_rate)
            self._dt_target = 1.0 / new_rate
        else:
            raise Exception('{} is not valid target rate!'.format(new_rate))

    @dt_target.setter
    def dt_target(self, new_dt):
        if new_dt is None:
            self._dt_target = None
        elif new_dt > 0.0:
            self._dt_target = float(new_dt)
            self._rate_hz = 1.0 / new_dt
        elif new_dt == 0:
            self._dt_target = float(new_dt)
            self._rate_hz = np.inf
        else:
            raise Exception('{} is not valid target dt!'.format(new_dt))

    def start_loop(self):
        """ should be called to initialize the timer just before the entering the first loop"""
        self.last_iteration_start_time = timer()
        self.last_log_time = self.last_iteration_start_time
        self.first_call_done = True

    def sleep_leftover_time(self):
        """
        Call at the very end of each iteration.
        """
        now = timer()

        if not self.first_call_done:
            raise Exception('Loop timer was not initialized properly')

        dt = (now - self.last_iteration_start_time)
        leftover_time = self.dt_target - dt
        if leftover_time > 0:
            sleep(leftover_time)
            dt_real = self.dt_target
        else:
            dt_real = dt

        self.last_iteration_start_time = timer()

        # You need buffers not only for diagnostics, but also to measure speed-up
        self.circ_buffer_dt.append(dt)
        self.circ_buffer_leftover.append(leftover_time)
        self.circ_buffer_dt_real.append(dt_real)

        # Main functionality ends here. Lines below are just for diagnostics
        if self.do_diagnostics:

            if now - self.last_log_time > self.LOG_INTERVAL_SEC:
                self.last_log_time = now
                if leftover_time > 0:
                    print('Loop_timer slept for {:.3f} ms leftover time for desired loop interval {:.3f} ms.'
                          .format(leftover_time * 1000, self.dt_target * 1000))
                else:
                    if self.dt_target == 0.0:
                        warnings.warn('\nYou target the maximal simulation speed, '
                                      'the average time for simulation step is {:.3f} ms.\n'
                                      .format(-leftover_time * 1000))
                    else:
                        warnings.warn('\nTime ran over by {:.3f}ms the allowed time of {:.3f} ms.\n'
                                      .format(-leftover_time * 1000, self.dt_target * 1000))
                print('Average leftover time is {:.3f} ms and its variance {:.3f} ms'
                      .format(np.mean(self.circ_buffer_leftover) * 1000,
                              np.std(self.circ_buffer_leftover) * 1000))
                print('Average total time of calculations is {:.3f} ms and its variance {:.3f} ms'
                      .format(np.mean(self.circ_buffer_dt) * 1000,
                              np.std(self.circ_buffer_dt) * 1000))