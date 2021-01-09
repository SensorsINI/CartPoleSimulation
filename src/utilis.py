# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:28:34 2020

@author: Marcin
"""

from math import fmod

from src.globals import *

from timeit import default_timer as timer
from time import sleep

from collections import deque

import warnings

import pandas as pd

from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from memory_profiler import profile


# warnings.warn("Warning...........Message")


def normalize_angle_rad(angle):
    Modulo = fmod(angle, 2 * np.pi)  # positive modulo
    if Modulo < -np.pi:
        angle = Modulo + 2 * np.pi
    elif Modulo > np.pi:
        angle = Modulo - 2 * np.pi
    else:
        angle = Modulo
    return angle


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

# @profile(precision=4)
def Generate_Experiment(MyCart,
                        initial_state = [None,None,None,None],
                        exp_len=random_length_globals,
                        dt=dt_main_simulation_globals,
                        track_relative_complexity = track_relative_complexity_globals,
                        interpolation_type=interpolation_type_globals,
                        turning_points_period=turning_points_period_globals,
                        start_random_target_position_at=start_random_target_position_at_globals,
                        end_random_target_position_at=end_random_target_position_at_globals,
                        csv=None,
                        save_data_online = True,
                        controller='manual-stabilization'):
    """
    This function runs a random CartPole experiment
    and returns the history of CartPole states, control inputs and desired cart position
    :param MyCart: instance of CartPole containing CartPole dynamics
    :param exp_len: How many time steps should the experiment have
                (default: 64+640+1 this format is used as it can )
    """



    # Set CartPole in the right (automatic control) mode
    mode = MyCart.controller_names.index(controller)
    MyCart.set_mode(mode)

    MyCart.turning_points = None
    MyCart.interpolation_type = interpolation_type
    MyCart.turning_points_period = turning_points_period
    MyCart.start_random_target_position_at = start_random_target_position_at
    MyCart.end_random_target_position_at = end_random_target_position_at

    if save_data_online:
        MyCart.save_history = False
    else:
        MyCart.save_history = True
    MyCart.use_pregenerated_target_position = 1
    MyCart.dt = dt

    MyCart.reset_state()

    # Generate new random function returning desired target position of the cart
    MyCart.random_length = exp_len
    MyCart.track_relative_complexity = track_relative_complexity  # Complexity of generated target position track
    MyCart.Generate_Random_Trace_Function()
    number_of_timesteps = int(np.ceil(MyCart.random_length / MyCart.dt))

    # Randomly set the initial state

    MyCart.time = 0.0
    if initial_state[0] is None:
        MyCart.s.position = np.random.uniform(low=-MyCart.HalfLength / 2.0,
                                              high=MyCart.HalfLength / 2.0)
    else:
        MyCart.s.position = initial_state[0]

    if initial_state[1] is None:
        MyCart.s.positionD = np.random.uniform(low=-10.0,
                                               high=10.0)
    else:
        MyCart.s.positionD = initial_state[1]

    if initial_state[2] is None:
        MyCart.s.angle = np.random.uniform(low=-17.5 * (np.pi / 180.0),
                                           high=17.5 * (np.pi / 180.0))
    else:
        MyCart.s.angle = initial_state[2]

    if initial_state[3] is None:
        MyCart.s.angleD = np.random.uniform(low=-15.5 * (np.pi / 180.0),
                                            high=15.5 * (np.pi / 180.0))
    else:
        MyCart.s.angleD = initial_state[3]

    # Target position at time 0
    MyCart.target_position = MyCart.random_track_f(MyCart.time)

    MyCart.Update_Q()

    MyCart.u = Q2u(MyCart.Q, MyCart.p)

    MyCart.reset_dict_history()

    if save_data_online and csv is not None:
        MyCart.save_history_csv(csv_name=csv, init=True, iter=False)
        MyCart.save_history_csv(csv_name=csv, init=False, iter=True)

    # Run the CartPole experiment for number of time
    for _ in trange(number_of_timesteps):

        # Print an error message if it runs already to long (should stop before)
        if MyCart.time > MyCart.t_max_pre:
            raise Exception('ERROR: It seems the experiment is running too long...')

        MyCart.update_state()

        # if abs(MyCart.s.angle) > 0.8*np.pi:
        #     raise ValueError('Cart went unstable')

        if save_data_online and csv is not None:
            MyCart.save_history_csv(csv_name=csv, init=False, iter=True)

    data = pd.DataFrame(MyCart.dict_history)

    if csv is not None and not save_data_online:
        MyCart.save_history_csv(csv_name=csv)

    if not save_data_online:
        MyCart.summary_plots()

    MyCart.reset_dict_history()
    MyCart.reset_state()

    return data




def pd_plotter_simple(df, x_name=None, y_name=None, idx_range=[0, -1], color='blue', dt=None):

    if y_name is None:
        raise ValueError ('You must provide y_name')
    y = df[y_name].iloc[idx_range[0]:idx_range[1]]

    if x_name is None:
        x = np.arange(len(y))
        if dt is not None:
            x = x*dt
    else:
        x = df[x_name].iloc[idx_range[0]:idx_range[1]]

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y, color=color, linewidth=3)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_xlabel(x_name, fontsize=18)
    plt.show()

