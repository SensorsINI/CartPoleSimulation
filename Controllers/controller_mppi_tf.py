from importlib import import_module
from operator import attrgetter
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from CartPole.cartpole_model import TrackHalfLength, s0, u_max
from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                      ANGLED_IDX, POSITION_IDX, POSITIOND_IDX,
                                      create_cartpole_state)
from others.globals_and_utils import create_rng
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from Controllers.template_controller import template_controller


class controller_mppi_tf(template_controller):
    def __init__(self, seed: int, num_control_inputs: int, cc_weight: float, R: float, LBD: float, mpc_horizon: float, num_rollouts: int, dt: float, predictor_intermediate_steps: int, NU: float, SQRTRHOINV: float, GAMMA: float, SAMPLING_TYPE: str, NET_NAME: str, predictor_name: str, cost_function: str, clip_control_input: Union["list[float]", "list[list[float]]"], **kwargs):
        #First configure random sampler
        self.rng_mppi = create_rng(self.__class__.__name__, seed, use_tf=True)

        # Parametrization
        self.num_control_inputs = num_control_inputs

        cost_function = cost_function.replace('-', '_')
        cost_function_module = import_module(f"others.cost_functions.{cost_function}")
        self.q, self.phi = attrgetter("q", "phi")(cost_function_module)

        self.mppi_horizon = mpc_horizon
        self.num_rollouts = num_rollouts

        self.cc_weight = cc_weight

        self.predictor_name = predictor_name
        self.mppi_samples = int(self.mppi_horizon / dt)  # Number of steps in MPC horizon

        self.R = tf.convert_to_tensor(R)
        self.LBD = LBD
        self.NU = tf.convert_to_tensor(NU)
        self.SQRTRHODTINV = tf.convert_to_tensor(SQRTRHOINV * (1 / np.math.sqrt(dt)))
        self.GAMMA = GAMMA
        self.SAMPLING_TYPE = SAMPLING_TYPE

        if isinstance(clip_control_input[0], list):
            self.clip_control_input_low = tf.constant(clip_control_input[0], dtype=tf.float32)
            self.clip_control_input_high = tf.constant(clip_control_input[1], dtype=tf.float32)
        else:
            self.clip_control_input_high = tf.constant(clip_control_input, dtype=tf.float32)
            self.clip_control_input_low = -self.clip_control_input_high

        #instantiate predictor
        predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
        self.predictor = getattr(predictor_module, predictor_name)(
            horizon=self.mppi_samples,
            dt=dt,
            intermediate_steps=predictor_intermediate_steps,
            disable_individual_compilation=True,
            batch_size=num_rollouts,
            net_name=NET_NAME,
        )
        if predictor_name == "predictor_autoregressive_tf":
            self.predictor_single_trajectory = getattr(predictor_module, predictor_name)(
            horizon=self.mppi_samples,
            dt=dt,
            intermediate_steps=predictor_intermediate_steps,
            disable_individual_compilation=True,
            batch_size=1,
            net_name=NET_NAME,
        )
        else:
            self.predictor_single_trajectory = self.predictor

        self.get_rollouts_from_mppi = True
        self.get_optimal_trajectory = False

        self.u_nom = tf.zeros([1, self.mppi_samples, num_control_inputs], dtype=tf.float32)
        self.u = tf.convert_to_tensor([0.0], dtype=tf.float32)

        self.rollout_trajectory = None
        self.traj_cost = None

        self.optimal_trajectory = None

        # Defining function - the compiled part must not have if-else statements with changing output dimensions
        if predictor_name == 'predictor_autoregressive_tf':
            self.update_internal_state = self.update_internal_state_of_RNN
        else:
            self.update_internal_state = lambda s, u_nom: ...

        if self.get_rollouts_from_mppi:
            self.mppi_output = self.return_all
        else:
            self.mppi_output = self.return_restricted

    def return_all(self, u, u_nom, rollout_trajectory, traj_cost, u_run):
        return u, u_nom, rollout_trajectory, traj_cost, u_run

    def return_restricted(self, u, u_nom, rollout_trajectory, traj_cost, u_run):
        return u, u_nom, None, None, None

    def check_dimensions_s(self, s):
        # Make sure the input is at least 2d
        if tf.rank(s) == 1:
            s = s[tf.newaxis, :]
        return s

    #mppi correction
    def mppi_correction_cost(self, u, delta_u):
        return tf.math.reduce_sum(self.cc_weight * (0.5 * (1 - 1.0 / self.NU) * self.R * (delta_u ** 2) + self.R * u * delta_u + 0.5 * self.R * (u ** 2)), axis=2)

    #total cost of the trajectory
    def cost(self, s_hor ,u, target, u_prev, delta_u):
        stage_cost = self.q(s_hor[:,1:,:],u,target, u_prev)
        stage_cost = stage_cost + self.mppi_correction_cost(u, delta_u)
        total_cost = tf.math.reduce_sum(stage_cost,axis=1)
        total_cost = total_cost + self.phi(s_hor, target)
        return total_cost

    def reward_weighted_average(self, S, delta_u):
        rho = tf.math.reduce_min(S)
        exp_s = tf.exp(-1.0/self.LBD * (S-rho))
        a = tf.math.reduce_sum(exp_s)
        b = tf.math.reduce_sum(exp_s[:, tf.newaxis, tf.newaxis]*delta_u, axis=0)/a
        return b

    def inizialize_pertubation(self, random_gen):
        stdev = self.SQRTRHODTINV
        sampling_type = self.SAMPLING_TYPE
        if sampling_type == "interpolated":
            step = 10
            range_stop = int(tf.math.ceil(self.mppi_samples / step)*step) + 1
            t = tf.range(range_stop, delta = step)
            t_interp = tf.cast(tf.range(range_stop), tf.float32)
            delta_u = random_gen.normal([self.num_rollouts, t.shape[0], self.num_control_inputs], dtype=tf.float32) * stdev
            interp = tfp.math.interp_regular_1d_grid(t_interp, t_interp[0], t_interp[-1], delta_u, axis=1)
            delta_u = interp[:,:self.mppi_samples, :]
        else:
            delta_u = random_gen.normal([self.num_rollouts, self.mppi_samples, self.num_control_inputs], dtype=tf.float32) * stdev
        return delta_u

    @Compile
    def predict_and_cost(self, s, target, u_nom, random_gen, u_old):
        s = tf.tile(s, tf.constant([self.num_rollouts, 1]))
        # generate random input sequence and clip to control limits
        u_nom = tf.concat([u_nom[:, 1:, :], u_nom[:, -1:, :]], axis=1)
        delta_u = self.inizialize_pertubation(random_gen)
        u_run = tf.tile(u_nom, [self.num_rollouts, 1, 1])+delta_u
        u_run = tf.clip_by_value(u_run, self.clip_control_input_low, self.clip_control_input_high)
        rollout_trajectory = self.predictor.predict_tf(s, u_run)
        traj_cost = self.cost(rollout_trajectory, u_run, target, u_old, delta_u)
        u_nom = tf.clip_by_value(u_nom + self.reward_weighted_average(traj_cost, delta_u), self.clip_control_input_low, self.clip_control_input_high)
        u = u_nom[0, 0, :]
        self.update_internal_state(s, u_nom)
        return self.mppi_output(u, u_nom, rollout_trajectory, traj_cost, u_run)

    def update_internal_state_of_RNN(self, s, u_nom):
        u_tiled = tf.tile(u_nom[:, :1, :], tf.constant([self.num_rollouts, 1, 1]))
        self.predictor.update_internal_state_tf(s=s, Q0=u_tiled)

    @Compile
    def predict_optimal_trajectory(self, s, u_nom):
        optimal_trajectory = self.predictor_single_trajectory.predict_tf(s, u_nom)
        if self.predictor_name ==  'predictor_autoregressive_tf':
            self.predictor_single_trajectory.update_internal_state_tf(s=s, Q0=u_nom[:, :1, :])
        return optimal_trajectory

    #step function to find control
    def step(self, s: np.ndarray, target: np.ndarray, time=None):
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        s = self.check_dimensions_s(s)
        target = tf.convert_to_tensor(target, dtype=tf.float32)

        self.u, self.u_nom, rollout_trajectory, traj_cost, u_run = self.predict_and_cost(s, target, self.u_nom, self.rng_mppi,
                                                                                  self.u)
        self.Q, self.J = u_run.numpy(), traj_cost.numpy()

        if self.get_rollouts_from_mppi:
            self.rollout_trajectory = rollout_trajectory.numpy()
            self.traj_cost = traj_cost.numpy()

        if self.get_optimal_trajectory:
            self.optimal_trajectory = self.predict_optimal_trajectory(s, self.u_nom).numpy()

        return tf.squeeze(self.u).numpy()

    def controller_reset(self):
        self.u_nom = tf.zeros([1, self.mppi_samples, self.num_control_inputs], dtype=tf.float32)
        self.u = 0.0


if __name__ == '__main__':
    ctrl = controller_mppi_tf()

    import timeit

    s0 = create_cartpole_state()
    # Set non-zero input
    s = s0
    s[POSITION_IDX] = -30.2
    s[POSITIOND_IDX] = 2.87
    s[ANGLE_IDX] = -0.32
    s[ANGLED_IDX] = 0.237
    u = -0.24

    ctrl.step(s0, 0.0)
    f_to_measure = 'ctrl.step(s0,0.0)'
    number = 1  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 1000  # Gives how many times timeit should be repeated
    timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
    min_time = min(timings) / float(number)
    max_time = max(timings) / float(number)
    average_time = np.mean(timings) / float(number)
    print()
    print('----------------------------------------------------------------------------------')
    print('Min time to evaluate is {} ms'.format(min_time * 1.0e3))  # ca. 5 us
    print('Average time to evaluate is {} ms'.format(average_time * 1.0e3))  # ca 5 us
    # The max is of little relevance as it is heavily influenced by other processes running on the computer at the same time
    print('Max time to evaluate is {} ms'.format(max_time * 1.0e3))  # ca. 100 us
    print('----------------------------------------------------------------------------------')
    print()
