import scipy
import numpy as np
from numpy.random import SFC64, Generator
from datetime import datetime
from numba import jit, prange
import tensorflow as tf
import tensorflow_probability as tfp

from Controllers.template_controller import template_controller
from CartPole.cartpole_model import TrackHalfLength

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state
from CartPole.cartpole_model import u_max, s0
from CartPole.cartpole_jacobian import cartpole_jacobian

import yaml

from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf

from SI_Toolkit.TF.TF_Functions.Compile import Compile

#load constants from config file
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

num_control_inputs = 1  # specific to a system

q, phi = None, None
cost_function = config["controller"]["general"]["cost_function"]
cost_function = cost_function.replace('-', '_')
cost_function_cmd = 'from others.cost_functions.'+cost_function+' import q, phi'
exec(cost_function_cmd)

dt = config["controller"]["mppi"]["dt"]
mppi_horizon = config["controller"]["mppi"]["mpc_horizon"]
num_rollouts = config["controller"]["mppi"]["num_rollouts"]

cc_weight = config["controller"]["mppi"]["cc_weight"]

NET_NAME = config["controller"]["mppi"]["NET_NAME"]
predictor_type = config["controller"]["mppi"]["predictor_type"]

mppi_samples = int(mppi_horizon / dt)  # Number of steps in MPC horizon

R = tf.convert_to_tensor(config["controller"]["mppi"]["R"])
LBD = config["controller"]["mppi"]["LBD"]
NU = tf.convert_to_tensor(config["controller"]["mppi"]["NU"])
SQRTRHODTINV = tf.convert_to_tensor(config["controller"]["mppi"]["SQRTRHOINV"]) * tf.convert_to_tensor((1 / np.math.sqrt(dt)))
GAMMA = config["controller"]["mppi"]["GAMMA"]
SAMPLING_TYPE = config["controller"]["mppi"]["SAMPLING_TYPE"]

clip_control_input = tf.constant(config["controller"]["mppi"]["CLIP_CONTROL_INPUT"], dtype=tf.float32)

#create predictor
predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)

"""Define Predictor"""
if predictor_type == "EulerTF":
    predictor = predictor_ODE_tf(horizon=mppi_samples, dt=dt, intermediate_steps=10, disable_individual_compilation=True)
    predictor_single_trajectory = predictor
elif predictor_type == "Euler":
    predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)
    predictor_single_trajectory = predictor
elif predictor_type == "NeuralNet":
    predictor = predictor_autoregressive_tf(
        horizon=mppi_samples, batch_size=num_rollouts, net_name=NET_NAME, disable_individual_compilation=True
    )
    predictor_single_trajectory = predictor_autoregressive_tf(
        horizon=mppi_samples, batch_size=1, net_name=NET_NAME, disable_individual_compilation=True
    )

GET_ROLLOUTS_FROM_MPPI = False

GET_OPTIMAL_TRAJECTORY = False

def check_dimensions_s(s):
    # Make sure the input is at least 2d
    if tf.rank(s) == 1:
        s = s[tf.newaxis, :]

    return s

#mppi correction
def mppi_correction_cost(u, delta_u):
    return tf.math.reduce_sum(cc_weight * (0.5 * (1 - 1.0 / NU) * R * (delta_u ** 2) + R * u * delta_u + 0.5 * R * (u ** 2)), axis=-1)

#total cost of the trajectory
def cost(s_hor ,u, target, u_prev, delta_u):
    stage_cost = q(s_hor[:,1:,:],u,target, u_prev)
    stage_cost = stage_cost + mppi_correction_cost(u, delta_u)
    total_cost = tf.math.reduce_sum(stage_cost,axis=1)
    total_cost = total_cost + phi(s_hor, target)
    return total_cost


def reward_weighted_average(S, delta_u):
    rho = tf.math.reduce_min(S)
    exp_s = tf.exp(-1.0/LBD * (S-rho))
    a = tf.math.reduce_sum(exp_s)
    b = tf.math.reduce_sum(exp_s[:, tf.newaxis, tf.newaxis]*delta_u, axis=0)/a
    return b

def inizialize_pertubation(random_gen, stdev = SQRTRHODTINV, sampling_type = SAMPLING_TYPE):
    if sampling_type == "interpolated":
        step = 10
        range_stop = int(tf.math.ceil(mppi_samples / step)*step) + 1
        t = tf.range(range_stop, delta = step)
        t_interp = tf.cast(tf.range(range_stop), tf.float32)
        delta_u = random_gen.normal([num_rollouts, t.shape[0], num_control_inputs], dtype=tf.float32) * stdev
        interp = tfp.math.interp_regular_1d_grid(t_interp, t_interp[0], t_interp[-1], delta_u, axis=1)
        delta_u = interp[:,:mppi_samples, :]
    else:
        delta_u = random_gen.normal([num_rollouts, mppi_samples, num_control_inputs], dtype=tf.float32) * stdev
    return delta_u



class controller_mppi_tf(template_controller):
    def __init__(self):
        #First configure random sampler
        SEED = config["controller"]["mppi"]["SEED"]
        if SEED == "None":
            SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)
        self.rng_cem = tf.random.Generator.from_seed(SEED)

        self.u_nom = tf.zeros([1, mppi_samples, num_control_inputs], dtype=tf.float32)
        self.u = tf.convert_to_tensor([0.0], dtype=tf.float32)

        self.rollout_trajectory = None
        self.traj_cost = None

        self.optimal_trajectory = None

    @Compile
    def predict_and_cost(self, s, target, u_nom, random_gen, u_old):
        s = check_dimensions_s(s)
        s = tf.tile(s, tf.constant([num_rollouts, 1]))
        # generate random input sequence and clip to control limits
        u_nom = tf.concat([u_nom[:, 1:, :], u_nom[:, -1, tf.newaxis, :]], axis=1)
        delta_u = inizialize_pertubation(random_gen)
        u_run = tf.tile(u_nom, [num_rollouts, 1, 1])+delta_u
        u_run = tf.clip_by_value(u_run, -clip_control_input, clip_control_input)
        rollout_trajectory = predictor.predict_tf(s, u_run)
        traj_cost = cost(rollout_trajectory, u_run, target, u_old, delta_u)
        u_nom = tf.clip_by_value(u_nom + reward_weighted_average(traj_cost, delta_u), -clip_control_input, clip_control_input)
        u = u_nom[0, 0, :]
        if predictor_type ==  'NeuralNet':
            u_tiled = tf.tile(u_nom[:, :1, :], tf.constant([num_rollouts, 1, 1]))
            predictor.update_internal_state_tf(s=s, Q0=u_tiled)
        if GET_ROLLOUTS_FROM_MPPI:
            return u, u_nom, rollout_trajectory, traj_cost
        else:
            return u, u_nom, None, None

    @Compile
    def predict_optimal_trajectory(self, s, u_nom):
        s = check_dimensions_s(s)
        optimal_trajectory = predictor_single_trajectory.predict_tf(s, u_nom)
        if predictor_type ==  'NeuralNet':
            predictor_single_trajectory.update_internal_state_tf(s=s, Q0=u_nom[:, :1, :])
        return optimal_trajectory

    #step function to find control
    def step(self, s: np.ndarray, target: np.ndarray, time=None):
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.float32)

        self.u, self.u_nom, rollout_trajectory, traj_cost = self.predict_and_cost(s, target, self.u_nom, self.rng_cem,
                                                                                  self.u)
        if GET_ROLLOUTS_FROM_MPPI:
            self.rollout_trajectory = rollout_trajectory.numpy()
            self.traj_cost = traj_cost.numpy()

        if GET_OPTIMAL_TRAJECTORY:
            self.optimal_trajectory = self.predict_optimal_trajectory(s, self.u_nom).numpy()

        return self.u.numpy()

    def controller_reset(self):
        self.u_nom = tf.zeros([1, mppi_samples, num_control_inputs], dtype=tf.float32)
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