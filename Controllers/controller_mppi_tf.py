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
from SI_Toolkit.Predictors.predictor_ODE_tf_pure import predictor_ODE_tf_pure
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
from SI_Toolkit.Predictors.predictor_autoregressive_GP_Euler import predictor_autoregressive_GP
from SI_Toolkit.Predictors.predictor_hybrid import predictor_hybrid

#load constants from config file
config = yaml.load(open("CartPoleSimulation/config.yml", "r"), Loader=yaml.FullLoader)

dt = config["controller"]["mppi"]["dt"]
mppi_horizon = config["controller"]["mppi"]["mpc_horizon"]
num_rollouts = config["controller"]["mppi"]["num_rollouts"]
dd_weight = config["controller"]["mppi"]["dd_weight"]
cc_weight = config["controller"]["mppi"]["cc_weight"]
ep_weight = config["controller"]["mppi"]["ep_weight"]

NET_NAME = config["controller"]["mppi"]["NET_NAME"]
predictor_type = config["controller"]["mppi"]["predictor_type"]
ccrc_weight = config["controller"]["mppi"]["ccrc_weight"]

mppi_samples = int(mppi_horizon / dt)  # Number of steps in MPC horizon

R = config["controller"]["mppi"]["R"]
LBD = config["controller"]["mppi"]["LBD"]
NU = config["controller"]["mppi"]["NU"]
SQRTRHODTINV = config["controller"]["mppi"]["SQRTRHOINV"] * (1 / np.math.sqrt(dt))
GAMMA = config["controller"]["mppi"]["GAMMA"]
SAMPLING_TYPE = config["controller"]["mppi"]["SAMPLING_TYPE"]

#create predictor
predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)

"""Define Predictor"""
if predictor_type == "EulerTF":
    predictor = predictor_ODE_tf_pure(horizon=mppi_samples, dt=dt, intermediate_steps=10)
elif predictor_type == "Euler":
    predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)
elif predictor_type == "NeuralNet":
    predictor = predictor_autoregressive_tf(
        horizon=mppi_samples, batch_size=num_rollouts, net_name=NET_NAME
    )
elif predictor_type == "GP":
    predictor = predictor_autoregressive_GP(horizon=mppi_samples, num_rollouts=num_rollouts)
elif predictor_type == "Hybrid":
    predictor = predictor_hybrid(horizon=mppi_samples, dt=dt, intermediate_steps=10, batch_size=num_rollouts, net_name=NET_NAME)

#cost for distance from track edge
def distance_difference_cost(position, target_position):
    """Compute penalty for distance of cart to the target position"""
    return ((position - target_position) / (2.0 * TrackHalfLength)) ** 2 + tf.cast(
        tf.abs(position) > 0.80 * TrackHalfLength
    , tf.float32) * 1.0e7  # Soft constraint: Do not crash into border

#cost for difference from upright position
def E_pot_cost(angle):
    """Compute penalty for not balancing pole upright (penalize large angles)"""
    return 0.25 * (1.0 - tf.cos(angle)) ** 2

#actuation cost
def CC_cost(u):
    return R * (u ** 2)

#final stage cost
def phi(s, target_position):
    """Calculate terminal cost of a set of trajectories

    Williams et al use an indicator function type of terminal cost in
    "Information theoretic MPC for model-based reinforcement learning"

    TODO: Try a quadratic terminal cost => Use the LQR terminal cost term obtained
    by linearizing the system around the unstable equilibrium.

    :param s: Reference to numpy array of states of all rollouts
    :type s: np.ndarray
    :param target_position: Target position to move the cart to
    :type target_position: np.float32
    :return: One terminal cost per rollout
    :rtype: np.ndarray
    """
    terminal_states = s[:, -1, :]
    terminal_cost = 10000 * tf.cast(
        (tf.abs(terminal_states[:, ANGLE_IDX]) > 0.2)
        | (
            tf.abs(terminal_states[:, POSITION_IDX] - target_position)
            > 0.1 * TrackHalfLength
        )
    , tf.float32)
    return terminal_cost

#optimized mean function
def mean_numba(a):

    res = []
    for i in prange(a.shape[0]):
        res.append(a[i, :].mean())

    return np.array(res)

#cost of changeing control to fast
def control_change_rate_cost(u, u_prev,nrol):
    """Compute penalty of control jerk, i.e. difference to previous control input"""
    u_prev_vec = tf.concat((tf.ones((nrol, 1))*u_prev,u[:,:-1]),axis=-1)
    return (u - u_prev_vec) ** 2

#all stage costs together
def q(s,u,target_position, u_prev, nrol = num_rollouts):
    dd = dd_weight * distance_difference_cost(
        s[:, :, POSITION_IDX], target_position
    )
    ep = ep_weight * E_pot_cost(s[:, :, ANGLE_IDX])
    cc = cc_weight * CC_cost(u)
    ccrc = ccrc_weight * control_change_rate_cost(u,u_prev, nrol)
    stage_cost = dd+ep+cc+ccrc
    return stage_cost


#mppi correction
def mppi_correction_cost(u, delta_u):
    return cc_weight * (0.5 * (1 - 1.0 / NU) * R * (delta_u ** 2) + R * u * delta_u + 0.5 * R * (u ** 2))

#total cost of the trajectory
def cost(s_hor ,u, target_position, u_prev, delta_u):
    stage_cost = q(s_hor[:,1:,:],u,target_position, u_prev)
    stage_cost = stage_cost + mppi_correction_cost(u, delta_u)
    total_cost = tf.math.reduce_sum(stage_cost,axis=1)
    total_cost = total_cost + phi(s_hor, target_position)
    return total_cost


def reward_weighted_average(S, delta_u):
    rho = tf.math.reduce_min(S)
    exp_s = tf.exp(-1.0/LBD * (S-rho))
    a = tf.math.reduce_sum(exp_s)
    b = tf.math.reduce_sum(exp_s[:,tf.newaxis]*delta_u, axis = 0, keepdims=True)/a
    return b

def inizialize_pertubation(random_gen, stdev = SQRTRHODTINV, sampling_type = SAMPLING_TYPE):
    if sampling_type == "interpolated":
        step = 10
        range_stop = int(tf.math.ceil(mppi_samples / step)*step) + 1
        t = tf.range(range_stop, delta = step)
        t_interp = tf.cast(tf.range(range_stop), tf.float32)
        delta_u = random_gen.normal([num_rollouts, t.shape[0]], dtype=tf.float32) * stdev
        interp = tfp.math.interp_regular_1d_grid(t_interp, t_interp[0], t_interp[-1], delta_u)
        delta_u = interp[:,:mppi_samples]
    else:
        delta_u = random_gen.normal([num_rollouts, mppi_samples], dtype=tf.float32) * stdev
    return delta_u



#cem class
class controller_mppi_tf(template_controller):
    def __init__(self):
        #First configure random sampler
        SEED = config["controller"]["mppi"]["SEED"]
        if SEED == "None":
            SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)
        self.rng_cem = tf.random.Generator.from_seed(SEED)

        self.u_nom = tf.zeros([1,mppi_samples], dtype=tf.float32)
        self.u = 0.0

        self.num_rollouts = num_rollouts
        self.horizon = mppi_horizon

    @tf.function(jit_compile=True)
    def predict_and_cost(self, s, target_position, u_nom, random_gen, u_old):
        # generate random input sequence and clip to control limits
        delta_u = inizialize_pertubation(random_gen)
        u_run = tf.tile(u_nom, [num_rollouts, 1])+delta_u
        u_run = tf.clip_by_value(u_run, -1.0, 1.0)
        rollout_trajectory = predictor.predict_tf(s, u_run[:, :, tf.newaxis])
        # print(rollout_trajectory[0,0,0])
        traj_cost = cost(rollout_trajectory, u_run, target_position, u_old, delta_u)
        u_nom = tf.clip_by_value(u_nom + reward_weighted_average(traj_cost, delta_u), -1.0, 1.0)
        u = u_nom[0, 0]
        u_nom = tf.concat([u_nom[:, 1:], tf.constant(0.0, shape=[1, 1])], -1)
        return u, u_nom

    #step function to find control
    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        s = np.tile(s, tf.constant([num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        target_position = tf.convert_to_tensor(target_position, dtype=tf.float32)

        self.u, self.u_nom = self.predict_and_cost(s, target_position, self.u_nom, self.rng_cem, self.u)
        return self.u.numpy()

    def controller_reset(self):
        self.u_nom = tf.zeros([1, mppi_samples], dtype=tf.float32)
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