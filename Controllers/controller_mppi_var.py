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


#load constants from config file
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

cost_function = config["controller"]["general"]["cost_function"]
cost_function = cost_function.replace('-', '_')
cost_function_cmd = 'from others.cost_functions.'+cost_function+' import q, phi'
exec(cost_function_cmd)

dt = config["controller"]["mppi"]["dt"]
mppi_horizon = config["controller"]["mppi"]["mpc_horizon"]
num_rollouts = config["controller"]["mppi"]["num_rollouts"]
SAMPLING_TYPE = config["controller"]["mppi"]["SAMPLING_TYPE"]

cc_weight = config["controller"]["mppi"]["cc_weight"]

NET_NAME = config["controller"]["cem"]["CEM_NET_NAME"]
predictor_type = config["controller"]["cem"]["cem_predictor_type"]

mppi_samples = int(mppi_horizon / dt)  # Number of steps in MPC horizon

R = config["controller"]["mppi"]["R"]
LBD = config["controller"]["mppi"]["LBD_mc"]
NU = config["controller"]["mppi"]["NU_mc"]
SQRTRHODTINV = config["controller"]["mppi"]["SQRTRHOINV_mc"] * (1 / np.math.sqrt(dt))
GAMMA = config["controller"]["mppi"]["GAMMA"]

mppi_lr = config["controller"]["mppi-grad"]["LR"]
stdev_min = config["controller"]["mppi-grad"]["STDEV_min"]
stdev_max = config["controller"]["mppi-grad"]["STDEV_max"]
max_grad_norm = config["controller"]["mppi-grad"]["max_grad_norm"]

#create predictor
predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)

"""Define Predictor"""
if predictor_type == "EulerTF":
    predictor = predictor_ODE_tf_pure(horizon=mppi_samples, dt=dt, intermediate_steps=1)
elif predictor_type == "Euler":
    predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)
elif predictor_type == "NeuralNet":
    predictor = predictor_autoregressive_tf(
        horizon=mppi_samples, batch_size=num_rollouts, net_name=NET_NAME
    )


#setup interpolation matrix
if SAMPLING_TYPE == "interpolated":
    step = 10
    num_valid_vals = int(np.ceil(mppi_samples / step) + 1)
    interp_mat = np.zeros(((num_valid_vals - 1) * step, num_valid_vals))
    step_block = np.zeros((step, 2))
    for j in range(step):
        step_block[j][0] = step - j
        step_block[j][1] = j
    for i in range(num_valid_vals - 1):
        interp_mat[i * step:(i + 1) * step, i:i + 2] = step_block
    interp_mat = interp_mat[:mppi_samples, :] / step
    interp_mat = tf.constant(interp_mat.T, dtype=tf.float32)
else:
    interp_mat = None
    num_valid_vals = mppi_samples

#mppi correction
def mppi_correction_cost(u, delta_u, nuvec):
    if SAMPLING_TYPE == "interpolated":
        nudiv = tf.matmul(nuvec, interp_mat)
    else:
        nudiv = nuvec
    return cc_weight * (0.5 * (1 - 1.0 / nudiv**2) * R * (delta_u ** 2) + R * u * delta_u + 0.5 * R * (u ** 2))

#total cost of the trajectory
def uncorr_cost(s_hor ,u, target_position, u_prev, delta_u):
    stage_cost = q(s_hor[:,1:,:],u,target_position, u_prev)
    stage_cost = stage_cost
    unc_cost = tf.math.reduce_sum(stage_cost,axis=1)
    unc_cost = unc_cost + phi(s_hor, target_position)
    return unc_cost


def reward_weighted_average(S, delta_u):
    rho = tf.math.reduce_min(S)
    exp_s = tf.exp(-1.0/LBD * (S-rho))
    a = tf.math.reduce_sum(exp_s)
    b = tf.math.reduce_sum(exp_s[:,tf.newaxis]*delta_u, axis = 0, keepdims=True)/a
    return b

def inizialize_pertubation(random_gen, nuvec ):
    delta_u = random_gen.normal([num_rollouts, num_valid_vals], dtype=tf.float32) * nuvec*SQRTRHODTINV
    if SAMPLING_TYPE == "interpolated":
        delta_u = tf.matmul(delta_u, interp_mat)
    return delta_u



#controller class
class controller_mppi_var(template_controller):
    def __init__(self):
        #First configure random sampler
        SEED = config["controller"]["mppi"]["SEED"]
        if SEED == "None":
            SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)
        self.rng_cem = tf.random.Generator.from_seed(SEED)
        self.u_nom = tf.zeros([1,mppi_samples], dtype=tf.float32)
        self.nuvec = np.math.sqrt(NU)*tf.ones([1, num_valid_vals])
        self.nuvec = tf.Variable(self.nuvec)
        self.u = 0.0

    @tf.function(jit_compile=True)
    def predict_and_cost(self, s, target_position, u_nom, random_gen, u_old, nuvec):
        # generate random input sequence and clip to control limits
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(nuvec)
            delta_u = inizialize_pertubation(random_gen, nuvec)
            u_run = tf.tile(u_nom, [num_rollouts, 1])+delta_u
            u_run = tfp.math.clip_by_value_preserve_gradient(u_run, -1.0, 1.0)
            rollout_trajectory = predictor.predict_tf(s, u_run[:, :, tf.newaxis])
            unc_cost = uncorr_cost(rollout_trajectory, u_run, target_position, u_old, delta_u)
            mean_uncost = tf.math.reduce_mean(unc_cost)
            dc_ds = tape.gradient(mean_uncost, nuvec)
            dc_ds = tf.clip_by_norm(dc_ds, max_grad_norm,axes = [1])
        cor_cost = mppi_correction_cost(u_run, delta_u, nuvec)
        cor_cost = tf.math.reduce_sum(cor_cost, axis=1)
        traj_cost = unc_cost + cor_cost
        u_nom = tf.clip_by_value(u_nom + reward_weighted_average(traj_cost, delta_u), -1.0, 1.0)
        u = u_nom[0, 0]
        u_nom = tf.concat([u_nom[:, 1:], tf.constant(0.0, shape=[1, 1])], -1)
        new_nuvec = nuvec-mppi_lr*dc_ds
        new_nuvec = tf.clip_by_value(new_nuvec, stdev_min, stdev_max)
        return u, u_nom, new_nuvec

    #step function to find control
    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        s = np.tile(s, tf.constant([num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        target_position = tf.convert_to_tensor(target_position, dtype=tf.float32)
        self.u, self.u_nom, new_nuvec = self.predict_and_cost(s, target_position, self.u_nom, self.rng_cem, self.u, self.nuvec)
        self.nuvec.assign(new_nuvec)
        # print("eyo here i go")
        return self.u.numpy()

    def controller_reset(self):
        self.u_nom = tf.zeros([1, mppi_samples], dtype=tf.float32)
        self.nuvec.assign(np.math.sqrt(NU)*tf.ones([1, num_valid_vals]))
        self.u = 0.0




if __name__ == '__main__':
    ctrl = controller_mppi_var()


    import timeit

    s0 = create_cartpole_state()
    # Set non-zero input
    s = s0
    s[POSITION_IDX] = -0.12
    s[POSITIOND_IDX] = 0.3
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