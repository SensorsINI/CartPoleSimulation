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
cost_function_cmd = 'from others.cost_functions.'+cost_function+' import q, phi, cost'
exec(cost_function_cmd)

dt = config["controller"]["mppi-optimize"]["dt"]
mppi_horizon = config["controller"]["mppi-optimize"]["mpc_horizon"]
num_rollouts = config["controller"]["mppi-optimize"]["num_rollouts"]

cc_weight = config["controller"]["mppi-optimize"]["cc_weight"]

NET_NAME = config["controller"]["mppi-optimize"]["NET_NAME"]
predictor_type = config["controller"]["mppi-optimize"]["predictor_type"]

mppi_samples = int(mppi_horizon / dt)  # Number of steps in MPC horizon

R = config["controller"]["mppi-optimize"]["R"]
LBD = config["controller"]["mppi-optimize"]["LBD"]
NU = config["controller"]["mppi-optimize"]["NU"]
SQRTRHODTINV = config["controller"]["mppi-optimize"]["SQRTRHOINV"] * (1 / np.math.sqrt(dt))
GAMMA = config["controller"]["mppi-optimize"]["GAMMA"]
SAMPLING_TYPE = config["controller"]["mppi-optimize"]["SAMPLING_TYPE"]

cem_LR = config["controller"]["mppi-optimize"]["LR"]

cem_LR = tf.constant(cem_LR, dtype=tf.float32)

adam_beta_1 = config["controller"]["mppi-optimize"]["adam_beta_1"]
adam_beta_2 = config["controller"]["mppi-optimize"]["adam_beta_2"]
adam_epsilon = float(config["controller"]["mppi-optimize"]["adam_epsilon"])
gradmax_clip = config["controller"]["mppi-optimize"]["gradmax_clip"]
gradmax_clip = tf.constant(gradmax_clip, dtype = tf.float32)
optim_steps = config["controller"]["mppi-optimize"]["optim_steps"]

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

#mppi correction
def mppi_correction_cost(u, delta_u):
    return cc_weight * (0.5 * (1 - 1.0 / NU) * R * (delta_u ** 2) + R * u * delta_u + 0.5 * R * (u ** 2))

#total cost of the trajectory
def mppi_cost(s_hor ,u, target_position, u_prev, delta_u):
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
class controller_mppi_optimize(template_controller):
    def __init__(self):
        #First configure random sampler
        SEED = config["controller"]["mppi"]["SEED"]
        if SEED == "None":
            SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)
        self.rng_cem = tf.random.Generator.from_seed(SEED)

        self.Q = tf.zeros([1,mppi_samples], dtype=tf.float32)
        self.Q = tf.Variable(self.Q)
        self.u = 0.0
        self.opt = tf.keras.optimizers.Adam(learning_rate=cem_LR, beta_1=adam_beta_1, beta_2=adam_beta_2,
                                            epsilon=adam_epsilon)

    @tf.function(jit_compile=True)
    def mppi_prior(self, s, target_position, u_nom, random_gen, u_old):
        # generate random input sequence and clip to control limits
        delta_u = inizialize_pertubation(random_gen)
        u_run = tf.tile(u_nom, [num_rollouts, 1])+delta_u
        u_run = tf.clip_by_value(u_run, -1.0, 1.0)
        rollout_trajectory = predictor.predict_tf(s, u_run[:, :, tf.newaxis])
        traj_cost = mppi_cost(rollout_trajectory, u_run, target_position, u_old, delta_u)
        u_nom = tf.clip_by_value(u_nom + reward_weighted_average(traj_cost, delta_u), -1.0, 1.0)
        return u_nom

    @tf.function(jit_compile=True)
    def grad_step(self, s, target_position, Q, opt):
        #do a gradient descent step
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = predictor.predict_tf(s, Q[:, :, tf.newaxis])
            traj_cost = cost(rollout_trajectory, Q, target_position, self.u)
        dc_dQ = tape.gradient(traj_cost, Q)
        dc_dQ_max = tf.math.reduce_max(tf.abs(dc_dQ), axis=1)
        mask = (dc_dQ_max > gradmax_clip)[:, tf.newaxis]
        invmask = tf.logical_not(mask)
        dc_dQ_prc = ((dc_dQ / dc_dQ_max[:, tf.newaxis]) * tf.cast(mask, tf.float32) * gradmax_clip + dc_dQ * tf.cast(
            invmask, tf.float32))
        opt.apply_gradients(zip([dc_dQ_prc], [Q]))
        Qn = tf.clip_by_value(Q, -1, 1)
        return Qn

    #step function to find control
    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        s = np.tile(s, tf.constant([num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        target_position = tf.convert_to_tensor(target_position, dtype=tf.float32)

        Q_mppi = self.mppi_prior(s, target_position, self.Q, self.rng_cem, self.u)
        self.Q.assign(Q_mppi)

        for _ in range(optim_steps):
            Q_opt = self.grad_step(s, target_position, self.Q, self.opt)
            self.Q.assign(Q_opt)

        self.u = self.Q[0, 0]
        self.Q.assign(tf.concat([self.Q[:, 1:], tf.zeros([1,1])], -1))
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])
        return self.u.numpy()

    def controller_reset(self):
        self.Q.assign(tf.zeros([1, mppi_samples], dtype=tf.float32))
        self.u = 0.0
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])




if __name__ == '__main__':
    ctrl = controller_mppi_optimize()


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