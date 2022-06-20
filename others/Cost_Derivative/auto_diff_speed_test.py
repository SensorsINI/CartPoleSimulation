import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SFC64, Generator
import math

from Controllers.template_controller import template_controller
from CartPole.cartpole_model import TrackHalfLength

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, \
    create_cartpole_state
from CartPole.cartpole_model import u_max, s0
from CartPole.cartpole_jacobian import cartpole_jacobian

import yaml

from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
from SI_Toolkit.Predictors.predictor_ODE_tf_pure import predictor_ODE_tf_pure
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
import tensorflow as tf
from others.cost_functions.quadratic_boundary import cost
import tensorflow_probability as tfp

tf.config.run_functions_eagerly(False)

import contextlib

@contextlib.contextmanager
def tf_options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


@tf.function(jit_compile=True)
def grad_calc(u, s, uprev):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(u)
        rollout_trajectory = predictor.predict_tf(s, u[:, :, tf.newaxis])
        realc = cost(rollout_trajectory,u,0.0,uprev)
    dc_du = tape.gradient(realc, u)
    u_new = u+0.01*dc_du
    out = realc+1
    return u_new, out

@tf.function(jit_compile=True)
def rollout_calc(u, s, uprev):
    rollout_trajectory = predictor.predict_tf(s, u[:, :, tf.newaxis])
    realc = cost(rollout_trajectory,u,0.0,uprev)
    u_new = u + 0.01 * u
    out = realc + 1
    return realc, out



# load constants from config file
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
dt = config["controller"]["mppi"]["dt"]
cem_horizon = config["controller"]["mppi"]["mpc_horizon"]
cem_samples = int(cem_horizon / dt)  # Number of steps in MPC horizon
predictor = predictor_ODE_tf_pure(horizon=cem_samples, dt=dt, intermediate_steps=1)

s0 = create_cartpole_state()
# Set non-zero input
s = s0
s[POSITION_IDX] = 0.0
s[POSITIOND_IDX] = 0
s[ANGLE_IDX] = 0 #math.pi
s[ANGLED_IDX] = 0
uprev = 0.14

s_org = tf.convert_to_tensor(s)


SEED = 5876
horizon = cem_samples
rng_gen = Generator(SFC64(SEED))
dist_var = 0.3 * np.ones([1, horizon])
stdev = np.sqrt(dist_var)
num_rollouts = 200
dist_mue = np.zeros([1, horizon])
Q = np.tile(dist_mue, (num_rollouts, 1)) + np.multiply(rng_gen.standard_normal(
    size=(num_rollouts, horizon), dtype=np.float32), stdev)
Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)

s = s_org
s = np.tile(s, tf.constant([num_rollouts, 1]))
u = tf.convert_to_tensor(Q)
u = u
s_start = predictor.predict_tf(s, u[:, :, tf.newaxis])
lr = 5
grad_calc(u, s, uprev)
rollout_calc(u, s, uprev)
with tf_options({}):
    print(tf.config.optimizer.get_experimental_options())

    import timeit

    ro_calc_meas = 'rollout_calc(u, s, uprev)'
    number = 100 # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 100  # Gives how many times timeit should be repeated
    timings = timeit.Timer(ro_calc_meas, globals=globals()).repeat(repeat_timeit, number)
    min_time = min(timings) / float(number)
    max_time = max(timings) / float(number)
    average_time = np.mean(timings) / float(number)
    print()
    print('----------------------------------------------------------------------------------')
    print('Min time to evaluate rollout is {} ms'.format(min_time * 1.0e3))  # ca. 5 us
    print('Average time to evaluate rollout is {} ms'.format(average_time * 1.0e3))  # ca 5 us
    # The max is of little relevance as it is heavily influenced by other processes running on the computer at the same time
    print('Max time to evaluate rollout is {} ms'.format(max_time * 1.0e3))  # ca. 100 us
    print('----------------------------------------------------------------------------------')

    grad_calc_meas = 'grad_calc(u, s, uprev)'
    number = 1  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 100  # Gives how many times timeit should be repeated
    timings = timeit.Timer(grad_calc_meas, globals=globals()).repeat(repeat_timeit, number)
    min_time = min(timings) / float(number)
    max_time = max(timings) / float(number)
    average_time = np.mean(timings) / float(number)
    print()
    print('----------------------------------------------------------------------------------')
    print('Min time to evaluate grad is {} ms'.format(min_time * 1.0e3))  # ca. 5 us
    print('Average time to evaluate grad is {} ms'.format(average_time * 1.0e3))  # ca 5 us
    # The max is of little relevance as it is heavily influenced by other processes running on the computer at the same time
    print('Max time to evaluate grad is {} ms'.format(max_time * 1.0e3))  # ca. 100 us
    print('----------------------------------------------------------------------------------')
pass

