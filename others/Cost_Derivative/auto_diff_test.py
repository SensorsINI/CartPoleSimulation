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
def grad_desc(u, s):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(u)
        rollout_trajectory = predictor.predict_tf(s, u)
        costx = tf.math.reduce_mean(
            0 * rollout_trajectory[:, :, POSITION_IDX] ** 2 + 2 * (1 - rollout_trajectory[:, :, ANGLE_COS_IDX]) ** 2,
            axis=1)
        costu = tf.math.reduce_mean(u[:, :, 0] ** 2, axis=1)
        cost = costx + 0.5 * costu
    dc_du = tape.gradient(cost, u)
    # dc_du = 0.0
    return dc_du, cost, costx, costu, rollout_trajectory


# @tf.function(jit_compile = True)
def test(u, s, lr):
    for i in range(0, 100):
        dc_du, cost, costx, costu, send = grad_desc(u, s)
        print(cost)
        u = u - lr * dc_du
        u = tf.clip_by_value(u, -1, 1)

    return send, u


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

s_org = tf.convert_to_tensor(s)

# u_org = tf.Variable([0.594623, 0.11093523, -0.32577565, 0.36339644, 0.19863953,
#                  -0.67005044, -0.00572653, 0.50473666, 0.82851535, 0.03227299,
#                  -0.89665616, -1., -0.15769833, -0.8742089, -0.00434032,
#                  -0.5908449, -0.8486508, 0.46566853, -0.26742178, -0.2585441,
#                  -1., 1., -1., 0.820513, 1.,
#                  0.65235853, 0.7771242, -0.834638, 0.9568739, 0.21720093,
#                  -0.18284637, 0.9694907, 0.68292177, -1., 1.,
#                  0.37337917, -0.46058115, -0.6156913, 0.52652395, 0.06510112,
#                  -0.13692386, 0.4193466, 0.08954383, -0.02065406, 0.7458399,
#                  -1., 0.83411133, -0.5809542, -0.5786972, -0.70775455],
#                 dtype=tf.float32)
SEED = 5876
horizon = cem_samples
rng_gen = Generator(SFC64(SEED))
dist_var = 0.5 * np.ones([1, horizon])
stdev = np.sqrt(dist_var)
num_rollouts = 50
dist_mue = np.zeros([1, horizon])
Q = np.tile(dist_mue, (num_rollouts, 1)) + np.multiply(rng_gen.standard_normal(
    size=(num_rollouts, horizon), dtype=np.float32), stdev)
Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)

s = s_org
s = np.tile(s, tf.constant([num_rollouts, 1]))
u = tf.convert_to_tensor(Q)
u = u[:, :, tf.newaxis]
s_start = predictor.predict_tf(s, u)
lr = 5
with tf_options({}):
    print(tf.config.optimizer.get_experimental_options())
    send, uend = test(u, s, lr)
    # %% plotting
    toplt = tf.transpose(send, perm=[1, 0, 2])
    fig, ax = plt.subplots()
    ax.plot(toplt[:, :, ANGLE_IDX])
    plt.title('Angle')
    plt.show()
    fig2, ax2 = plt.subplots()
    ax2.plot(toplt[:, :, POSITION_IDX])
    plt.title('Position')
    plt.show()
    toplt2 = tf.transpose(s_start, perm=[1, 0, 2])
    fig3, ax3 = plt.subplots()
    ax3.plot(toplt2[:, :, POSITION_IDX])
    plt.title('Position start')
    plt.show()

    fig5, ax5 = plt.subplots()
    ax5.plot(toplt2[:, :, ANGLE_IDX])
    plt.title('Angle start')
    plt.show()

    fig4, ax4 = plt.subplots()
    toplt3 = tf.transpose(uend, perm=[1, 0, 2])
    ax4.plot(toplt3[:, :, 0])
    # ax4.plot(u[0,:,0])
    plt.title('Control')
    plt.show()

    import timeit

    f_to_measure = 'test(u,s,lr)'
    number = 10  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 1
    timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
    print(timings)
pass

