import numpy as np
from numpy.random import SFC64, Generator

from Controllers.template_controller import template_controller
from CartPole.cartpole_model import TrackHalfLength

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state
from CartPole.cartpole_model import u_max, s0
from CartPole.cartpole_jacobian import cartpole_jacobian

import yaml

from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
from SI_Toolkit.Predictors.predictor_ODE_tf_pure import predictor_ODE_tf_pure
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
import tensorflow as tf
import tensorflow_probability as tfp
tf.config.run_functions_eagerly(False)

@tf.function(jit_compile = True)
def grad_desc(u,s):
        with tf.GradientTape() as tape:
            tape.watch(u)
            rollout_trajectory = predictor.predict_tf(s,u)
            cost = rollout_trajectory[:,-1,POSITION_IDX]**2
        dc_du = tape.gradient(cost,u)
        # dc_du = 0.0
        return dc_du,cost

# @tf.function(jit_compile = True)
def test(u,s):
    for i in range(0,100):
        dc_du,cost  = grad_desc(u,s)
        print(cost)
        u = u - lr * dc_du




#load constants from config file
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
dt = config["controller"]["mppi"]["dt"]
cem_horizon = config["controller"]["mppi"]["mpc_horizon"]
cem_samples = int(cem_horizon / dt)  # Number of steps in MPC horizon
predictor = predictor_ODE_tf_pure(horizon=cem_samples, dt=dt, intermediate_steps=10)

s0 = create_cartpole_state()
# Set non-zero input
s = s0
s[POSITION_IDX] = -30.2
s[POSITIOND_IDX] = 2.87
s[ANGLE_IDX] = -0.32
s[ANGLED_IDX] = 0.237

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
rng_gen = Generator(SFC64(SEED))
dist_var = 0.5*np.ones([1,50])
stdev = np.sqrt(dist_var)
num_rollouts = 100
dist_mue = np.zeros([1,50])
Q = np.tile(dist_mue,(num_rollouts,1))+ np.multiply(rng_gen.standard_normal(
                size=(num_rollouts, 50), dtype=np.float32),stdev)
Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)


s = s_org
s = np.tile(s,tf.constant([num_rollouts,1]))
u = tf.convert_to_tensor(Q)
u = u[:, :, tf.newaxis]
rollout_trajectory = predictor.predict_tf(s, u)
lr = 10

test(u,s)

import timeit
f_to_measure = 'test(u,s)'
number = 10  # Gives the number of times each timeit call executes the function which we want to measure
repeat_timeit = 1
timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
print(timings)

pass