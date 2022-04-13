import scipy
import numpy as np
from numpy.random import SFC64, Generator
from datetime import datetime
from numba import jit, prange
import tensorflow as tf

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

dt = config["controller"]["mppi"]["dt"]
cem_horizon = config["controller"]["mppi"]["mpc_horizon"]
num_rollouts = config["controller"]["cem"]["cem_rollouts"]
dd_weight = config["controller"]["mppi"]["dd_weight"]
cc_weight = config["controller"]["mppi"]["cc_weight"]
ep_weight = config["controller"]["mppi"]["ep_weight"]
R = config["controller"]["cem"]["cem_R"]

cem_outer_it = config["controller"]["cem"]["cem_outer_it"]
NET_NAME = config["controller"]["cem"]["CEM_NET_NAME"]
predictor_type = config["controller"]["cem"]["cem_predictor_type"]
cem_stdev_min = config["controller"]["cem"]["cem_stdev_min"]
ccrc_weight = config["controller"]["cem"]["cem_ccrc_weight"]
cem_best_k = config["controller"]["cem"]["cem_best_k"]
cem_samples = int(cem_horizon / dt)  # Number of steps in MPC horizon

cem_LR = config["controller"]["cem"]["cem_LR"]
cem_max_LR = config["controller"]["cem"]["cem_max_LR"]
cem_LR = tf.constant(cem_LR, dtype=tf.float32)
cem_max_LR = tf.constant(cem_max_LR, dtype = tf.float32)

#create predictor
predictor = predictor_ODE(horizon=cem_samples, dt=dt, intermediate_steps=10)

"""Define Predictor"""
if predictor_type == "EulerTF":
    predictor = predictor_ODE_tf_pure(horizon=cem_samples, dt=dt, intermediate_steps=1)
elif predictor_type == "Euler":
    predictor = predictor_ODE(horizon=cem_samples, dt=dt, intermediate_steps=10)
elif predictor_type == "NeuralNet":
    predictor = predictor_autoregressive_tf(
        horizon=cem_samples, batch_size=num_rollouts, net_name=NET_NAME
    )


#cost for distance from track edge
def distance_difference_cost(position, target_position):
    """Compute penalty for distance of cart to the target position"""
    return ((position - target_position) / (2.0 * TrackHalfLength)) ** 2 + tf.cast(
        tf.abs(position) > 0.90 * TrackHalfLength
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
    u_prev_vec = tf.concat((tf.ones((nrol,1))*u_prev,u[:,:-1]),axis=-1)
    return (u - u_prev_vec) ** 2

#all stage costs together
def q(s,u,target_position, u_prev):
    dd = dd_weight * distance_difference_cost(
        s[:, :, POSITION_IDX], target_position
    )
    ep = ep_weight * E_pot_cost(s[:, :, ANGLE_IDX])
    cc = cc_weight * CC_cost(u)
    ccrc = ccrc_weight * control_change_rate_cost(u,u_prev,num_rollouts)
    stage_cost = dd+ep+cc+ccrc
    return stage_cost

#total cost of the trajectory
def cost(s_hor ,u,target_position,u_prev):
    stage_cost = q(s_hor[:,1:,:],u,target_position,u_prev)
    total_cost = tf.math.reduce_sum(stage_cost,axis=1)
    total_cost = total_cost + phi(s_hor,target_position)
    return total_cost

#cem class
class controller_grad_cem(template_controller):
    def __init__(self):
        #First configure random sampler
        SEED = config["controller"]["mppi"]["SEED"]
        if SEED == "None":
            SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)
        self.rng_cem = tf.random.Generator.from_seed(SEED)

        self.dist_mue = tf.zeros([1,cem_samples], dtype=tf.float32)
        self.dist_var = 0.5*tf.ones([1,cem_samples], dtype=tf.float32)
        self.stdev = tf.sqrt(self.dist_var)
        self.u = 0.0

    @tf.function(jit_compile=True)
    def predict_and_cost(self, s, target_position, dist_mue, dist_std, random_gen):
        # generate random input sequence and clip to control limits
        Q = tf.tile(dist_mue, [num_rollouts, 1]) + random_gen.normal(
            [num_rollouts, cem_samples], dtype=tf.float32) * dist_std
        Q = tf.clip_by_value(Q, -1.0, 1.0)
        # simulate once with gradient on
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = predictor.predict_tf(s, Q[:, :, tf.newaxis])
            traj_cost = cost(rollout_trajectory, Q, target_position, self.u)
        for _ in range(0,cem_outer_it):
            #sort indecies according to cost and take k best ones
            sorted_cost = tf.argsort(traj_cost)
            best_idx = sorted_cost[0:cem_best_k]
            #gather the good ones
            elite_Q = tf.gather(Q, best_idx, axis=0)

            #retrive gradients of best ones
            dc_dQ = tape.gradient(traj_cost, Q)
            dc_dQ_elite = tf.gather(dc_dQ, best_idx, axis = 0)

            #create sensible step, so step makes sense
            dc_dQ_max = tf.math.reduce_max(tf.abs(dc_dQ_elite), axis = 1)
            mask = (dc_dQ_max > 1)[:,tf.newaxis]
            invmask = tf.logical_not(mask)
            Q_update = (cem_max_LR*(dc_dQ_elite/dc_dQ_max[:,tf.newaxis])*tf.cast(mask,tf.float32) + cem_LR*dc_dQ_elite*tf.cast(invmask,tf.float32))

            #update best Q's
            elite_Q = elite_Q-Q_update
            elite_Q = tf.clip_by_value(elite_Q,-1,1)

            #update distribution
            dist_mue = tf.math.reduce_mean(elite_Q, axis=0, keepdims=True)
            dist_std = tf.math.reduce_std(elite_Q, axis=0, keepdims=True)

            #resample
            Q = tf.tile(dist_mue, [num_rollouts, 1]) + random_gen.normal(
                [num_rollouts, cem_samples], dtype=tf.float32) * dist_std
            Q = tf.clip_by_value(Q, -1.0, 1.0)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(Q)
                rollout_trajectory = predictor.predict_tf(s, Q[:, :, tf.newaxis])
                traj_cost = cost(rollout_trajectory, Q, target_position, self.u)

        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[0:cem_best_k]
        # gather the good ones
        elite_Q = tf.gather(Q, best_idx, axis=0)
        dist_mue = tf.math.reduce_mean(elite_Q, axis=0, keepdims=True)
        dist_std = tf.math.reduce_std(elite_Q, axis=0, keepdims=True)
        # after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        dist_std = tf.clip_by_value(dist_std, cem_stdev_min, 10.0)
        dist_std = tf.concat([dist_std[:, 1:], tf.sqrt(0.5)[tf.newaxis, tf.newaxis]], -1)
        u = dist_mue[0, 0]
        dist_mue = tf.concat([dist_mue[:, 1:], tf.constant(0.0, shape=[1, 1])], -1)
        return u, dist_mue, dist_std

    #step function to find control
    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        s = np.tile(s, tf.constant([num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        target_position = tf.convert_to_tensor(target_position, dtype=tf.float32)

        self.u, self.dist_mue, self.stdev = self.predict_and_cost(s, target_position, self.dist_mue, self.stdev, self.rng_cem)
        return self.u.numpy()

    def controller_reset(self):
        self.dist_mue = tf.zeros([1, cem_samples])
        self.dist_var = 0.5 * tf.ones([1, cem_samples])
        self.stdev = tf.sqrt(self.dist_var)




if __name__ == '__main__':
    ctrl = controller_grad_cem()


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
    repeat_timeit = 100  # Gives how many times timeit should be repeated
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