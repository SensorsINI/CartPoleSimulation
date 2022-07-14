#Controller equivalent to the cem+grad controller from Bharadhwaj et al 2020
#

from importlib import import_module
from operator import attrgetter
import numpy as np
import tensorflow as tf

from Controllers.template_controller import template_controller

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state
from CartPole.cartpole_model import s0

import yaml

from SI_Toolkit.TF.TF_Functions.Compile import Compile

from others.globals_and_utils import create_rng

#load constants from config file
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

num_control_inputs = config["cartpole"]["num_control_inputs"]

#import cost function parts from folder according to config file
cost_function = config["controller"]["general"]["cost_function"]
cost_function = cost_function.replace('-', '_')
cost_function_module = import_module(f"others.cost_functions.{cost_function}")
cost = attrgetter("cost")(cost_function_module)

#cem params
dt = config["controller"]["cem-naive-grad"]["dt"]
cem_horizon = config["controller"]["cem-naive-grad"]["mpc_horizon"]
num_rollouts = config["controller"]["cem-naive-grad"]["cem_rollouts"]
cem_outer_it = config["controller"]["cem-naive-grad"]["cem_outer_it"]

cem_stdev_min = config["controller"]["cem-naive-grad"]["cem_stdev_min"]
cem_best_k = config["controller"]["cem-naive-grad"]["cem_best_k"]
cem_samples = int(cem_horizon / dt)  # Number of steps in MPC horizon
intermediate_steps = config["controller"]["cem-naive-grad"]["predictor_intermediate_steps"]

NET_NAME = config["controller"]["cem-naive-grad"]["CEM_NET_NAME"]
predictor_name = config["controller"]["cem-naive-grad"]["predictor_name"]

#optimization params
cem_LR = config["controller"]["cem-naive-grad"]["cem_LR"]
cem_LR = tf.constant(cem_LR, dtype=tf.float32)
gradmax_clip = config["controller"]["cem-naive-grad"]["gradmax_clip"]
gradmax_clip = tf.constant(gradmax_clip, dtype = tf.float32)

#instantiate predictor
predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
predictor = getattr(predictor_module, predictor_name)(
    horizon=cem_samples,
    dt=dt,
    intermediate_steps=intermediate_steps,
    disable_individual_compilation=True,
    batch_size=num_rollouts,
    net_name=NET_NAME,
)

#controller class
class controller_cem_naive_grad_tf(template_controller):
    def __init__(self):
        #First configure random sampler
        self.rng_cem = create_rng(self.__class__.__name__, config["controller"]["cem-naive-grad"]["SEED"], use_tf=True)

        self.dist_mue = tf.zeros([1,cem_samples,num_control_inputs], dtype=tf.float32)
        self.dist_var = 0.5*tf.ones([1,cem_samples,num_control_inputs], dtype=tf.float32)
        self.stdev = tf.sqrt(self.dist_var)
        self.u = 0.0

    @Compile
    def predict_and_cost(self, s, target_position, rng_cem, dist_mue, stdev):
        # generate random input sequence and clip to control limits
        Q = tf.tile(dist_mue, [num_rollouts, 1, 1]) + rng_cem.normal(
            [num_rollouts, cem_samples, num_control_inputs], dtype=tf.float32) * stdev
        Q = tf.clip_by_value(Q, -1.0, 1.0)
        # rollout the trajectories and record gradient
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = predictor.predict_tf(s, Q)
            traj_cost = cost(rollout_trajectory, Q, target_position, self.u)
        # retrieve gradient
        dc_dQ = tape.gradient(traj_cost, Q)
        dc_dQ_max = tf.math.reduce_max(tf.abs(dc_dQ), axis=1, keepdims=True)
        # modify gradients: makes sure biggest entry of each gradient is at most "gradmax_clip".
        mask = (dc_dQ_max > gradmax_clip)
        invmask = tf.logical_not(mask)
        Q_update = (gradmax_clip* (dc_dQ/dc_dQ_max)*tf.cast(mask,tf.float32) + dc_dQ*tf.cast(invmask,tf.float32))
        # update Q with gradient descent step
        Qn = Q-cem_LR*Q_update
        Qn = tf.clip_by_value(Qn,-1,1)
        #rollout all trajectories a last time
        rollout_trajectory = predictor.predict_tf(s, Qn)
        traj_cost = cost(rollout_trajectory, Qn, target_position, self.u)

        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[0:cem_best_k]
        elite_Q = tf.gather(Qn, best_idx, axis=0)
        # update the distribution for next inner loop
        self.dist_mue = tf.math.reduce_mean(elite_Q, axis=0, keepdims=True)
        self.stdev = tf.math.reduce_std(elite_Q, axis=0, keepdims=True)
        return self.dist_mue, self.stdev

    #step function to find control
    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        # tile s and convert inputs to tensor
        s = np.tile(s, tf.constant([num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        target_position = tf.convert_to_tensor(target_position, dtype=tf.float32)

        #cem steps updating distribution
        for _ in range(0,cem_outer_it):
            self.dist_mue, self.stdev = self.predict_and_cost(s, target_position, self.rng_cem, self.dist_mue, self.stdev)

        #after all inner loops, clip std min, so enough is explored
        #and shove all the values down by one for next control input
        self.stdev = tf.clip_by_value(self.stdev, cem_stdev_min, 10.0)
        self.stdev = tf.concat([self.stdev[:, 1:, :], tf.sqrt(0.5)*tf.ones(shape=(1,1,num_control_inputs))], axis=1)
        self.u = tf.squeeze(self.dist_mue[0,0,:])
        self.dist_mue = tf.concat([self.dist_mue[:, 1:, :], tf.constant(0.0, shape=(1,1,num_control_inputs))], axis=1)
        return self.u.numpy()

    def controller_reset(self):
        #reset controller initial distribution
        self.dist_mue = tf.zeros([1, cem_samples, num_control_inputs])
        self.dist_var = 0.5 * tf.ones([1, cem_samples, num_control_inputs])
        self.stdev = tf.sqrt(self.dist_var)



# speed test, which is activated if script is run directly and not as module
if __name__ == '__main__':
    ctrl = controller_cem_naive_grad_tf()


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
