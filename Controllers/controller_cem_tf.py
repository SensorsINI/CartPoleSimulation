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
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
from SI_Toolkit.TF.TF_Functions.Compile import Compile

#load constants from config file
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

num_control_inputs = config["cartpole"]["num_control_inputs"]

#import cost function parts from folder according to config file
cost_function = config["controller"]["general"]["cost_function"]
cost_function = cost_function.replace('-', '_')
cost_function_cmd = 'from others.cost_functions.'+cost_function+' import q, phi, cost'
exec(cost_function_cmd)

#cem params
dt = config["controller"]["cem"]["dt"]
num_rollouts = config["controller"]["cem"]["cem_rollouts"]
mpc_horizon = config["controller"]["cem"]["mpc_horizon"]
cem_outer_it = config["controller"]["cem"]["cem_outer_it"]
cem_stdev_min = config["controller"]["cem"]["cem_stdev_min"]
cem_best_k = config["controller"]["cem"]["cem_best_k"]
cem_samples = int(mpc_horizon / dt)  # Number of steps in MPC horizon

NET_NAME = config["controller"]["cem"]["CEM_NET_NAME"]
predictor_type = config["controller"]["cem"]["cem_predictor_type"]


#create predictor
predictor = predictor_ODE(horizon=cem_samples, dt=dt, intermediate_steps=10)

"""Define Predictor"""
if predictor_type == "EulerTF":
    predictor = predictor_ODE_tf(horizon=cem_samples, dt=dt, intermediate_steps=1, disable_individual_compilation=True)
elif predictor_type == "Euler":
    predictor = predictor_ODE(horizon=cem_samples, dt=dt, intermediate_steps=10)
elif predictor_type == "NeuralNet":
    predictor = predictor_autoregressive_tf(
        horizon=cem_samples, batch_size=num_rollouts, net_name=NET_NAME
    )


#cem class
class controller_cem_tf(template_controller):
    def __init__(self):
        #First configure random sampler
        SEED = config["controller"]["cem"]["SEED"]
        if SEED == "None":
            SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)
        self.rng_cem = Generator(SFC64(SEED))

        self.dist_mue = np.zeros([1,cem_samples,num_control_inputs])
        self.dist_var = 0.5*np.ones([1,cem_samples,num_control_inputs])
        self.stdev = np.sqrt(self.dist_var)
        self.u = 0

    @Compile
    def predict_and_cost(self, s, Q, target_position):
        # rollout trajectories and retrieve cost
        rollout_trajectory = predictor.predict_tf(s, Q)
        traj_cost = cost(rollout_trajectory, Q, target_position, self.u)
        return traj_cost, rollout_trajectory

    #step function to find control
    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        s = np.tile(s, tf.constant([num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        for _ in range(0,cem_outer_it):
            #generate random input sequence and clip to control limits
            Q = np.tile(self.dist_mue,(num_rollouts,1,1)) + np.multiply(self.rng_cem.standard_normal(
                size=(num_rollouts, cem_samples, num_control_inputs), dtype=np.float32), self.stdev)
            Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)

            Q = tf.convert_to_tensor(Q, dtype=tf.float32)
            target_position = tf.convert_to_tensor(target_position, dtype=tf.float32)

            #rollout the trajectories and get cost
            traj_cost, rollout_trajectory = self.predict_and_cost(s, Q, target_position)
            Q = Q.numpy()
            #sort the costs and find best k costs
            sorted_cost = np.argsort(traj_cost.numpy())
            best_idx = sorted_cost[:cem_best_k]
            elite_Q = Q[best_idx,:,:]
            #update the distribution for next inner loop
            self.dist_mue = np.mean(elite_Q, axis=0, keepdims=True)
            self.stdev = np.std(elite_Q, axis=0, keepdims=True)

        #after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        self.stdev = np.clip(self.stdev, cem_stdev_min, None)
        self.stdev = np.append(self.stdev[:,1:,:], np.sqrt(0.5)*np.ones((1,1,num_control_inputs)), axis=1).astype(np.float32)
        self.u = np.squeeze(self.dist_mue[0,0,:])
        self.dist_mue = np.append(self.dist_mue[:,1:,:], np.zeros((1,1,num_control_inputs)), axis=1).astype(np.float32)
        return self.u

    def controller_reset(self):
        self.dist_mue = np.zeros([1, cem_samples, num_control_inputs])
        self.dist_var = 0.5 * np.ones([1, cem_samples, num_control_inputs])
        self.stdev = np.sqrt(self.dist_var)



# speed test, which is activated if script is run directly and not as module
if __name__ == '__main__':
    ctrl = controller_cem_tf()


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