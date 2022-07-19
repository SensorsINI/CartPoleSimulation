from importlib import import_module

import numpy as np
import tensorflow as tf
from others.globals_and_utils import create_rng
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from Controllers.template_controller import template_controller


#cem class
class controller_cem_tf(template_controller):
    def __init__(self, environment, seed: int, num_control_inputs: int, dt: float, mpc_horizon: float, cem_outer_it: int, cem_rollouts: int, predictor_name: str, predictor_intermediate_steps: int, CEM_NET_NAME: str, cem_stdev_min: float, cem_best_k: int, **kwargs):
        #First configure random sampler
        self.rng_cem = create_rng(self.__class__.__name__, seed)

        # Parametrization
        self.num_control_inputs = num_control_inputs

        #cem params
        self.num_rollouts = cem_rollouts
        self.mpc_horizon = mpc_horizon
        self.cem_outer_it = cem_outer_it
        self.cem_stdev_min = cem_stdev_min
        self.cem_best_k = cem_best_k
        self.cem_samples = int(mpc_horizon / dt)  # Number of steps in MPC horizon
        self.intermediate_steps = predictor_intermediate_steps

        self.NET_NAME = CEM_NET_NAME

        #instantiate predictor
        predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
        self.predictor = getattr(predictor_module, predictor_name)(
            horizon=self.cem_samples,
            dt=dt,
            intermediate_steps=self.intermediate_steps,
            disable_individual_compilation=True,
            batch_size=self.num_rollouts,
            net_name=self.NET_NAME,
        )

        # Initialization
        self.dist_mue = np.zeros([1,self.cem_samples,num_control_inputs])
        self.dist_var = 0.5*np.ones([1,self.cem_samples,num_control_inputs])
        self.stdev = np.sqrt(self.dist_var)
        self.u = 0

        super().__init__(environment)

    @Compile
    def predict_and_cost(self, s, Q):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.env_mock.cost_functions.get_trajectory_cost(rollout_trajectory, Q, self.u)
        return traj_cost, rollout_trajectory

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        for _ in range(0,self.cem_outer_it):
            #generate random input sequence and clip to control limits
            Q = np.tile(self.dist_mue,(self.num_rollouts,1,1)) + np.multiply(self.rng_cem.standard_normal(
                size=(self.num_rollouts, self.cem_samples, self.num_control_inputs), dtype=np.float32), self.stdev)
            Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)

            Q = tf.convert_to_tensor(Q, dtype=tf.float32)

            #rollout the trajectories and get cost
            traj_cost, rollout_trajectory = self.predict_and_cost(s, Q)
            Q, traj_cost = Q.numpy(), traj_cost.numpy()
            #sort the costs and find best k costs
            sorted_cost = np.argsort(traj_cost)
            best_idx = sorted_cost[:self.cem_best_k]
            elite_Q = Q[best_idx,:,:]
            #update the distribution for next inner loop
            self.dist_mue = np.mean(elite_Q, axis=0, keepdims=True)
            self.stdev = np.std(elite_Q, axis=0, keepdims=True)

        #after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        self.stdev = np.clip(self.stdev, self.cem_stdev_min, None)
        self.stdev = np.append(self.stdev[:,1:,:], np.sqrt(0.5)*np.ones((1,1,self.num_control_inputs)), axis=1).astype(np.float32)
        self.Q, self.J = Q, traj_cost
        self.u = np.squeeze(self.dist_mue[0,0,:])
        self.dist_mue = np.append(self.dist_mue[:,1:,:], np.zeros((1,1,self.num_control_inputs)), axis=1).astype(np.float32)
        return self.u

    def controller_reset(self):
        self.dist_mue = np.zeros([1, self.cem_samples, self.num_control_inputs])
        self.dist_var = 0.5 * np.ones([1, self.cem_samples, self.num_control_inputs])
        self.stdev = np.sqrt(self.dist_var)



# speed test, which is activated if script is run directly and not as module
if __name__ == '__main__':
    ctrl = controller_cem_tf()
    import timeit

    from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX,
                                          ANGLE_SIN_IDX, ANGLED_IDX,
                                          POSITION_IDX, POSITIOND_IDX,
                                          create_cartpole_state)

    s0 = create_cartpole_state()
    # Set non-zero input
    s = s0
    s[POSITION_IDX] = -30.2
    s[POSITIOND_IDX] = 2.87
    s[ANGLE_IDX] = -0.32
    s[ANGLED_IDX] = 0.237
    u = -0.24

    ctrl.step(s0)
    f_to_measure = 'ctrl.step(s0)'
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
