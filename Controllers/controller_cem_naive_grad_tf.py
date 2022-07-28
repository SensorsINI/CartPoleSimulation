#Controller equivalent to the cem+grad controller from Bharadhwaj et al 2020
#

from importlib import import_module

import numpy as np
import tensorflow as tf
from others.globals_and_utils import create_rng
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from Controllers import template_controller


#controller class
class controller_cem_naive_grad_tf(template_controller):
    def __init__(self, environment, seed: int, num_control_inputs: int, dt: float, mpc_horizon: float, cem_outer_it: int, cem_rollouts: int, predictor_name: str, predictor_intermediate_steps: int, CEM_NET_NAME: str, cem_stdev_min: float, cem_R: int, cem_ccrc_weight: float, cem_best_k: int, cem_LR: float, gradmax_clip: float, **kwargs):
        # First configure random sampler
        self.rng_cem = create_rng(self.__class__.__name__, seed, use_tf=True)

        # Parametrization
        self.num_control_inputs = num_control_inputs

        #cem params
        self.cem_horizon = mpc_horizon
        self.num_rollouts = cem_rollouts
        self.cem_outer_it = cem_outer_it

        self.cem_stdev_min = cem_stdev_min
        self.cem_best_k = cem_best_k
        self.cem_samples = int(self.cem_horizon / dt)  # Number of steps in MPC horizon
        self.intermediate_steps = predictor_intermediate_steps

        self.NET_NAME = CEM_NET_NAME
        self.predictor_name = predictor_name

        #optimization params
        self.cem_LR = cem_LR
        self.cem_LR = tf.constant(self.cem_LR, dtype=tf.float32)
        self.gradmax_clip = gradmax_clip
        self.gradmax_clip = tf.constant(self.gradmax_clip, dtype = tf.float32)

        #instantiate predictor
        self.predictor_module = import_module(f"SI_Toolkit.Predictors.{self.predictor_name}")
        self.predictor = getattr(self.predictor_module, self.predictor_name)(
            horizon=self.cem_samples,
            dt=dt,
            intermediate_steps=self.intermediate_steps,
            disable_individual_compilation=True,
            batch_size=self.num_rollouts,
            net_name=self.NET_NAME,
        )

        # Initialization
        self.dist_mue = tf.zeros([1,self.cem_samples,self.num_control_inputs], dtype=tf.float32)
        self.dist_var = 0.5*tf.ones([1,self.cem_samples,self.num_control_inputs], dtype=tf.float32)
        self.stdev = tf.sqrt(self.dist_var)
        self.u = 0.0

        super().__init__(environment)
        self.action_low = tf.convert_to_tensor(self.env_mock.action_space.low)
        self.action_high = tf.convert_to_tensor(self.env_mock.action_space.high)

    @Compile
    def predict_and_cost(self, s, rng_cem, dist_mue, stdev):
        # generate random input sequence and clip to control limits
        Q = tf.tile(dist_mue, [self.num_rollouts, 1, 1]) + rng_cem.normal(
            [self.num_rollouts, self.cem_samples, self.num_control_inputs], dtype=tf.float32) * stdev
        Q = tf.clip_by_value(Q, self.action_low, self.action_high)
        # rollout the trajectories and record gradient
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = self.predictor.predict_tf(s, Q)
            traj_cost = self.env_mock.cost_functions.get_trajectory_cost(rollout_trajectory, Q, self.u)
        # retrieve gradient
        dc_dQ = tape.gradient(traj_cost, Q)
        # modify gradients: makes sure norm of each gradient is at most "gradmax_clip".
        Q_update = tf.clip_by_norm(dc_dQ, self.gradmax_clip, axes=[1, 2])
        # update Q with gradient descent step
        Qn = Q-self.cem_LR*Q_update
        Qn = tf.clip_by_value(Qn, self.action_low, self.action_high)
        #rollout all trajectories a last time
        rollout_trajectory = self.predictor.predict_tf(s, Qn)
        traj_cost = self.env_mock.cost_functions.get_trajectory_cost(rollout_trajectory, Qn, self.u)

        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[0:self.cem_best_k]
        elite_Q = tf.gather(Qn, best_idx, axis=0)
        # update the distribution for next inner loop
        self.dist_mue = tf.math.reduce_mean(elite_Q, axis=0, keepdims=True)
        self.stdev = tf.math.reduce_std(elite_Q, axis=0, keepdims=True)
        return self.dist_mue, self.stdev, Qn, traj_cost

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        # tile s and convert inputs to tensor
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)

        #cem steps updating distribution
        for _ in range(0,self.cem_outer_it):
            self.dist_mue, self.stdev, Q, J = self.predict_and_cost(s, self.rng_cem, self.dist_mue, self.stdev)
        
        self.Q, self.J = Q.numpy(), J.numpy()

        #after all inner loops, clip std min, so enough is explored
        #and shove all the values down by one for next control input
        self.stdev = tf.clip_by_value(self.stdev, self.cem_stdev_min, 10.0)
        self.stdev = tf.concat([self.stdev[:, 1:, :], tf.sqrt(0.5)*tf.ones(shape=(1,1,self.num_control_inputs))], axis=1)
        self.u = tf.squeeze(self.dist_mue[0,0,:])
        self.dist_mue = tf.concat([self.dist_mue[:, 1:, :], tf.constant(0.0, shape=(1,1,self.num_control_inputs))], axis=1)
        return self.u.numpy()

    def controller_reset(self):
        #reset controller initial distribution
        self.dist_mue = tf.zeros([1, self.cem_samples, self.num_control_inputs])
        self.dist_var = 0.5 * tf.ones([1, self.cem_samples, self.num_control_inputs])
        self.stdev = tf.sqrt(self.dist_var)



# speed test, which is activated if script is run directly and not as module
if __name__ == '__main__':
    ctrl = controller_cem_naive_grad_tf()
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
