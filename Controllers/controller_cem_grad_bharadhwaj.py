#Controller equivalent to the cem+grad controller from Bharadhwaj et al 2020
#

from importlib import import_module

import numpy as np
import tensorflow as tf
from others.globals_and_utils import create_rng
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from Controllers import template_controller


#controller class
class controller_cem_grad_bharadhwaj(template_controller):
    def __init__(self, environment, seed: int, num_control_inputs: int, dt: float, mpc_horizon: float, cem_outer_it: int, cem_rollouts: int, predictor_name: str, predictor_intermediate_steps: int, CEM_NET_NAME: str, cem_initial_action_stdev: float, cem_stdev_min: float, cem_best_k: int, cem_LR: float, gradmax_clip: float, **kwargs):
        # First configure random sampler
        self.rng_cem = create_rng(self.__class__.__name__, seed, use_tf=True)

        # Parametrization
        self.num_control_inputs = num_control_inputs

        #cem params
        self.num_rollouts = cem_rollouts
        self.cem_outer_it = cem_outer_it

        self.cem_initial_action_stdev = cem_initial_action_stdev
        self.cem_stdev_min = cem_stdev_min
        self.cem_best_k = cem_best_k
        self.cem_samples = int(mpc_horizon / dt)  # Number of steps in MPC horizon
        self.intermediate_steps = predictor_intermediate_steps

        self.NET_NAME = CEM_NET_NAME
        self.predictor_name = predictor_name

        #optimization params
        cem_LR = tf.constant(cem_LR, dtype=tf.float32)
        self.gradmax_clip = tf.constant(gradmax_clip, dtype = tf.float32)

        self.optim = tf.keras.optimizers.RMSprop(learning_rate=cem_LR, rho=0.99, momentum=0.0, epsilon=1.0e-08, centered=False)

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
        self.dist_var = self.cem_initial_action_stdev*tf.ones([1,self.cem_samples,self.num_control_inputs], dtype=tf.float32)
        self.stdev = tf.sqrt(self.dist_var)
        self.u = 0.0
        self.Q_tf = tf.Variable(
            initial_value=tf.zeros([self.num_rollouts, self.cem_samples, self.num_control_inputs]),
            trainable=True,
            dtype=tf.float32,
        )

        super().__init__(environment)
        self.action_low = tf.convert_to_tensor(self.env_mock.action_space.low)
        self.action_high = tf.convert_to_tensor(self.env_mock.action_space.high)

    @Compile
    def predict_and_cost(self, s, Q_tf: tf.Variable, opt):
        # rollout the trajectories and record gradient
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q_tf)
            rollout_trajectory = self.predictor.predict_tf(s, Q_tf)
            traj_cost = self.env_mock.cost_functions.get_trajectory_cost(rollout_trajectory, Q_tf, self.u)
        # retrieve gradient
        dc_dQ = tape.gradient(traj_cost, Q_tf)
        # modify gradients: makes sure norm of each gradient is at most "gradmax_clip".
        dc_dQ_clipped = tf.clip_by_norm(dc_dQ, self.gradmax_clip, axes=[1, 2])
        # update Q_tf with gradient descent step
        opt.apply_gradients(zip([dc_dQ_clipped], [Q_tf]))
        Qn = tf.clip_by_value(Q_tf, self.action_low, self.action_high)
        #rollout all trajectories a last time
        rollout_trajectory = self.predictor.predict_tf(s, Qn)
        traj_cost = self.env_mock.cost_functions.get_trajectory_cost(rollout_trajectory, Qn, self.u)

        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[0:self.cem_best_k]
        elite_Q = tf.gather(Qn, best_idx, axis=0)
        # update the distribution for next inner loop
        dist_mue = tf.math.reduce_mean(elite_Q, axis=0, keepdims=True)
        stdev = tf.math.reduce_std(elite_Q, axis=0, keepdims=True)
        return dist_mue, stdev, Qn, elite_Q, traj_cost, rollout_trajectory
    
    def _sample_actions(self, rng: tf.random.Generator, num_samples: int):
        return (
            tf.tile(self.dist_mue, [num_samples, 1, 1])
            + self.stdev * rng.normal(
                [num_samples, self.cem_samples, self.num_control_inputs], dtype=tf.float32
            )
        )

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        # tile s and convert inputs to tensor
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)

        # generate random input sequence and clip to control limits
        elite_Q = self._sample_actions(self.rng_cem, self.cem_best_k)

        #cem steps updating distribution
        for _ in range(0, self.cem_outer_it):
            Q_sampled = self._sample_actions(self.rng_cem, self.num_rollouts - self.cem_best_k)
            Q = tf.concat([elite_Q, Q_sampled], axis=0)
            Q = tf.clip_by_value(Q, self.action_low, self.action_high)
            self.Q_tf.assign(Q)
            self.dist_mue, self.stdev, Q, elite_Q, J, rollout_trajectory = self.predict_and_cost(s, self.Q_tf, self.optim)
        
        #after all inner loops, clip std min, so enough is explored
        #and shove all the values down by one for next control input
        self.stdev = tf.clip_by_value(self.stdev, self.cem_stdev_min, 10.0)
        self.stdev = tf.concat([self.stdev[:, 1:, :], tf.sqrt(self.cem_initial_action_stdev)*tf.ones(shape=(1,1,self.num_control_inputs))], axis=1)
        self.u = tf.squeeze(self.dist_mue[0,0,:])
        self.dist_mue = tf.concat([self.dist_mue[:, 1:, :], tf.constant(0.0, shape=(1,1,self.num_control_inputs))], axis=1)
        
        self.Q_logged, self.J_logged = Q.numpy(), J.numpy()
        self.rollout_trajectories_logged = rollout_trajectory.numpy()
        self.u_logged = self.u
        
        return self.u.numpy()

    def controller_reset(self):
        #reset controller initial distribution
        self.dist_mue = tf.zeros([1, self.cem_samples, self.num_control_inputs])
        self.dist_var = self.cem_initial_action_stdev * tf.ones([1, self.cem_samples, self.num_control_inputs])
        self.stdev = tf.sqrt(self.dist_var)

