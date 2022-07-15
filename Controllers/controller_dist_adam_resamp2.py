#the best working controller

from importlib import import_module

import numpy as np
import tensorflow as tf
from ..others.globals_and_utils import create_rng
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from .template_controller import template_controller


#cem class
class controller_dist_adam_resamp2(template_controller):
    def __init__(self, environment, seed: int, num_control_inputs: int, dt: float, mpc_horizon: float, num_rollouts: int, outer_its: int, sample_stdev: float, resamp_per: int, predictor_name: str, predictor_intermediate_steps: int, NET_NAME: str, SAMPLING_TYPE: str, interpolation_step: int, warmup: bool, cem_LR: float, opt_keep_k: int, gradmax_clip: float, adam_beta_1: float, adam_beta_2: float, adam_epsilon: float, **kwargs):
        #First configure random sampler
        self.rng_cem = create_rng(self.__class__.__name__, seed, use_tf=True)

        # Parametrization
        self.num_control_inputs = num_control_inputs

        #basic params
        self.mpc_horizon = mpc_horizon
        self.num_rollouts = num_rollouts
        self.outer_its = outer_its
        self.samp_stdev = sample_stdev
        self.cem_samples = int(mpc_horizon / dt)  # Number of steps in MPC horizon
        self.intermediate_steps = predictor_intermediate_steps

        #First configure random sampler
        self.resamp_per = resamp_per

        self.NET_NAME = NET_NAME
        self.predictor_name = predictor_name

        self.SAMPLING_TYPE = SAMPLING_TYPE
        self.interpolation_step = interpolation_step
        self.do_warmup = warmup

        #optimization params
        self.opt_keep_k = opt_keep_k
        self.cem_LR = tf.constant(cem_LR, dtype=tf.float32)

        self.gradmax_clip = tf.constant(gradmax_clip, dtype = tf.float32)

        #instantiate predictor
        predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
        self.predictor = getattr(predictor_module, predictor_name)(
            horizon=self.cem_samples,
            dt=dt,
            intermediate_steps=self.intermediate_steps,
            disable_individual_compilation=True,
            batch_size=self.num_rollouts,
            net_name=NET_NAME,
        )

        #warmup setup
        self.first_iter_count = self.outer_its
        if self.do_warmup:
            self.first_iter_count = self.cem_samples * self.outer_its

        #if sampling type is "interpolated" setup linear interpolation as a matrix multiplication
        if SAMPLING_TYPE == "interpolated":
            step = interpolation_step
            self.num_valid_vals = int(np.ceil(self.cem_samples / step) + 1)
            self.interp_mat = np.zeros(((self.num_valid_vals - 1) * step, self.num_valid_vals, self.num_control_inputs), dtype=np.float32)
            step_block = np.zeros((step, 2, self.num_control_inputs), dtype=np.float32)
            for j in range(step):
                step_block[j, 0, :] = (step - j) * np.ones((self.num_control_inputs), dtype=np.float32)
                step_block[j, 1, :] = j * np.ones((self.num_control_inputs), dtype=np.float32)
            for i in range(self.num_valid_vals - 1):
                self.interp_mat[i * step:(i + 1) * step, i:i + 2, :] = step_block
            self.interp_mat = self.interp_mat[:self.cem_samples, :, :] / step
            self.interp_mat = tf.constant(tf.transpose(self.interp_mat, perm=(1,0,2)), dtype=tf.float32)
        else:
            self.interp_mat = None
            self.num_valid_vals = self.cem_samples

        #setup sampling distribution
        self.dist_mue = tf.zeros([1,self.cem_samples,self.num_control_inputs], dtype=tf.float32)
        self.dist_var = 0.5*tf.ones([1,self.cem_samples,self.num_control_inputs], dtype=tf.float32)
        self.stdev = tf.sqrt(self.dist_var)
        self.u = 0.0

        #intial input sequence guesses
        self.Q_tf = self.sample_actions(self.rng_cem, self.num_rollouts)

        #setup interpolation
        self.Q_tf = tf.Variable(self.Q_tf)
        self.count = 0
        self.opt = tf.keras.optimizers.Adam(learning_rate=cem_LR, beta_1 = adam_beta_1, beta_2 = adam_beta_2, epsilon = adam_epsilon)
        self.bestQ = None

        super().__init__(environment)

    @Compile
    def sample_actions(self, rng_gen, batchsize):
        #sample actions
        Qn = rng_gen.normal(
            [batchsize, self.num_valid_vals, self.num_control_inputs], dtype=tf.float32) * self.samp_stdev
        Qn = tf.clip_by_value(Qn, -1.0, 1.0)
        if self.SAMPLING_TYPE == "interpolated":
            Qn = tf.transpose(tf.matmul(tf.transpose(Qn, perm=(2,0,1)), tf.transpose(self.interp_mat, perm=(2,0,1))), perm=(1,2,0))
        return Qn

    @Compile
    def grad_step(self, s, Q, opt):
        # rollout trajectories and retrieve cost
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = self.predictor.predict_tf(s, Q)
            traj_cost = self.env_mock.cost_functions.get_trajectory_cost(rollout_trajectory, Q, self.u)
        #retrieve gradient of cost w.r.t. input sequence
        dc_dQ = tape.gradient(traj_cost, Q)
        # modify gradients: makes sure biggest entry of each gradient is at most "gradmax_clip".
        dc_dQ_max = tf.math.reduce_max(tf.abs(dc_dQ), axis=1, keepdims=True) #find max gradient for every sequence
        mask = (dc_dQ_max > self.gradmax_clip) #generate binary mask
        invmask = tf.logical_not(mask)
        dc_dQ_prc = ((dc_dQ/dc_dQ_max)*tf.cast(mask,tf.float32)*self.gradmax_clip + dc_dQ*tf.cast(invmask,tf.float32)) #modify gradients
        # use optimizer to applay gradients and retrieve next set of input sequences
        opt.apply_gradients(zip([dc_dQ_prc], [Q]))
        # clip
        Qn = tf.clip_by_value(Q,-1,1)
        return Qn

    @Compile
    def get_action(self, s, Q):
        # Rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.env_mock.cost_functions.get_trajectory_cost(rollout_trajectory, Q, self.u)
        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[0:self.opt_keep_k]
        # after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        elite_Q = tf.gather(Q, best_idx, axis=0)

        #get distribution of kept trajectories. This is actually unnecessary for this controller, might be incorparated into another one tho
        dist_mue = tf.math.reduce_mean(elite_Q, axis=0, keepdims=True)
        dist_std = tf.math.reduce_std(elite_Q, axis=0, keepdims=True)
        dist_std = tf.clip_by_value(dist_std, self.samp_stdev, 10.0)
        dist_std = tf.concat([dist_std[:, 1:, :], tf.sqrt(0.5)*tf.ones(shape=[1,1,self.num_control_inputs])], axis=1)
        #end of unnecessary part

        #retrieve optimal input and warmstart for next iteration
        u = tf.squeeze(elite_Q[0, 0, :])
        dist_mue = tf.concat([dist_mue[:, 1:, :], tf.zeros([1, 1, self.num_control_inputs])], axis=1)
        Qn = tf.concat([Q[:, 1:, :], Q[:, -1:, :]], axis=1)
        return u, dist_mue, dist_std, Qn, best_idx, traj_cost

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        # tile inital state and convert inputs to tensorflow tensors
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)

        #warm start setup
        if self.count == 0:
            iters = self.first_iter_count
        else:
            iters = self.outer_its
        #optimize control sequences with gradient based optimization
        for _ in range(0, iters):
            Qn = self.grad_step(s, self.Q_tf, self.opt)
            self.Q_tf.assign(Qn)

        #retrieve optimal input and prepare warmstart
        self.u, self.dist_mue, self.stdev, Qn, self.bestQ, J = self.get_action(s, self.Q_tf)
        self.Q, self.J = self.Q_tf.numpy(), J.numpy()

        #modify adam optimizers. The optimizer optimizes all rolled out trajectories at once
        #and keeps weights for all these, which need to get modified.
        #The algorithm not only warmstrats the initial guess, but also the intial optimizer weights
        adam_weights = self.opt.get_weights()
        if self.count % self.resamp_per == 0:
            # if it is time to resample, new random input sequences are drawn for the worst bunch of trajectories
            Qres = self.sample_actions(self.rng_cem, self.num_rollouts - self.opt_keep_k)
            Q_keep = tf.gather(Qn, self.bestQ) #resorting according to costs
            Qn = tf.concat([Qres, Q_keep], axis=0)
            # Updating the weights of adam:
            # For the trajectories which are kept, the weights are shifted for a warmstart
            wk1 = tf.concat([tf.gather(adam_weights[1], self.bestQ)[:,1:,:], tf.zeros([self.opt_keep_k, 1, self.num_control_inputs])], axis=1)
            wk2 = tf.concat([tf.gather(adam_weights[2], self.bestQ)[:,1:,:], tf.zeros([self.opt_keep_k, 1, self.num_control_inputs])], axis=1)
            # For the new trajectories they are reset to 0
            w1 = tf.zeros([self.num_rollouts-self.opt_keep_k, self.cem_samples, self.num_control_inputs])
            w2 = tf.zeros([self.num_rollouts-self.opt_keep_k, self.cem_samples, self.num_control_inputs])
            w1 = tf.concat([w1, wk1], axis=0)
            w2 = tf.concat([w2, wk2], axis=0)
            # Set weights
            self.opt.set_weights([adam_weights[0], w1, w2])
        else:
            # if it is not time to reset, all optimizer weights are shifted for a warmstart
            w1 = tf.concat([adam_weights[1][:,1:,:], tf.zeros([self.num_rollouts,1,self.num_control_inputs])], axis=1)
            w2 = tf.concat([adam_weights[2][:,1:,:], tf.zeros([self.num_rollouts,1,self.num_control_inputs])], axis=1)
            self.opt.set_weights([adam_weights[0], w1, w2])
        self.Q_tf.assign(Qn)
        self.count += 1
        return self.u.numpy()

    def controller_reset(self):
        self.dist_mue = tf.zeros([1, self.cem_samples, self.num_control_inputs])
        self.dist_var = 0.5 * tf.ones([1, self.cem_samples, self.num_control_inputs])
        self.stdev = tf.sqrt(self.dist_var)
        #sample new initial guesses for trajectories
        Qn = self.sample_actions(self.rng_cem, self.num_rollouts)
        self.Q_tf.assign(Qn)
        self.count = 0
        #reset optimizer
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])
        # opt.variables().__init__()



# speed test which is activated if script is run directly and not as module
if __name__ == '__main__':
    ctrl = controller_dist_adam_resamp2()
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
