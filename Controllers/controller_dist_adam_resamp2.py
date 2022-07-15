#the best working controller

import importlib

import numpy as np
import tensorflow as tf
import yaml
from others.globals_and_utils import create_rng
from SI_Toolkit.Predictors.predictor_autoregressive_tf import \
    predictor_autoregressive_tf
from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from Controllers.template_controller import template_controller

#load constants from config file
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

num_control_inputs = config["cartpole"]["num_control_inputs"]

#basic params
dt = config["controller"]["dist-adam-resamp2"]["dt"]
mpc_horizon = config["controller"]["dist-adam-resamp2"]["mpc_horizon"]
num_rollouts = config["controller"]["dist-adam-resamp2"]["num_rollouts"]
outer_its = config["controller"]["dist-adam-resamp2"]["outer_its"]
samp_stdev = config["controller"]["dist-adam-resamp2"]["sample_stdev"]
cem_samples = int(mpc_horizon / dt)  # Number of steps in MPC horizon

resamp_per = config["controller"]["dist-adam-resamp2"]["resamp_per"]

NET_NAME = config["controller"]["dist-adam-resamp2"]["NET_NAME"]
predictor_type = config["controller"]["dist-adam-resamp2"]["predictor_type"]

SAMPLING_TYPE = config["controller"]["dist-adam-resamp2"]["SAMPLING_TYPE"]
interpolation_step = config["controller"]["dist-adam-resamp2"]["interpolation_step"]
do_warmup = config["controller"]["dist-adam-resamp2"]["warmup"]

#optimization params
opt_keep_k = config["controller"]["dist-adam-resamp2"]["opt_keep_k"]
cem_LR = config["controller"]["dist-adam-resamp2"]["LR"]
cem_LR = tf.constant(cem_LR, dtype=tf.float32)

adam_beta_1 = config["controller"]["dist-adam-resamp2"]["adam_beta_1"]
adam_beta_2 = config["controller"]["dist-adam-resamp2"]["adam_beta_2"]
adam_epsilon = float(config["controller"]["dist-adam-resamp2"]["adam_epsilon"])
gradmax_clip = config["controller"]["dist-adam-resamp2"]["gradmax_clip"]
gradmax_clip = tf.constant(gradmax_clip, dtype = tf.float32)

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

#warmup setup
first_iter_count = outer_its
if do_warmup:
    first_iter_count = cem_samples * outer_its

#if sampling type is "interpolated" setup linear interpolation as a matrix multiplication
if SAMPLING_TYPE == "interpolated":
    step = interpolation_step
    num_valid_vals = int(np.ceil(cem_samples / step) + 1)
    interp_mat = np.zeros(((num_valid_vals - 1) * step, num_valid_vals, num_control_inputs), dtype=np.float32)
    step_block = np.zeros((step, 2, num_control_inputs), dtype=np.float32)
    for j in range(step):
        step_block[j, 0, :] = (step - j) * np.ones((num_control_inputs), dtype=np.float32)
        step_block[j, 1, :] = j * np.ones((num_control_inputs), dtype=np.float32)
    for i in range(num_valid_vals - 1):
        interp_mat[i * step:(i + 1) * step, i:i + 2, :] = step_block
    interp_mat = interp_mat[:cem_samples, :, :] / step
    interp_mat = tf.constant(tf.transpose(interp_mat, perm=(1,0,2)), dtype=tf.float32)
else:
    interp_mat = None
    num_valid_vals = cem_samples




#cem class
class controller_dist_adam_resamp2(template_controller):
    def __init__(self, environment):
        #First configure random sampler
        self.rng_cem = create_rng(self.__class__.__name__, config["controller"]["dist-adam-resamp2"]["SEED"], use_tf=True)

        #setup sampling distribution
        self.dist_mue = tf.zeros([1,cem_samples,num_control_inputs], dtype=tf.float32)
        self.dist_var = 0.5*tf.ones([1,cem_samples,num_control_inputs], dtype=tf.float32)
        self.stdev = tf.sqrt(self.dist_var)
        self.u = 0.0

        #intial input sequence guesses
        self.Q = self.sample_actions(self.rng_cem, num_rollouts)

        #setup interpolation
        self.Q = tf.Variable(self.Q)
        self.count = 0
        self.opt = tf.keras.optimizers.Adam(learning_rate=cem_LR, beta_1 = adam_beta_1, beta_2 = adam_beta_2, epsilon = adam_epsilon)
        self.bestQ = None

        super().__init__(environment)

    @Compile
    def sample_actions(self, rng_gen, batchsize):
        #sample actions
        Qn = rng_gen.normal(
            [batchsize, num_valid_vals, num_control_inputs], dtype=tf.float32) * samp_stdev
        Qn = tf.clip_by_value(Qn, -1.0, 1.0)
        if SAMPLING_TYPE == "interpolated":
            Qn = tf.transpose(tf.matmul(tf.transpose(Qn, perm=(2,0,1)), tf.transpose(interp_mat, perm=(2,0,1))), perm=(1,2,0))
        return Qn

    @Compile
    def grad_step(self, s, Q, opt):
        # rollout trajectories and retrieve cost
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = predictor.predict_tf(s, Q)
            traj_cost = self.env_mock.cost_functions.get_trajectory_cost(rollout_trajectory, Q, self.u)
        #retrieve gradient of cost w.r.t. input sequence
        dc_dQ = tape.gradient(traj_cost, Q)
        # modify gradients: makes sure biggest entry of each gradient is at most "gradmax_clip".
        dc_dQ_max = tf.math.reduce_max(tf.abs(dc_dQ), axis=1, keepdims=True) #find max gradient for every sequence
        mask = (dc_dQ_max > gradmax_clip) #generate binary mask
        invmask = tf.logical_not(mask)
        dc_dQ_prc = ((dc_dQ/dc_dQ_max)*tf.cast(mask,tf.float32)*gradmax_clip + dc_dQ*tf.cast(invmask,tf.float32)) #modify gradients
        # use optimizer to applay gradients and retrieve next set of input sequences
        opt.apply_gradients(zip([dc_dQ_prc], [Q]))
        # clip
        Qn = tf.clip_by_value(Q,-1,1)
        return Qn

    @Compile
    def get_action(self, s, Q):
        # Rollout trajectories and retrieve cost
        rollout_trajectory = predictor.predict_tf(s, Q)
        traj_cost = self.env_mock.cost_functions.get_trajectory_cost(rollout_trajectory, Q, self.u)
        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[0:opt_keep_k]
        # after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        elite_Q = tf.gather(Q, best_idx, axis=0)

        #get distribution of kept trajectories. This is actually unnecessary for this controller, might be incorparated into another one tho
        dist_mue = tf.math.reduce_mean(elite_Q, axis=0, keepdims=True)
        dist_std = tf.math.reduce_std(elite_Q, axis=0, keepdims=True)
        dist_std = tf.clip_by_value(dist_std, samp_stdev, 10.0)
        dist_std = tf.concat([dist_std[:, 1:, :], tf.sqrt(0.5)*tf.ones(shape=[1,1,num_control_inputs])], axis=1)
        #end of unnecessary part

        #retrieve optimal input and warmstart for next iteration
        u = tf.squeeze(elite_Q[0, 0, :])
        dist_mue = tf.concat([dist_mue[:, 1:, :], tf.zeros([1, 1, num_control_inputs])], axis=1)
        Qn = tf.concat([Q[:, 1:, :], Q[:, -1:, :]], axis=1)
        return u, dist_mue, dist_std, Qn, best_idx

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        # tile inital state and convert inputs to tensorflow tensors
        s = np.tile(s, tf.constant([num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)

        #warm start setup
        if self.count == 0:
            iters = first_iter_count
        else:
            iters = outer_its
        #optimize control sequences with gradient based optimization
        for _ in range(0, iters):
            Qn = self.grad_step(s, self.Q, self.opt)
            self.Q.assign(Qn)

        #retrieve optimal input and prepare warmstart
        self.u, self.dist_mue, self.stdev, Qn, self.bestQ = self.get_action(s, self.Q)

        #modify adam optimizers. The optimizer optimizes all rolled out trajectories at once
        #and keeps weights for all these, which need to get modified.
        #The algorithm not only warmstrats the initial guess, but also the intial optimizer weights
        adam_weights = self.opt.get_weights()
        if self.count % resamp_per == 0:
            # if it is time to resample, new random input sequences are drawn for the worst bunch of trajectories
            Qres = self.sample_actions(self.rng_cem, num_rollouts - opt_keep_k)
            Q_keep = tf.gather(Qn, self.bestQ) #resorting according to costs
            Qn = tf.concat([Qres, Q_keep], axis=0)
            # Updating the weights of adam:
            # For the trajectories which are kept, the weights are shifted for a warmstart
            wk1 = tf.concat([tf.gather(adam_weights[1], self.bestQ)[:,1:,:], tf.zeros([opt_keep_k, 1, num_control_inputs])], axis=1)
            wk2 = tf.concat([tf.gather(adam_weights[2], self.bestQ)[:,1:,:], tf.zeros([opt_keep_k, 1, num_control_inputs])], axis=1)
            # For the new trajectories they are reset to 0
            w1 = tf.zeros([num_rollouts-opt_keep_k, cem_samples, num_control_inputs])
            w2 = tf.zeros([num_rollouts-opt_keep_k, cem_samples, num_control_inputs])
            w1 = tf.concat([w1, wk1], axis=0)
            w2 = tf.concat([w2, wk2], axis=0)
            # Set weights
            self.opt.set_weights([adam_weights[0], w1, w2])
        else:
            # if it is not time to reset, all optimizer weights are shifted for a warmstart
            w1 = tf.concat([adam_weights[1][:,1:,:], tf.zeros([num_rollouts,1,num_control_inputs])], axis=1)
            w2 = tf.concat([adam_weights[2][:,1:,:], tf.zeros([num_rollouts,1,num_control_inputs])], axis=1)
            self.opt.set_weights([adam_weights[0], w1, w2])
        self.Q.assign(Qn)
        self.count += 1
        return self.u.numpy()

    def controller_reset(self):
        self.dist_mue = tf.zeros([1, cem_samples, num_control_inputs])
        self.dist_var = 0.5 * tf.ones([1, cem_samples, num_control_inputs])
        self.stdev = tf.sqrt(self.dist_var)
        #sample new initial guesses for trajectories
        Qn = self.sample_actions(self.rng_cem, num_rollouts)
        self.Q.assign(Qn)
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
