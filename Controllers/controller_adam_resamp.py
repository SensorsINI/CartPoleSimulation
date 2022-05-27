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
num_rollouts = config["controller"]["adam-resamp"]["num_rollouts"]

cost_function = config["controller"]["general"]["cost_function"]
cost_function = cost_function.replace('-', '_')
cost_function_cmd = 'from others.cost_functions.'+cost_function+' import cost'
exec(cost_function_cmd)


cem_outer_it = config["controller"]["cem"]["cem_outer_it"]
NET_NAME = config["controller"]["cem"]["CEM_NET_NAME"]
predictor_type = config["controller"]["cem"]["cem_predictor_type"]
samp_stdev_min = config["controller"]["adam-resamp"]["stdev_min"]
ccrc_weight = config["controller"]["cem"]["cem_ccrc_weight"]
cem_best_k = config["controller"]["adam-resamp"]["cem_best_k"]
cem_samples = int(cem_horizon / dt)  # Number of steps in MPC horizon

cem_LR = config["controller"]["adam-resamp"]["LR"]
cem_max_LR = config["controller"]["cem"]["cem_max_LR"]
cem_LR = tf.constant(cem_LR, dtype=tf.float32)
cem_max_LR = tf.constant(cem_max_LR, dtype = tf.float32)
resamp_per = config["controller"]["adam-resamp"]["resamp_per"]
adam_beta_1 = config["controller"]["adam-resamp"]["adam_beta_1"]
adam_beta_2 = config["controller"]["adam-resamp"]["adam_beta_2"]
adam_epsilon = float(config["controller"]["adam-resamp"]["adam_epsilon"])
adam_norm_clip_val = float(config["controller"]["adam-resamp"]["norm_clip_val"])

SAMPLING_TYPE = config["controller"]["adam-resamp"]["SAMPLING_TYPE"]
interpolation_step = config["controller"]["adam-resamp"]["interpolation_step"]
do_warmup = config["controller"]["adam-resamp"]["warmup"]

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

first_iter_count = cem_outer_it
if do_warmup:
    first_iter_count = cem_samples * cem_outer_it

if SAMPLING_TYPE == "interpolated":
    step = interpolation_step
    num_valid_vals = int(np.ceil(cem_samples / step) + 1)
    interp_mat = np.zeros(((num_valid_vals - 1) * step, num_valid_vals))
    step_block = np.zeros((step, 2))
    for j in range(step):
        step_block[j][0] = step - j
        step_block[j][1] = j
    for i in range(num_valid_vals - 1):
        interp_mat[i * step:(i + 1) * step, i:i + 2] = step_block
    interp_mat = interp_mat[:cem_samples, :] / step
    interp_mat = tf.constant(interp_mat.T, dtype=tf.float32)
else:
    interp_mat = None
    num_valid_vals = cem_samples




#cem class
class controller_adam_resamp(template_controller):
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
        self.Q = self.sample_actions(self.rng_cem, num_rollouts)

        self.Q = tf.Variable(self.Q)
        self.count = 0
        self.opt = tf.keras.optimizers.Adam(learning_rate=cem_LR, beta_1 = adam_beta_1, beta_2 = adam_beta_2, epsilon = adam_epsilon)
        self.bestQ = None

    @tf.function(jit_compile=True)
    def sample_actions(self, rng_gen, batchsize):
        Qn = rng_gen.normal(
            [batchsize, num_valid_vals], dtype=tf.float32) * samp_stdev_min
        Qn = tf.clip_by_value(Qn, -1.0, 1.0)
        if SAMPLING_TYPE == "interpolated":
            Qn = tf.matmul(Qn, interp_mat)
        return Qn

    @tf.function(jit_compile=True)
    def grad_step(self, s, target_position, Q, opt):
        # generate random input sequence and clip to control limits
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = predictor.predict_tf(s, Q[:, :, tf.newaxis])
            traj_cost = cost(rollout_trajectory, Q, target_position, self.u)
        dc_dQ = tape.gradient(traj_cost, Q)
        # dc_dQ_max = tf.math.reduce_max(tf.abs(dc_dQ), axis = 1)
        # mask = (dc_dQ_max > 1)[:,tf.newaxis]
        # invmask = tf.logical_not(mask)
        # dc_dQ_prc = ((dc_dQ/dc_dQ_max[:,tf.newaxis])*tf.cast(mask,tf.float32) + dc_dQ*tf.cast(invmask,tf.float32))
        dc_dQ_prc = tf.clip_by_norm(dc_dQ, adam_norm_clip_val, axes = [1])
        opt.apply_gradients(zip([dc_dQ], [Q]))
        Qn = tf.clip_by_value(Q,-1,1)
        return Qn

    @tf.function(jit_compile=True)
    def get_action(self, s, target_position, Q):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = predictor.predict_tf(s, Q[:, :, tf.newaxis])
            traj_cost = cost(rollout_trajectory, Q, target_position, self.u)
            # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[0:cem_best_k]
        # after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        elite_Q = tf.gather(Q, best_idx, axis=0)
        dist_mue = tf.math.reduce_mean(elite_Q, axis=0, keepdims=True)
        dist_std = tf.math.reduce_std(elite_Q, axis=0, keepdims=True)
        dist_std = tf.clip_by_value(dist_std, samp_stdev_min, 10.0)
        dist_std = tf.concat([dist_std[:, 1:], tf.sqrt(0.5)[tf.newaxis, tf.newaxis]], -1)
        u = elite_Q[0, 0]
        dist_mue = tf.concat([dist_mue[:, 1:], tf.zeros([1, 1])], -1)
        Qn = tf.concat([Q[:, 1:], Q[:, -1, tf.newaxis]], -1)
        return u, dist_mue, dist_std, Qn, best_idx

    #step function to find control
    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        s = np.tile(s, tf.constant([num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        target_position = tf.convert_to_tensor(target_position, dtype=tf.float32)
        if self.count == 0:
            iters = first_iter_count
        else:
            iters = cem_outer_it
        for _ in range(0, iters):
            Qn = self.grad_step(s, target_position, self.Q, self.opt)
            self.Q.assign(Qn)

        self.u, self.dist_mue, self.stdev, Qn, self.bestQ = self.get_action(s, target_position, self.Q)

        adam_weights = self.opt.get_weights()
        if self.count % resamp_per == 0:
            Qres = self.sample_actions(self.rng_cem, num_rollouts - cem_best_k)
            Q_keep = tf.gather(Qn, self.bestQ)
            Qn = tf.concat([Qres, Q_keep], 0)
            wk1 = tf.concat([tf.gather(adam_weights[1], self.bestQ)[:,1:], tf.zeros([cem_best_k, 1])], -1)
            wk2 = tf.concat([tf.gather(adam_weights[2], self.bestQ)[:,1:], tf.zeros([cem_best_k, 1])], -1)
            w1 = tf.zeros([num_rollouts-cem_best_k, cem_samples])
            w2 = tf.zeros([num_rollouts-cem_best_k, cem_samples])
            w1 = tf.concat([w1, wk1], 0)
            w2 = tf.concat([w2, wk2], 0)
            self.opt.set_weights([adam_weights[0], w1, w2])
        else:
            w1 = tf.concat([adam_weights[1][:,1:], tf.zeros([num_rollouts,1])], -1)
            w2 = tf.concat([adam_weights[2][:,1:], tf.zeros([num_rollouts,1])], -1)
            self.opt.set_weights([adam_weights[0], w1, w2])
        self.Q.assign(Qn)
        self.count += 1
        return self.u.numpy()

    def controller_reset(self):
        self.dist_mue = tf.zeros([1, cem_samples])
        self.dist_var = 0.5 * tf.ones([1, cem_samples])
        self.stdev = tf.sqrt(self.dist_var)
        Qn = self.sample_actions(self.rng_cem, num_rollouts)
        self.Q.assign(Qn)
        self.count = 0
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])
        # opt.variables().__init__()




if __name__ == '__main__':
    ctrl = controller_adam_resamp()


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