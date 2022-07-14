from importlib import import_module
from operator import attrgetter
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from Controllers.template_controller import template_controller

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state
from CartPole.cartpole_model import s0

import yaml

from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from others.globals_and_utils import create_rng


#load constants from config file
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

num_control_inputs = config["cartpole"]["num_control_inputs"]

cost_function = config["controller"]["general"]["cost_function"]
cost_function = cost_function.replace('-', '_')
cost_function_module = import_module(f"others.cost_functions.{cost_function}")
q, phi = attrgetter("q", "phi")(cost_function_module)

dt = config["controller"]["mppi-var"]["dt"]
mppi_horizon = config["controller"]["mppi-var"]["mpc_horizon"]
num_rollouts = config["controller"]["mppi-var"]["num_rollouts"]
SAMPLING_TYPE = config["controller"]["mppi-var"]["SAMPLING_TYPE"]
interpolation_step = config["controller"]["mppi-var"]["interpolation_step"]

cc_weight = config["controller"]["mppi-var"]["cc_weight"]

NET_NAME = config["controller"]["mppi-var"]["NET_NAME"]
predictor_name = config["controller"]["mppi-var"]["predictor_name"]

mppi_samples = int(mppi_horizon / dt)  # Number of steps in MPC horizon
intermediate_steps = config["controller"]["mppi-var"]["predictor_intermediate_steps"]

R = config["controller"]["mppi-var"]["R"]
LBD = config["controller"]["mppi-var"]["LBD_mc"]
NU = config["controller"]["mppi-var"]["NU_mc"]
SQRTRHODTINV = config["controller"]["mppi-var"]["SQRTRHOINV_mc"] * (1 / np.math.sqrt(dt))
GAMMA = config["controller"]["mppi-var"]["GAMMA"]

mppi_lr = config["controller"]["mppi-var"]["LR"]
stdev_min = config["controller"]["mppi-var"]["STDEV_min"]
stdev_max = config["controller"]["mppi-var"]["STDEV_max"]
max_grad_norm = config["controller"]["mppi-var"]["max_grad_norm"]


#instantiate predictor
predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
predictor = getattr(predictor_module, predictor_name)(
    horizon=mppi_samples,
    dt=dt,
    intermediate_steps=intermediate_steps,
    disable_individual_compilation=True,
    batch_size=num_rollouts,
    net_name=NET_NAME,
)


#setup interpolation matrix
if SAMPLING_TYPE == "interpolated":
    step = interpolation_step
    num_valid_vals = int(np.ceil(mppi_samples / step) + 1)
    interp_mat = np.zeros(((num_valid_vals - 1) * step, num_valid_vals, num_control_inputs), dtype=np.float32)
    step_block = np.zeros((step, 2, num_control_inputs), dtype=np.float32)
    for j in range(step):
        step_block[j, 0, :] = (step - j) * np.ones((num_control_inputs), dtype=np.float32)
        step_block[j, 1, :] = j * np.ones((num_control_inputs), dtype=np.float32)
    for i in range(num_valid_vals - 1):
        interp_mat[i * step:(i + 1) * step, i:i + 2, :] = step_block
    interp_mat = interp_mat[:mppi_samples, :, :] / step
    interp_mat = tf.constant(tf.transpose(interp_mat, perm=(1,0,2)), dtype=tf.float32)
else:
    interp_mat = None
    num_valid_vals = mppi_samples

#mppi correction
def mppi_correction_cost(u, delta_u, nuvec):
    if SAMPLING_TYPE == "interpolated":
        nudiv = tf.transpose(tf.matmul(tf.transpose(nuvec, perm=(2,0,1)), tf.transpose(interp_mat, perm=(2,0,1))), perm=(1,2,0))
    else:
        nudiv = nuvec
    return tf.reduce_sum(cc_weight * (0.5 * (1 - 1.0 / nudiv**2) * R * (delta_u ** 2) + R * u * delta_u + 0.5 * R * (u ** 2)), axis=2)

#total cost of the trajectory
def uncorr_cost(s_hor ,u, target_position, u_prev, delta_u):
    stage_cost = q(s_hor[:,1:,:],u,target_position, u_prev)
    stage_cost = stage_cost
    unc_cost = tf.math.reduce_sum(stage_cost,axis=1)
    unc_cost = unc_cost + phi(s_hor, target_position)
    return unc_cost

#mppi averaging of trajectories
def reward_weighted_average(S, delta_u):
    rho = tf.math.reduce_min(S)
    exp_s = tf.exp(-1.0/LBD * (S-rho))
    a = tf.math.reduce_sum(exp_s)
    b = tf.math.reduce_sum(exp_s[:,tf.newaxis,tf.newaxis]*delta_u, axis=0, keepdims=True)/a
    return b

#initialize the pertubations
def inizialize_pertubation(random_gen, nuvec):
    delta_u = random_gen.normal([num_rollouts, num_valid_vals, num_control_inputs], dtype=tf.float32) * nuvec * SQRTRHODTINV
    if SAMPLING_TYPE == "interpolated":
        delta_u = tf.transpose(tf.matmul(tf.transpose(delta_u, perm=(2,0,1)), tf.transpose(interp_mat, perm=(2,0,1))), perm=(1,2,0)) #here interpolation is simply a multiplication with a matrix
    return delta_u



#controller class
class controller_mppi_var(template_controller):
    def __init__(self):
        #First configure random sampler
        self.rng_mppi = create_rng(self.__class__.__name__, config["controller"]["mppi-var"]["SEED"], use_tf=True)
        #set up nominal u
        self.u_nom = tf.zeros([1,mppi_samples,num_control_inputs], dtype=tf.float32)
        #set up vector of variances to be optimized
        self.nuvec = np.math.sqrt(NU)*tf.ones([1, num_valid_vals, num_control_inputs])
        self.nuvec = tf.Variable(self.nuvec)
        self.u = 0.0

    @Compile
    def do_step(self, s, target_position, u_nom, random_gen, u_old, nuvec):
        #start gradient tape
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(nuvec) #watch variances on tape
            delta_u = inizialize_pertubation(random_gen, nuvec) #initialize pertubations
            #build real input and clip, preserving gradient
            u_run = tf.tile(u_nom, [num_rollouts, 1, 1]) + delta_u
            u_run = tfp.math.clip_by_value_preserve_gradient(u_run, -1.0, 1.0)
            #rollout and cost
            rollout_trajectory = predictor.predict_tf(s, u_run)
            unc_cost = uncorr_cost(rollout_trajectory, u_run, target_position, u_old, delta_u)
            mean_uncost = tf.math.reduce_mean(unc_cost)
            #retrieve gradient
            dc_ds = tape.gradient(mean_uncost, nuvec)
            dc_ds = tf.clip_by_norm(dc_ds, max_grad_norm,axes = [1])
        #correct cost of mppi
        cor_cost = mppi_correction_cost(u_run, delta_u, nuvec)
        cor_cost = tf.math.reduce_sum(cor_cost, axis=1)
        traj_cost = unc_cost + cor_cost
        #build optimal input
        u_nom = tf.clip_by_value(u_nom + reward_weighted_average(traj_cost, delta_u), -1.0, 1.0)
        u = u_nom[0, 0, :]
        u_nom = tf.concat([u_nom[:, 1:, :], tf.constant(0.0, shape=[1, 1, num_control_inputs])], axis=1)
        #adapt variance
        new_nuvec = nuvec-mppi_lr*dc_ds
        new_nuvec = tf.clip_by_value(new_nuvec, stdev_min, stdev_max)
        return u, u_nom, new_nuvec

    #step function to find control
    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        s = np.tile(s, tf.constant([num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        target_position = tf.convert_to_tensor(target_position, dtype=tf.float32)
        self.u, self.u_nom, new_nuvec = self.do_step(s, target_position, self.u_nom, self.rng_mppi, self.u, self.nuvec)
        self.nuvec.assign(new_nuvec)
        return tf.squeeze(self.u).numpy()

    #reset to initial values
    def controller_reset(self):
        self.u_nom = tf.zeros([1, mppi_samples, num_control_inputs], dtype=tf.float32)
        self.nuvec.assign(np.math.sqrt(NU)*tf.ones([1, num_valid_vals, num_control_inputs]))
        self.u = 0.0




if __name__ == '__main__':
    ctrl = controller_mppi_var()


    import timeit

    s0 = create_cartpole_state()
    # Set non-zero input
    s = s0
    s[POSITION_IDX] = -0.12
    s[POSITIOND_IDX] = 0.3
    s[ANGLE_IDX] = -0.32
    s[ANGLED_IDX] = 0.237
    u = -0.24

    ctrl.step(s0, 0.0)
    f_to_measure = 'ctrl.step(s0,0.0)'
    number = 1  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 1000  # Gives how many times timeit should be repeated
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