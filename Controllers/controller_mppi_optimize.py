import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from Controllers.template_controller import template_controller

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state
from CartPole.cartpole_model import s0
from CartPole.cartpole_jacobian import cartpole_jacobian

import yaml

from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from others.globals_and_utils import create_rng


#load constants from config file
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

num_control_inputs = config["cartpole"]["num_control_inputs"]

#import cost function parts from folder according to config file
cost_function = config["controller"]["general"]["cost_function"]
cost_function = cost_function.replace('-', '_')
cost_function_cmd = 'from others.cost_functions.'+cost_function+' import q, phi, cost'
exec(cost_function_cmd)

#cost funciton params
cc_weight = config["controller"]["mppi-optimize"]["cc_weight"]
R = config["controller"]["mppi-optimize"]["R"]
LBD = config["controller"]["mppi-optimize"]["LBD"]

#mppi params
dt = config["controller"]["mppi-optimize"]["dt"]
mppi_horizon = config["controller"]["mppi-optimize"]["mpc_horizon"]
num_rollouts = config["controller"]["mppi-optimize"]["num_rollouts"]
mppi_samples = int(mppi_horizon / dt)  # Number of steps in MPC horizon

NU = config["controller"]["mppi-optimize"]["NU"]
SQRTRHODTINV = config["controller"]["mppi-optimize"]["SQRTRHOINV"] * (1 / np.math.sqrt(dt))
GAMMA = config["controller"]["mppi-optimize"]["GAMMA"]
SAMPLING_TYPE = config["controller"]["mppi-optimize"]["SAMPLING_TYPE"]

NET_NAME = config["controller"]["mppi-optimize"]["NET_NAME"]
predictor_type = config["controller"]["mppi-optimize"]["predictor_type"]


#optimization params
cem_LR = config["controller"]["mppi-optimize"]["LR"]
cem_LR = tf.constant(cem_LR, dtype=tf.float32)

adam_beta_1 = config["controller"]["mppi-optimize"]["adam_beta_1"]
adam_beta_2 = config["controller"]["mppi-optimize"]["adam_beta_2"]
adam_epsilon = float(config["controller"]["mppi-optimize"]["adam_epsilon"])
gradmax_clip = config["controller"]["mppi-optimize"]["gradmax_clip"]
gradmax_clip = tf.constant(gradmax_clip, dtype = tf.float32)
optim_steps = config["controller"]["mppi-optimize"]["optim_steps"]

#create default predictor
predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)

"""Define Predictor"""
if predictor_type == "EulerTF":
    predictor = predictor_ODE_tf(horizon=mppi_samples, dt=dt, intermediate_steps=1, disable_individual_compilation=True)
elif predictor_type == "Euler":
    predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)
elif predictor_type == "NeuralNet":
    predictor = predictor_autoregressive_tf(
        horizon=mppi_samples, batch_size=num_rollouts, net_name=NET_NAME
    )

#mppi correction for importance sampling
def mppi_correction_cost(u, delta_u):
    return tf.reduce_sum(cc_weight * (0.5 * (1 - 1.0 / NU) * R * (delta_u ** 2) + R * u * delta_u + 0.5 * R * (u ** 2)), axis=2)

#total cost of the trajectory
def mppi_cost(s_hor ,u, target_position, u_prev, delta_u):
    #stage costs
    stage_cost = q(s_hor[:,1:,:],u,target_position, u_prev)
    stage_cost = stage_cost + mppi_correction_cost(u, delta_u)
    #reduce alonge rollouts and add final cost
    total_cost = tf.math.reduce_sum(stage_cost,axis=1)
    total_cost = total_cost + phi(s_hor, target_position)
    return total_cost

#path integral approximation: sum deltaU's weighted with exponential funciton of trajectory costs
#according to mppi theory
def reward_weighted_average(S, delta_u):
    rho = tf.math.reduce_min(S)
    exp_s = tf.exp(-1.0/LBD * (S-rho))
    a = tf.math.reduce_sum(exp_s)
    b = tf.math.reduce_sum(exp_s[:,tf.newaxis,tf.newaxis]*delta_u, axis=0, keepdims=True)/a
    return b

#initialize pertubation
def inizialize_pertubation(random_gen, stdev = SQRTRHODTINV, sampling_type = SAMPLING_TYPE):
    #if interpolation on, interpolate with method from tensor flow probability
    if sampling_type == "interpolated":
        step = 10
        range_stop = int(tf.math.ceil(mppi_samples / step) * step) + 1
        t = tf.range(range_stop, delta = step)
        t_interp = tf.cast(tf.range(range_stop), tf.float32)
        delta_u = random_gen.normal([num_rollouts, t.shape[0], num_control_inputs], dtype=tf.float32) * stdev
        interp = tfp.math.interp_regular_1d_grid(t_interp, t_interp[0], t_interp[-1], delta_u, axis=1)
        delta_u = interp[:,:mppi_samples,:]
    else:
        #otherwise i.i.d. generation
        delta_u = random_gen.normal([num_rollouts, mppi_samples, num_control_inputs], dtype=tf.float32) * stdev
    return delta_u



#controller class
class controller_mppi_optimize(template_controller):
    def __init__(self):
        #First configure random sampler
        self.rng_cem = create_rng(self.__class__.__name__, config["controller"]["mppi-optimize"]["SEED"], use_tf=True)

        #Setup prototype control sequence
        self.Q = tf.zeros([1,mppi_samples,num_control_inputs], dtype=tf.float32)
        self.Q = tf.Variable(self.Q)
        self.u = 0.0
        #setup adam optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate=cem_LR, beta_1=adam_beta_1, beta_2=adam_beta_2,
                                            epsilon=adam_epsilon)

    @Compile
    def mppi_prior(self, s, target_position, u_nom, random_gen, u_old):
        # generate random input sequence and clip to control limits
        delta_u = inizialize_pertubation(random_gen)
        u_run = tf.tile(u_nom, [num_rollouts, 1, 1]) + delta_u
        u_run = tf.clip_by_value(u_run, -1.0, 1.0)
        #predict trajectories
        rollout_trajectory = predictor.predict_tf(s, u_run)
        #rollout cost
        traj_cost = mppi_cost(rollout_trajectory, u_run, target_position, u_old, delta_u)
        #retrive control sequence via path integral
        u_nom = tf.clip_by_value(u_nom + reward_weighted_average(traj_cost, delta_u), -1.0, 1.0)
        return u_nom

    @Compile
    def grad_step(self, s, target_position, Q, opt):
        #do a gradient descent step
        #setup gradient tape
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            #rollout trajectory and retrive cost
            rollout_trajectory = predictor.predict_tf(s, Q)
            traj_cost = cost(rollout_trajectory, Q, target_position, self.u)
        #retrieve gradient of cost w.r.t. input sequence
        dc_dQ = tape.gradient(traj_cost, Q)
        #modify gradients: makes sure biggest entry of each gradient is at most "gradmax_clip". (For this controller only one sequence
        dc_dQ_max = tf.math.reduce_max(tf.abs(dc_dQ), axis=1, keepdims=True) #find max gradient for every sequence
        mask = (dc_dQ_max > gradmax_clip) #generate binary mask
        invmask = tf.logical_not(mask)
        dc_dQ_prc = ((dc_dQ / dc_dQ_max) * tf.cast(mask, tf.float32) * gradmax_clip + dc_dQ * tf.cast(
            invmask, tf.float32)) #modify gradients
        #use optimizer to applay gradients and retrieve next set of input sequences
        opt.apply_gradients(zip([dc_dQ_prc], [Q]))
        #clip
        Qn = tf.clip_by_value(Q, -1, 1)
        return Qn

    #step function to find control
    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        # tile inital state and convert inputs to tensorflow tensors
        s = np.tile(s, tf.constant([num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        target_position = tf.convert_to_tensor(target_position, dtype=tf.float32)

        #first retrieve suboptimal control sequence with mppi
        Q_mppi = self.mppi_prior(s, target_position, self.Q, self.rng_cem, self.u)
        self.Q.assign(Q_mppi)

        #optimize control sequence with gradient based optimization
        for _ in range(optim_steps):
            Q_opt = self.grad_step(s, target_position, self.Q, self.opt)
            self.Q.assign(Q_opt)

        self.u = self.Q[0, 0, :]
        self.Q.assign(tf.concat([self.Q[:, 1:, :], tf.zeros([1,1,num_control_inputs])], axis=1)) #shift and initialize new input with 0
        #reset adam optimizer
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])
        return np.squeeze(self.u.numpy())

    def controller_reset(self):
        #reset prototype control sequence
        self.Q.assign(tf.zeros([1, mppi_samples, num_control_inputs], dtype=tf.float32))
        self.u = 0.0
        #reset adam optimizer
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])



# speed test, which is activated if script is run directly and not as module
if __name__ == '__main__':
    ctrl = controller_mppi_optimize()


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