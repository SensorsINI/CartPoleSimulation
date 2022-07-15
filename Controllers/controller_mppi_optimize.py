from importlib import import_module
from operator import attrgetter
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from Controllers.template_controller import template_controller

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state
from CartPole.cartpole_model import s0
from CartPole.cartpole_jacobian import cartpole_jacobian

from SI_Toolkit.TF.TF_Functions.Compile import Compile

from others.globals_and_utils import create_rng

#controller class
class controller_mppi_optimize(template_controller):
    def __init__(self, seed: int, num_control_inputs: int, cc_weight: float, R: float, LBD: float, mpc_horizon: float, num_rollouts: int, dt: float, predictor_intermediate_steps: int, NU: float, SQRTRHOINV: float, GAMMA: float, SAMPLING_TYPE: str, NET_NAME: str, gradmax_clip: float, optim_steps: int, predictor_name: str, cost_function: str, mppi_LR: float, adam_beta_1: float, adam_beta_2: float, adam_epsilon: float, **kwargs):
        #First configure random sampler
        self.rng_mppi = create_rng(self.__class__.__name__, seed, use_tf=True)

        # Parametrization
        self.num_control_inputs = num_control_inputs

        #import cost function parts from folder according to config file
        cost_function = cost_function.replace('-', '_')
        cost_function_module = import_module(f"others.cost_functions.{cost_function}")
        self.q, self.phi, self.cost = attrgetter("q", "phi", "cost")(cost_function_module)

        #cost funciton params
        self.cc_weight = cc_weight
        self.R = R
        self.LBD = LBD

        #mppi params
        self.mppi_horizon = mpc_horizon
        self.num_rollouts = num_rollouts
        self.mppi_samples = int(self.mppi_horizon / dt)  # Number of steps in MPC horizon

        self.NU = NU
        self.SQRTRHODTINV = SQRTRHOINV * (1 / np.math.sqrt(dt))
        self.GAMMA = GAMMA
        self.SAMPLING_TYPE = SAMPLING_TYPE

        self.NET_NAME = NET_NAME

        #optimization params
        mppi_LR = tf.constant(mppi_LR, dtype=tf.float32)

        self.gradmax_clip = tf.constant(gradmax_clip, dtype = tf.float32)
        self.optim_steps = optim_steps

        #instantiate predictor
        predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
        self.predictor = getattr(predictor_module, predictor_name)(
            horizon=self.mppi_samples,
            dt=dt,
            intermediate_steps=predictor_intermediate_steps,
            disable_individual_compilation=True,
            batch_size=num_rollouts,
            net_name=NET_NAME,
        )

        #Setup prototype control sequence
        self.Q_opt = tf.zeros([1,self.mppi_samples,num_control_inputs], dtype=tf.float32)
        self.Q_opt = tf.Variable(self.Q_opt)
        self.u = 0.0
        #setup adam optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate=mppi_LR, beta_1=adam_beta_1, beta_2=adam_beta_2,
                                            epsilon=adam_epsilon)

    #mppi correction for importance sampling
    def mppi_correction_cost(self, u, delta_u):
        return tf.reduce_sum(self.cc_weight * (0.5 * (1 - 1.0 / self.NU) * self.R * (delta_u ** 2) + self.R * u * delta_u + 0.5 * self.R * (u ** 2)), axis=2)

    #total cost of the trajectory
    def mppi_cost(self, s_hor ,u, target_position, u_prev, delta_u):
        #stage costs
        stage_cost = self.q(s_hor[:,1:,:],u,target_position, u_prev)
        stage_cost = stage_cost + self.mppi_correction_cost(u, delta_u)
        #reduce alonge rollouts and add final cost
        total_cost = tf.math.reduce_sum(stage_cost,axis=1)
        total_cost = total_cost + self.phi(s_hor, target_position)
        return total_cost

    #path integral approximation: sum deltaU's weighted with exponential funciton of trajectory costs
    #according to mppi theory
    def reward_weighted_average(self, S, delta_u):
        rho = tf.math.reduce_min(S)
        exp_s = tf.exp(-1.0/self.LBD * (S-rho))
        a = tf.math.reduce_sum(exp_s)
        b = tf.math.reduce_sum(exp_s[:,tf.newaxis,tf.newaxis]*delta_u, axis=0, keepdims=True)/a
        return b

    #initialize pertubation
    def inizialize_pertubation(self, random_gen):
        #if interpolation on, interpolate with method from tensor flow probability
        stdev = self.SQRTRHODTINV
        sampling_type = self.SAMPLING_TYPE
        if sampling_type == "interpolated":
            step = 10
            range_stop = int(tf.math.ceil(self.mppi_samples / step) * step) + 1
            t = tf.range(range_stop, delta = step)
            t_interp = tf.cast(tf.range(range_stop), tf.float32)
            delta_u = random_gen.normal([self.num_rollouts, t.shape[0], self.num_control_inputs], dtype=tf.float32) * stdev
            interp = tfp.math.interp_regular_1d_grid(t_interp, t_interp[0], t_interp[-1], delta_u, axis=1)
            delta_u = interp[:,:self.mppi_samples,:]
        else:
            #otherwise i.i.d. generation
            delta_u = random_gen.normal([self.num_rollouts, self.mppi_samples, self.num_control_inputs], dtype=tf.float32) * stdev
        return delta_u

    @Compile
    def mppi_prior(self, s, target_position, u_nom, random_gen, u_old):
        # generate random input sequence and clip to control limits
        delta_u = self.inizialize_pertubation(random_gen)
        u_run = tf.tile(u_nom, [self.num_rollouts, 1, 1]) + delta_u
        u_run = tf.clip_by_value(u_run, -1.0, 1.0)
        #predict trajectories
        rollout_trajectory = self.predictor.predict_tf(s, u_run)
        #rollout cost
        traj_cost = self.mppi_cost(rollout_trajectory, u_run, target_position, u_old, delta_u)
        #retrive control sequence via path integral
        u_nom = tf.clip_by_value(u_nom + self.reward_weighted_average(traj_cost, delta_u), -1.0, 1.0)
        return u_nom

    @Compile
    def grad_step(self, s, target_position, Q, opt):
        #do a gradient descent step
        #setup gradient tape
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            #rollout trajectory and retrive cost
            rollout_trajectory = self.predictor.predict_tf(s, Q)
            traj_cost = self.cost(rollout_trajectory, Q, target_position, self.u)
        #retrieve gradient of cost w.r.t. input sequence
        dc_dQ = tape.gradient(traj_cost, Q)
        #modify gradients: makes sure biggest entry of each gradient is at most "gradmax_clip". (For this controller only one sequence
        dc_dQ_max = tf.math.reduce_max(tf.abs(dc_dQ), axis=1, keepdims=True) #find max gradient for every sequence
        mask = (dc_dQ_max > self.gradmax_clip) #generate binary mask
        invmask = tf.logical_not(mask)
        dc_dQ_prc = ((dc_dQ / dc_dQ_max) * tf.cast(mask, tf.float32) * self.gradmax_clip + dc_dQ * tf.cast(
            invmask, tf.float32)) #modify gradients
        #use optimizer to applay gradients and retrieve next set of input sequences
        opt.apply_gradients(zip([dc_dQ_prc], [Q]))
        #clip
        Qn = tf.clip_by_value(Q, -1, 1)
        return Qn, traj_cost

    #step function to find control
    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        # tile inital state and convert inputs to tensorflow tensors
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        target_position = tf.convert_to_tensor(target_position, dtype=tf.float32)

        #first retrieve suboptimal control sequence with mppi
        Q_mppi = self.mppi_prior(s, target_position, self.Q_opt, self.rng_mppi, self.u)
        self.Q_opt.assign(Q_mppi)

        #optimize control sequence with gradient based optimization
        for _ in range(self.optim_steps):
            Q_opt, traj_cost = self.grad_step(s, target_position, self.Q_opt, self.opt)
            self.Q_opt.assign(Q_opt)

        self.u = self.Q_opt[0, 0, :]
        self.Q, self.J = self.Q_opt.numpy(), traj_cost.numpy()
        self.Q_opt.assign(tf.concat([self.Q_opt[:, 1:, :], tf.zeros([1,1,self.num_control_inputs])], axis=1)) #shift and initialize new input with 0
        #reset adam optimizer
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])
        return np.squeeze(self.u.numpy())

    def controller_reset(self):
        #reset prototype control sequence
        self.Q_opt.assign(tf.zeros([1, self.mppi_samples, self.num_control_inputs], dtype=tf.float32))
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