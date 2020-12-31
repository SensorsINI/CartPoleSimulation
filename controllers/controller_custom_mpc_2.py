"""mpc controller"""

from src.globals import *
from types import SimpleNamespace

import scipy.optimize

from copy import deepcopy

import matplotlib.pyplot as plt

import timeit

from modeling.rnn_tf.utilis_rnn import *

# method = 'L-BFGS-B'
method = 'SLSQP'
maxiter = 80 # I think it was a key thing.
ftol = 1.0e-5
mpc_horizon = 2

RNN_FULL_NAME = 'GRU-4IN-1024H1-1024H2-2OUT-2'
# RNN_FULL_NAME = 'GRU-4IN-8H1-8H2-2OUT-0'
INPUTS_LIST = ['s.angle', 's.position', 'target_position', 'u']
OUTPUTS_LIST = ['s.angle', 's.position']
PATH_SAVE = './controllers/nets/mpc_on_rnn_tf/'

# weights
wr = 0.0  # rterm
l1 = 2.0  # -pot
l2 = 0.0  # distance
l3 = 0.0  # kin_pol
m1 = 0.0  # kin_pol
m2 = 2.0  # -pot
m3 = 0.0  # kin_cart
m4 = 0.0  # distance

w_sum = wr + l1 + l2 + l3 + m1 + m2 + m3

wr /= w_sum
l1 /= w_sum
l2 /= w_sum
l3 /= w_sum
m1 /= w_sum
m2 /= w_sum
m3 /= w_sum


class controller_custom_mpc_2:
    def __init__(self):

        # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
        self.net, self.rnn_name, self.inputs_list, self.outputs_list \
            = create_rnn_instance(load_rnn=RNN_FULL_NAME, path_save=PATH_SAVE,
                                  return_sequence=False, stateful=True,
                                  warm_up_len=1, batchSize=1)

        self.normalization_info = load_normalization_info(PATH_SAVE, RNN_FULL_NAME)

        self.rnn_initial_input = pd.DataFrame(columns=self.inputs_list)
        self.rnn_input = pd.DataFrame(columns=self.inputs_list)
        self.rnn_output = pd.DataFrame(columns=self.outputs_list)

        # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
        self.net, self.rnn_name, self.inputs_list, self.outputs_list \
            = create_rnn_instance(load_rnn=RNN_FULL_NAME, path_save=PATH_SAVE,
                                  return_sequence=False, stateful=True,
                                  warm_up_len=1, batchSize=1)

        self.rnn_internal_states = self.net.get_internal_states()

        self.rnn_eval_time = []
        self.predictor_time = []
        self.nfun = []

        # I do the norm and unnorm unnecessarilly!
        # You need only to scale once!
        """
        Get configured do-mpc modules:
        """

        # Physical parameters of the cart
        self.p = SimpleNamespace()  # p like parameters
        self.p.m = m_globals  # mass of pend, kg
        self.p.M = M_globals  # mass of cart, kg
        self.p.L = L_globals  # half length of pend, m
        self.p.u_max = u_max_globals  # max cart force, N
        self.p.M_fric = M_fric_globals  # cart friction, N/m/s
        self.p.J_fric = J_fric_globals  # friction coefficient on angular velocity, Nm/rad/s
        self.p.v_max = v_max_globals  # max DC motor speed, m/s, in absence of friction, used for motor back EMF model
        self.p.controlDisturbance = controlDisturbance_globals  # disturbance, as factor of u_max
        self.p.sensorNoise = sensorNoise_globals  # noise, as factor of max values
        self.p.g = g_globals  # gravity, m/s^2
        self.p.k = k_globals  # Dimensionless factor, for moment of inertia of the pend (with L being half if the length)

        # State of the cart
        self.s = SimpleNamespace()  # s like state

        self.target_position = 0.0
        self.target_position_normed = 0.0

        self.mpc_horizon = mpc_horizon
        self.dt = dt_mpc_simulation_globals
        self.f = 1.0/self.dt  # We prefare to perform multiplication than division

        self.yp_hat = np.zeros(self.mpc_horizon, dtype=object)  # MPC prediction of future states
        self.Q_hat = np.zeros(self.mpc_horizon)  # MPC prediction of future control inputs
        self.Q_hat0 = np.zeros(self.mpc_horizon)  # initial guess for future control inputs to be predicted
        self.Q_previous = 0.0

        self.E_kin_cart = lambda positionD: (positionD) ** 2
        self.E_kin_pol = lambda angleD: (angleD) ** 2
        self.E_pot_cost = lambda angle: 1 - np.cos(angle)
        self.E_pot = lambda angle: np.cos(angle)**2
        self.distance_difference = lambda position: ((position - self.target_position_normed))**2

        # self.Q_bounds = [(-1, 1)] * self.mpc_horizon
        self.Q_bounds = scipy.optimize.Bounds(lb=-1.0, ub=1.0)
        self.max_mterm = 0.0
        self.max_rterms = 0.0
        self.max_l_terms = 0.0
        self.E_kin_cart_max = 0.0
        self.E_kin_pol_max = 0.0
        self.E_pot_cost_max = 0.0
        self.distance_difference_max = 0.0

        self.initial_state = pd.DataFrame(columns=['s.angle', 's.angleD', 's.position', 's.positionD'])

    def step_rnn(self, rnn_input):

        rnn_input = np.squeeze(rnn_input.to_numpy())
        rnn_input = rnn_input[np.newaxis, np.newaxis, :]
        rnn_input = tf.convert_to_tensor(rnn_input, dtype=tf.float64)

        # t00 = timeit.default_timer()
        rnn_output = self.net.predict_on_batch(rnn_input)
        # t11 = timeit.default_timer()

        rnn_output = np.squeeze(rnn_output).tolist()
        rnn_output = deepcopy(pd.DataFrame(data=[rnn_output], columns=self.outputs_list))

        # self.rnn_eval_time.append((t11 - t00) * 1.0e6)

        return rnn_output

    def predictor(self, Q_hat):

        self.net.load_internal_states(self.rnn_internal_states)
        prediction = pd.DataFrame(data=np.zeros((self.mpc_horizon+1, len(self.initial_state_normed.columns))),
                                  columns=self.initial_state_normed.columns)

        # FIXME: This is ugly. You should just train it on Q
        #   Otherwise if u(Q,p) is known you can train RNN on normed u,
        #   optimize for normed u, and find Q analytically
        #   For the moment as normed u is approx Q we plug Q for the RNN trained on normed u
        if 'u' in self.rnn_input:
            self.rnn_initial_input['u'] = [Q_hat[0]]

        for col_name in prediction:
            if col_name in self.initial_state_normed:
                prediction.loc[prediction.index[0], col_name] = self.initial_state_normed.loc[
                    self.initial_state_normed.index[0], col_name]

        for k in range(0, self.mpc_horizon):

            if k == 0:
                self.rnn_input = deepcopy(self.rnn_initial_input)
            else:
                for col_name in self.rnn_input.columns:
                    if col_name in self.rnn_output:
                        self.rnn_input[col_name] = self.rnn_output[col_name]
                if 'target_position' in self.rnn_input:
                    self.rnn_input['target_position'] = self.target_position_normed
                if 'u' in self.rnn_input:
                    self.rnn_input['u'] = [Q_hat[k]]


            self.rnn_output = self.step_rnn(self.rnn_input)

            for col_name in prediction:
                if col_name in self.rnn_output:
                    prediction.loc[prediction.index[k+1], col_name] = self.rnn_output.loc[self.rnn_output.index[0], col_name]

        if 's.positionD' not in self.outputs_list:
            # Remark: first line which is initial state already has D
            prediction['s.positionD'] = (prediction['s.position']-prediction['s.position'].shift(1)) * self.f
            prediction.loc[prediction.index[0], 's.positionD'] = self.initial_state_normed.loc[self.initial_state_normed.index[0], 's.positionD']

        if 's.angleD' not in self.outputs_list:
            prediction['s.angleD'] = (prediction['s.angle']-prediction['s.angle'].shift(1)) * self.f
            prediction.loc[prediction.index[0], 's.angleD'] = self.initial_state_normed.loc[prediction.index[0], 's.angleD']

        return prediction

    def cost_function(self, Q_hat):
        # t0 = timeit.default_timer()
        # Predict future states given control_inputs Q_hat
        self.predictions = self.predictor(Q_hat)

        # t1 = timeit.default_timer()

        self.predictions['E_pot'] = self.E_pot(self.predictions['s.angle'])
        self.predictions['E_kin_pol'] = self.E_kin_pol(self.predictions['s.angleD'])
        self.predictions['E_kin_cart'] = self.E_kin_cart(self.predictions['s.positionD'])
        self.predictions['distance_difference'] = self.distance_difference(self.predictions['s.position'])

        self.predictions['lterm'] = - l1 * self.predictions['E_pot'] + \
                                        l2 * self.predictions['distance_difference'] + \
                                             l3 * self.predictions['E_kin_pol']
        cost = 0.0



        # Calculate sum of r-terms
        r_terms = wr * ((Q_hat[0] - self.Q_previous) ** 2)
        r_terms += wr * sum((Q_hat[1:] - Q_hat[:-1]) ** 2)


        l_terms = self.predictions['lterm'].sum()

        m_term = (m1 * self.predictions['E_kin_pol'].iloc[-1]
                    - m2 * self.predictions['E_pot'].iloc[-1]
                        + m3 * self.predictions['E_kin_cart'].iloc[-1]
                            + m4 * self.predictions['distance_difference'].iloc[-1])

        cost += r_terms + m_term + l_terms

        # t2 = timeit.default_timer()
        # print('cost function eval {} ms'.format((t2-t0)*1000.0))
        # print('predictor eval {} ms'.format((t1-t0)*1000.0))
        # self.predictor_time.append((t1-t0)*1000.0)
        # print('predictor/all {}%'.format(np.round(100*(t1-t0)/(t2-t0))))

        return cost

    def step(self, s, target_position):
        # TODO: Step should get already a dataframe/dataseries from which it should pick the columns it needs
        #   Optimally you should operate on the normed values all the time, try to eliminate devisions

        self.rnn_internal_states = self.net.get_internal_states()

        self.s = deepcopy(s)
        self.target_position = deepcopy(target_position)

        self.initial_state['s.position'] = [s.position]
        self.initial_state['s.angle'] = [s.angle]
        self.initial_state['s.positionD'] = [s.positionD]
        self.initial_state['s.angleD'] = [s.angleD]

        self.initial_state_normed = normalize_df(self.initial_state, self.normalization_info)
        self.target_position_normed = normalize_feature(self.target_position, self.normalization_info,
                                                        name='s.position')

        for col_name in self.rnn_input.columns:
            if col_name in self.initial_state_normed:
                self.rnn_initial_input[col_name] = self.initial_state_normed[col_name]
        if 'target_position' in self.rnn_input:
            self.rnn_initial_input['target_position'] = [self.target_position_normed]


        #  FIXME: IT IS NOT GOOD THE NORMALIZATION OF Q
        #    FOR THE MOMEMT I ASSUME THAT NORMED AND UNNORMED Q IS THE SAME,
        #    WHICH IS IN THE SPECIAL CASE OF MY DATASET AND NOTMALIZATION PROCEDURE EVEN EXACTLY TRUE
        solution = scipy.optimize.minimize(self.cost_function, self.Q_hat0, bounds=self.Q_bounds, method=method,
                                           options={'maxiter': maxiter, 'ftol': ftol})
        self.Q_hat = solution.x
        # self.nfun.append(solution.nfev)
        print(solution)

        self.Q_hat0 = np.hstack((self.Q_hat[1:], self.Q_hat[-1]))
        self.Q_previous = self.Q_hat[0]

        Q = self.Q_hat[0]

        self.net.load_internal_states(self.rnn_internal_states)
        self.step_rnn(self.rnn_initial_input)

        return Q

    def reset(self):
        self.net.reset_states()
        self.Q_hat0 = np.zeros(self.mpc_horizon)

    def controller_summary(self):
        print('******************************************************************')
        print('Controller summary:')
        print('Optimizer name: {}'.format(method))
        print('Model name: {}'.format(RNN_FULL_NAME))
        print('Number of timesteps: {}'.format(len(self.nfun)))
        print('Number of RNN evaluations: {}'.format(len(self.rnn_eval_time)))
        print('')
        print('--------------------')
        print('RNN evaluation time:')
        print('Average: {} us'.format(np.around(np.mean(self.rnn_eval_time))))
        print('Std: {} us'.format(np.around(np.std(self.rnn_eval_time))))
        print('Min: {} us'.format(np.around(np.min(self.rnn_eval_time))))
        print('Max: {} us'.format(np.around(np.max(self.rnn_eval_time))))
        print('--------------------')
        print('Predictor evaluation time:')
        print('Average: {} ms'.format(np.around(np.mean(self.predictor_time))))
        print('Std: {} ms'.format(np.around(np.std(self.predictor_time))))
        print('Min: {} ms'.format(np.around(np.min(self.predictor_time))))
        print('Max: {} ms'.format(np.around(np.max(self.predictor_time))))
        print('--------------------')
        print('number of cost function (=number of predictor) evaluations per time step:')
        print('Average: {}'.format(np.around(np.mean(self.nfun))))
        print('Std: {}'.format(np.around(np.std(self.nfun))))
        print('Min: {}'.format(np.around(np.min(self.nfun))))
        print('Max: {}'.format(np.around(np.max(self.nfun))))

        print('******************************************************************')


















    def plot_prediction(self):

        self.fig, self.axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)  # share x axis so zoom zooms all plots

        angle = []
        angleD = []
        position = []
        positionD = []
        for s in self.yp_hat:
            angle.append(s.angle)
            angleD.append(s.angleD)
            position.append(s.position)
            positionD.append(s.positionD)

        # Plot angle
        self.axs[0].set_ylabel("Angle (deg)", fontsize=18)
        self.axs[0].plot(np.array(angle) * 180.0 / np.pi, color='b',
                         markersize=12, label='Angel prediction')
        self.axs[0].tick_params(axis='both', which='major', labelsize=16)

        self.axs[1].set_ylabel("AngleD (deg/s)", fontsize=18)
        self.axs[1].plot(np.array(angleD), color='r',
                         markersize=12, label='AngelD prediction')
        self.axs[1].tick_params(axis='both', which='major', labelsize=16)

        self.axs[2].set_ylabel("Position (m)", fontsize=18)
        self.axs[2].plot(np.array(position), color='green',
                         markersize=12, label='Position prediction')
        self.axs[2].tick_params(axis='both', which='major', labelsize=16)

        self.axs[3].set_ylabel("PositionD (m/s)", fontsize=18)
        self.axs[3].plot(np.array(positionD), color='magenta',
                         markersize=12, label='PositionD prediction')
        self.axs[3].tick_params(axis='both', which='major', labelsize=16)

        self.axs[4].set_ylabel("Motor (-1,1)", fontsize=18)
        self.axs[4].plot(np.array(self.Q_hat), color='orange',
                         markersize=12, label='Force of motor')
        self.axs[4].tick_params(axis='both', which='major', labelsize=16)

        self.axs[4].set_xlabel('Time (samples)', fontsize=18)

        self.fig.align_ylabels()

        plt.show()

        return self.fig, self.axs
