"""mpc controller"""

import scipy.optimize
import numpy as np

from CartPole.state_utilities import create_cartpole_state, cartpole_state_varname_to_index, \
    cartpole_state_indices_to_varnames
from Predictores.predictor_ideal import predictor_ideal

from Controllers.controller_lqr import controller_lqr

import matplotlib.pyplot as plt
import numpy as np

import yaml
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

predictor = predictor_ideal
# WARNING: if using RNN to provide CartPole model to MPC
# make sure that it is trained to predict future states with this timestep
DT = config['controller']['custom_mpc_scipy']['DT']

# method = 'L-BFGS-B'
method = config['controller']['custom_mpc_scipy']['method']
ftol = config['controller']['custom_mpc_scipy']['ftol']
mpc_horizon = config['controller']['custom_mpc_scipy']['mpc_horizon']

# weights
wr = config['controller']['custom_mpc_scipy']['wr']

l1 = config['controller']['custom_mpc_scipy']['l1']
l1_2 = config['controller']['custom_mpc_scipy']['l1_2']
l2 = config['controller']['custom_mpc_scipy']['l2']
l3 = config['controller']['custom_mpc_scipy']['l3']
l4 = config['controller']['custom_mpc_scipy']['l4']

m1 = config['controller']['custom_mpc_scipy']['m1']
m2 = config['controller']['custom_mpc_scipy']['m2']
m3 = config['controller']['custom_mpc_scipy']['m3']
m4 = config['controller']['custom_mpc_scipy']['m4']

# w_sum = wr + l1 + l2 + l3 + m1 + m2 + m3 + m4
w_sum = 1.0

wr /= w_sum
l1 /= w_sum
l2 /= w_sum
l3 /= w_sum
m1 /= w_sum
m2 /= w_sum
m3 /= w_sum
m4 /= w_sum


class controller_custom_mpc_scipy:
    def __init__(self):

        # LQR to stabilize pole in the starting phase
        self.lqr_controller = controller_lqr()

        self.horizon = mpc_horizon

        self.Predictor = predictor(horizon=self.horizon, dt=DT)

        self.rnn_eval_time = []
        self.predictor_time = []
        self.nfun = []

        # I do the norm and unnorm unnecessarilly!
        # You need only to scale once!
        """
        Get configured do-mpc modules:
        """

        # State of the cart
        self.s = create_cartpole_state()  # s like state

        self.target_position = 0.0

        self.Q_hat = np.zeros(self.horizon)  # MPC prediction of future control inputs
        self.Q_hat0 = np.random.normal(0.0, 0.1 ,self.horizon)  # initial guess for future control inputs to be predicted
        self.Q_previous = 0.0

        self.angle_cost = lambda angle: angle ** 2
        self.angle_sin_cost = lambda angle_sin: angle_sin ** 2
        self.angleD_cost = lambda angleD: angleD ** 2
        self.position_cost = lambda position: (position - self.target_position) ** 2
        self.positionD_cost = lambda positionD: positionD ** 2

        self.Q_bounds = scipy.optimize.Bounds(lb=-1.0, ub=1.0)

        self.initial_state = create_cartpole_state()

        self.prediction_features_names = cartpole_state_indices_to_varnames(range(len(self.s)))
        self.predictions = np.zeros((self.horizon + 1, len(self.prediction_features_names) + 1))

        # Array keeping individual costs for every timestep of prediction
        self.costs_names = ['angle_cost', 'angle_sin_cost', 'angleD_cost', 'position_cost', 'positionD_cost']
        self.cost_array = np.zeros((self.horizon + 1, len(self.costs_names) + 1))

        # Numbers of samples which should be bridged with output from a P controller
        # To give RNN time to settle
        self.warm_up_len = 0
        self.sample_counter = 0  # Counting samples for above aim

    def cost_function(self, Q_hat):

        # Predict future states given control_inputs Q_hat
        self.predictions = self.Predictor.predict(Q_hat)

        self.cost_array[:, self.costs_names.index('angle_cost')] = \
            self.angle_cost(self.predictions[:, cartpole_state_varname_to_index('angle')])
        self.cost_array[:, self.costs_names.index('angle_sin_cost')] = \
            self.angle_sin_cost(self.predictions[:, cartpole_state_varname_to_index('angle_sin')])
        self.cost_array[:, self.costs_names.index('angleD_cost')] = \
            self.angleD_cost(self.predictions[:, cartpole_state_varname_to_index('angleD')])
        self.cost_array[:, self.costs_names.index('position_cost')] = \
            self.position_cost(self.predictions[:, cartpole_state_varname_to_index('position')])
        self.cost_array[:, self.costs_names.index('positionD_cost')] = \
            self.positionD_cost(self.predictions[:, cartpole_state_varname_to_index('positionD')])

        # Calculate l-cost for every timestep
        self.cost_array[:, -1] = (
                                  + l1 * self.cost_array[:, self.costs_names.index('angle_cost')]
                                  + l1_2 * self.cost_array[:, self.costs_names.index('angle_sin_cost')]
                                  + l2 * self.cost_array[:, self.costs_names.index('angleD_cost')]
                                  + l3 * self.cost_array[:, self.costs_names.index('position_cost')]
                                  + l4 * self.cost_array[:, self.costs_names.index('positionD_cost')])

        l_terms = np.sum(self.cost_array[:, -1])

        # Calculate sum of r-terms
        r_terms = wr * ((Q_hat[0] - self.Q_previous) ** 2)
        r_terms += wr * sum((Q_hat[1:] - Q_hat[:-1]) ** 2)

        m_term = (m1 * self.cost_array[-1, self.costs_names.index('angle_sin_cost')]
                  + m2 * self.cost_array[-1, self.costs_names.index('angleD_cost')]
                  + m3 * self.cost_array[-1, self.costs_names.index('position_cost')]
                  + m4 * self.cost_array[-1, self.costs_names.index('positionD_cost')])

        cost = r_terms + m_term + l_terms

        return cost

    def step(self, s, target_position, time=None):

        self.s[...] = s
        self.target_position = target_position

        # Setup Predictor
        self.Predictor.setup(initial_state=self.s, prediction_denorm=True)

        if self.sample_counter >= self.warm_up_len:
            # Solve Optimization problem
            # solution = scipy.optimize.basinhopping(self.cost_function, self.Q_hat0, niter=2,
            #                                        minimizer_kwargs={ "method": method,"bounds":self.Q_bounds })
            # self.Q_hat0 = solution.x

            # self.Q_hat0 = np.clip(self.Q_hat0*(1+0.01*np.random.uniform(-1.0,1.0)), -1, 1)
            solution = scipy.optimize.minimize(self.cost_function, self.Q_hat0,
                                               bounds=self.Q_bounds, method=method,
                                               options={'ftol': ftol})

            self.Q_hat = solution.x

            # Compose new initial guess
            self.Q_previous = Q0 = self.Q_hat[0]
            # self.Q_hat0 = np.hstack((self.Q_hat[1:], self.Q_hat[-1]))
            self.Q_hat0 = self.Q_hat + np.random.normal(0.0, 0.01, self.horizon)
            # self.plot_prediction()
        else:
            Q0 = self.lqr_controller.step(s, target_position)

        # Make predictor ready for the next timestep
        self.Predictor.update_internal_state(Q0)

        # Conclude by collecting/printing some info about this iteration
        # self.nfun.append(solution.nfev)
        # print(solution)

        # for i in range(len(self.prediction_features_names)):
        #     plt.figure()
        #     plt.title(self.prediction_features_names[i])
        #     plt.plot(range(self.horizon+1), self.predictions[:, i], label=self.prediction_features_names[i])
        # plt.figure()
        # plt.title('Q')
        # plt.plot(range(self.horizon + 1), self.predictions[:, -1], label='Q')
        # plt.show()

        return Q0

    def controller_reset(self):
        self.Predictor.net.reset_states()
        self.Q_hat0 = self.Q_hat = np.zeros(self.horizon)
        self.sample_counter = 0

    def controller_summary(self):
        print('******************************************************************')
        print('Controller summary:')
        print('Optimizer name: {}'.format(method))
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

        if 'angle' in self.predictions:
            # raise KeyError('You should not be there')
            angle = np.rad2deg(self.predictions['angle'].to_numpy())
        elif ('angle_sin' in self.predictions) and ('angle_cos' in self.predictions):
            angle = np.rad2deg(
                np.arctan2(self.predictions['angle_sin'].to_numpy(), self.predictions['angle_cos'].to_numpy()))
        else:
            raise ValueError('No data for angle in self.predictions')

        angleD = self.predictions['angleD'].to_numpy()
        position = self.predictions['position'].to_numpy()
        positionD = self.predictions['positionD'].to_numpy()

        # Plot angle
        self.axs[0].set_ylabel("Angle (deg)", fontsize=18)
        self.axs[0].plot(angle, color='b',
                         markersize=12, label='Angel prediction')
        self.axs[0].tick_params(axis='both', which='major', labelsize=16)

        self.axs[1].set_ylabel("AngleD (deg/s)", fontsize=18)
        self.axs[1].plot(angleD, color='r',
                         markersize=12, label='AngelD prediction')
        self.axs[1].tick_params(axis='both', which='major', labelsize=16)

        self.axs[2].set_ylabel("Position (m)", fontsize=18)
        self.axs[2].plot(position, color='green',
                         markersize=12, label='Position prediction')
        self.axs[2].tick_params(axis='both', which='major', labelsize=16)

        self.axs[3].set_ylabel("PositionD (m/s)", fontsize=18)
        self.axs[3].plot(positionD, color='magenta',
                         markersize=12, label='PositionD prediction')
        self.axs[3].tick_params(axis='both', which='major', labelsize=16)

        self.axs[4].set_ylabel("Motor (-1,1)", fontsize=18)
        self.axs[4].plot(self.Q_hat, color='orange',
                         markersize=12, label='Force of motor')
        self.axs[4].tick_params(axis='both', which='major', labelsize=16)

        self.axs[4].set_xlabel('Time (samples)', fontsize=18)

        self.fig.align_ylabels()

        plt.show()

        return self.fig, self.axs
