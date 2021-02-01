"""mpc controller"""

from types import SimpleNamespace

import scipy.optimize

from copy import deepcopy

from modeling.rnn_tf.utilis_rnn import *
from predictores.predictor_autoregressive_tf import predictor_autoregressive_tf
from predictores.predictor_ideal import predictor_ideal

predictor = predictor_ideal
DT = 0.2

# method = 'L-BFGS-B'
method = 'SLSQP'
maxiter = 80 # I think it was a key thing.
maxiter = 800
ftol = 1.0e-8
mpc_horizon = 5

# weights
wr = 1.0  # rterm
l1 = 5.0  # -pot
l2 = 50.0  # distance
l3 = 0.0  # kin_pol
m1 = 0.0  # kin_pol
m2 = 20.0  # -pot
m3 = 0.0  # kin_cart
m4 = 20.0*100.0  # distance

w_sum = wr + l1 + l2 + l3 + m1 + m2 + m3 + m4

wr /= w_sum
l1 /= w_sum
l2 /= w_sum
l3 /= w_sum
m1 /= w_sum
m2 /= w_sum
m3 /= w_sum
m4 /= w_sum

class controller_custom_mpc:
    def __init__(self):


        self.Predictor = predictor(horizon=mpc_horizon, dt=DT)

        self.rnn_eval_time = []
        self.predictor_time = []
        self.nfun = []

        # I do the norm and unnorm unnecessarilly!
        # You need only to scale once!
        """
        Get configured do-mpc modules:
        """

        # State of the cart
        self.s = SimpleNamespace()  # s like state

        self.target_position = 0.0
        self.target_position_normed = 0.0

        self.mpc_horizon = mpc_horizon

        self.yp_hat = np.zeros(self.mpc_horizon, dtype=object)  # MPC prediction of future states
        self.Q_hat = np.zeros(self.mpc_horizon)  # MPC prediction of future control inputs
        self.Q_hat0 = np.zeros(self.mpc_horizon)  # initial guess for future control inputs to be predicted
        self.Q_previous = 0.0

        self.E_kin_cart = lambda positionD: (positionD) ** 2
        self.E_kin_pol = lambda angleD: (angleD) ** 2
        self.E_pot_cost = lambda angle: 1 - np.cos(angle)
        self.E_pot = lambda cos_angle: cos_angle
        self.E_pot_cost = lambda cos_angle: 1 - cos_angle

        self.distance_difference = lambda position: ((position - self.target_position_normed))**2

        self.Q_bounds = scipy.optimize.Bounds(lb=-1.0, ub=1.0)

        self.initial_state = pd.DataFrame(0, index=np.arange(1), columns=['s.angle.cos', 's.angle.sin', 's.angleD', 's.position', 's.positionD'])

        self.step_number = 0
        self.warm_up_len = 20

    def cost_function(self, Q_hat):
        t0 = timeit.default_timer()
        # Predict future states given control_inputs Q_hat
        self.predictions = copy.copy(self.Predictor.predict(Q_hat))

        t1 = timeit.default_timer()

        self.predictions['E_pot'] = self.E_pot(self.predictions['s.angle.cos'])
        self.predictions['E_kin_pol'] = self.E_kin_pol(self.predictions['s.angleD'])
        self.predictions['E_kin_cart'] = self.E_kin_cart(self.predictions['s.positionD'])
        self.predictions['distance_difference'] = self.distance_difference(self.predictions['s.position'])

        self.predictions['lterm'] = - l1 * self.predictions['E_pot'] + \
                                        l2 * self.predictions['distance_difference'] + \
                                             l3 * self.predictions['E_kin_pol']

        # print(self.predictions['distance_difference'])

        # Calculate sum of r-terms
        r_terms = wr * ((Q_hat[0] - self.Q_previous) ** 2)
        r_terms += wr * sum((Q_hat[1:] - Q_hat[:-1]) ** 2)


        l_terms = self.predictions['lterm'].sum()

        m_term = (m1 * self.predictions['E_kin_pol'].iloc[-1]
                    - m2 * self.predictions['E_pot'].iloc[-1]
                        + m3 * self.predictions['E_kin_cart'].iloc[-1]
                            + m4 * self.predictions['distance_difference'].iloc[-1])

        cost = r_terms + m_term + l_terms

        # t2 = timeit.default_timer()
        # print('cost function eval {} ms'.format((t2-t0)*1000.0))
        # print('predictor eval {} ms'.format((t1-t0)*1000.0))
        # self.predictor_time.append((t1-t0)*1000.0)
        # print('predictor/all {}%'.format(np.round(100*(t1-t0)/(t2-t0))))

        return cost

    def step(self, s, target_position):
        # TODO: Step should get already a dataframe/dataseries from which it should pick the columns it needs
        #   Optimally you should operate on the normed values all the time, try to eliminate devisions
        # IMPORTANT: take care how often it is called!!!

        # get initial state

        self.s = deepcopy(s)
        self.target_position = deepcopy(target_position)

        self.initial_state['s.angle.cos'] = [np.cos(s.angle)]
        self.initial_state['s.angle.sin'] = [np.sin(s.angle)]
        self.initial_state['s.angleD'] = [s.angleD]
        self.initial_state['s.position'] = [s.position]
        self.initial_state['s.positionD'] = [s.positionD]

        # Setup Predictor
        self.Predictor.setup(initial_state=self.initial_state, prediction_denorm=False)

        # FIXME: You are now norming target position manually...
        self.target_position_normed = self.target_position/50.0

        if self.step_number > self.warm_up_len:
            # Solve Optimization problem
            # solution = scipy.optimize.basinhopping(self.cost_function, self.Q_hat0, niter=2,
            #                                        minimizer_kwargs={ "method": method,"bounds":self.Q_bounds })
            # self.Q_hat0 = solution.x

            # self.Q_hat0 = np.clip(self.Q_hat0*(1+0.01*np.random.uniform(-1.0,1.0)), -1, 1)
            solution = scipy.optimize.minimize(self.cost_function, self.Q_hat0,
                                               bounds=self.Q_bounds, method=method,
                                               options={'maxiter': maxiter, 'ftol': ftol})

            self.Q_hat = solution.x

            # Compose new initial guess
            self.Q_previous = Q0 = self.Q_hat[0]
            self.Q_hat0 = np.hstack((self.Q_hat[1:], self.Q_hat[-1]))
            # self.plot_prediction()
        else:
            self.Q_previous = Q0 = -np.sin(s.angle) * 0.2
            self.step_number = self.step_number + 1



        # Make predictor ready for the next timestep
        self.Predictor.update_internal_state(Q0)

        # Conclude by collecting/printing some info about this iteration
        # self.nfun.append(solution.nfev)
        # print(solution)


        return Q0

    def reset(self):
        self.Predictor.net.reset_states()
        self.Q_hat0 = np.zeros(self.mpc_horizon)
        self.step_number = 0

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

        if 's.angle' in self.predictions:
            # raise KeyError('You should not be there')
            angle = np.rad2deg(self.predictions['s.angle'].to_numpy())
        elif ('s.angle.sin' in self.predictions) and ('s.angle.cos' in self.predictions):
            angle = np.rad2deg(np.arctan2(self.predictions['s.angle.sin'].to_numpy(), self.predictions['s.angle.cos'].to_numpy()))
        else:
            raise ValueError('No data for angle in self.predictions')

        angleD = self.predictions['s.angleD'].to_numpy()
        position = self.predictions['s.position'].to_numpy()
        positionD = self.predictions['s.positionD'].to_numpy()

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