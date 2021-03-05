"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""



"""
Using predictor:
1. Initialize while initializing controller
    This step load the RNN (if applies) - it make take quite a bit of time
    During initialization you only need to provide RNN which should be loaded
2. Call iterativelly three functions
    a) setup(initial_state, horizon, etc.)
    b) predict(Q)
    c) update_rnn
    
    ad a) at this stage you can change the parameters for prediction like e.g. horizon, dt
            It also prepares 0 state of the prediction, and tensors for saving the results,
            to make b) max performance. This function should be called BEFORE starting solving an optim
    ad b) predict is optimized to get the prediction of future states of the system as fast as possible.
        It accepts control input (vector) as its only input and is intended to be used at every evaluation of the cost functiomn
    ad c) this method updates the internal state of RNN (if applies). It accepts control input for current time step (scalar) as its only input
            it should be called only after the optimization problem is solved with the control input used in simulation
            
"""

#TODO: for the moment it is not possible to update RNN more often than mpc dt
#   Updating it more often will lead to false results.


from Modeling.load_and_normalize import load_normalization_info, normalize_df
from CartPole.cartpole_model import p_globals, Q2u

from types import SimpleNamespace

import numpy as np
from copy import deepcopy
import pandas as pd
import copy
import timeit

from CartPole.cartpole_model import cartpole_ode

RNN_FULL_NAME = 'GRU-6IN-64H1-64H2-5OUT-0' # You need it to get normalization info
RNN_PATH = './save_tf/'
# RNN_PATH = './controllers/nets/mpc_on_rnn_tf/'
PREDICTION_FEATURES_NAMES = ['s.angle.cos', 's.angle.sin', 's.angle', 's.angleD', 's.position', 's.positionD']
PATH_TO_NORMALIZATION_INFO = './Modeling/NormalizationInfo/' + 'NI_2021-03-01_11-51-13.csv'

def mpc_next_state(s, p, u, dt):
    """Wrapper for CartPole ODE. Given a current state (without second derivatives), returns a state after time dt
    """

    s_next = s

    s_next.angleDD, s_next.positionDD = cartpole_ode(p, s_next, u)  # Calculates CURRENT second derivatives

    # Calculate NEXT state:
    s_next.position = s.position + s.positionD * dt
    s_next.positionD = s.positionD + s.positionDD * dt

    s_next.angle = s.angle + s.angleD * dt
    s_next.angleD = s.angleD + s.angleDD * dt

    return s_next


class predictor_ideal:
    def __init__(self, horizon, dt):

        self.normalization_info = load_normalization_info(PATH_TO_NORMALIZATION_INFO)

        # Physical parameters of the cart
        self.p = p_globals

        # State of the cart
        self.s = SimpleNamespace()  # s like state


        self.target_position = 0.0
        self.target_position_normed = 0.0

        self.horizon = horizon

        self.dt = dt

        self.prediction_features_names = PREDICTION_FEATURES_NAMES
        self.prediction_denorm = False

        # self.prediction_list = pd.DataFrame(columns=PREDICTION_FEATURES_NAMES, index=range(horizon + 1))
        self.prediction_list = pd.DataFrame(data=np.zeros((self.horizon+1, len(PREDICTION_FEATURES_NAMES)+1)),
                                            columns=['Q'] + PREDICTION_FEATURES_NAMES)

        pass

    def setup(self, initial_state: pd.DataFrame, prediction_denorm=False):

        if ('s.angle' in initial_state.columns):
            self.s.angle = initial_state['s.angle'].to_numpy().squeeze()
        elif ('s.angle.cos' in initial_state.columns) and ('s.angle.sin' in initial_state.columns):
            self.s.angle = np.arctan2(initial_state['s.angle.sin'].to_numpy(), initial_state['s.angle.cos'].to_numpy())
        else:
            raise ValueError('Angle info missing')

        if ('s.angle.cos' in initial_state.columns) and ('s.angle.sin' in initial_state.columns):
            self.s.angle_cos = initial_state['s.angle.cos'].to_numpy().squeeze()
            self.s.angle_sin = initial_state['s.angle.sin'].to_numpy().squeeze()
        elif ('s.angle' in initial_state.columns):
            self.s.angle_cos = np.cos(initial_state['s.angle']).to_numpy().squeeze()
            self.s.angle_sin = np.sin(initial_state['s.angle']).to_numpy().squeeze()
        else:
            raise ValueError('Angle info missing')

        self.s.angleD = initial_state['s.angleD'].to_numpy().squeeze()

        self.s.position = initial_state['s.position'].to_numpy().squeeze()
        self.s.positionD = initial_state['s.positionD'].to_numpy().squeeze()

        if prediction_denorm:
            self.prediction_denorm = True
        else:
            self.prediction_denorm = False


    def predict(self, Q) -> pd.DataFrame:

        if len(Q) != self.horizon:
            raise IndexError('Number of provided control inputs does not match the horizon')
        else:
            Q_hat = np.atleast_1d(np.asarray(Q).squeeze())

        # t0 = timeit.default_timer()
        yp_hat = np.zeros(self.horizon + 1, dtype=object)

        for k in range(self.horizon):
            if k == 0:
                yp_hat[0] = deepcopy(self.s)
                s_next = deepcopy(self.s)

            t0 = timeit.default_timer()
            s_next = mpc_next_state(s_next, self.p, Q2u(Q_hat[k], self.p), dt=self.dt)
            s_next.angle_cos = np.cos(s_next.angle)
            s_next.angle_sin = np.sin(s_next.angle)
            t1 = timeit.default_timer()
            # self.eq_eval_time.append((t1 - t0) * 1.0e6)
            yp_hat[k + 1] = s_next

        all_features = []
        for k in range(len(yp_hat)):
            s = yp_hat[k]
            if k < self.horizon:
                Q = Q_hat[k]
            else:
                Q = Q_hat[k-1]
            timestep_features = [Q, s.angle_cos, s.angle_sin, s.angle, s.angleD, s.position, s.positionD]
            all_features.append(timestep_features)
        all_features = np.asarray(all_features)
        self.prediction_list.values[:, :] = all_features
        # self.prediction_list = normalize_df(self.prediction_list, self.normalization_info)

        predictions = copy.copy(self.prediction_list)

        if self.prediction_denorm:
            return predictions
        else:
            return normalize_df(predictions, self.normalization_info)

    # @tf.function
    def update_internal_state(self, Q0):
        pass
