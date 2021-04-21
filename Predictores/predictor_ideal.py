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


from Modeling.load_and_normalize import load_normalization_info, normalize_numpy_array
from CartPole.cartpole_model import Q2u, TrackHalfLength
from CartPole.state_utilities import create_cartpole_state, \
    cartpole_state_varname_to_index, cartpole_state_varnames_to_indices, \
    cartpole_state_indices_to_varnames

import numpy as np

from CartPole.cartpole_model import cartpole_ode

from copy import deepcopy

PATH_TO_NORMALIZATION_INFO = './Modeling/NormalizationInfo/' + '2500.csv'

def next_state(s, u, dt, intermediate_steps=2):
    """Wrapper for CartPole ODE. Given a current state (without second derivatives), returns a state after time dt
    """

    s_next = deepcopy(s)

    # # Calculates CURRENT second derivatives
    # s_next[cartpole_state_varnames_to_indices(['angleDD', 'positionDD'])] = cartpole_ode(s, u)
    for i in range(intermediate_steps):
        # Calculate NEXT state:
        s_next[..., cartpole_state_varname_to_index('position')] = \
            s_next[..., cartpole_state_varname_to_index('position')] + s_next[..., cartpole_state_varname_to_index('positionD')] * (dt/float(intermediate_steps))
        s_next[..., cartpole_state_varname_to_index('positionD')] = \
            s_next[..., cartpole_state_varname_to_index('positionD')] + s_next[..., cartpole_state_varname_to_index('positionDD')] * (dt/float(intermediate_steps))
        
        # Simulate bouncing off edges (does not consider cart dimensions)
        s_next[..., abs(s_next[..., cartpole_state_varname_to_index('position')]) > TrackHalfLength, cartpole_state_varname_to_index('positionD')] \
            = -s_next[..., abs(s_next[..., cartpole_state_varname_to_index('position')]) > TrackHalfLength, cartpole_state_varname_to_index('positionD')]

        s_next[..., cartpole_state_varname_to_index('angle')] = \
            s_next[..., cartpole_state_varname_to_index('angle')] + s_next[..., cartpole_state_varname_to_index('angleD')] * (dt/float(intermediate_steps))
        s_next[..., cartpole_state_varname_to_index('angleD')] = \
            s_next[..., cartpole_state_varname_to_index('angleD')] + s_next[..., cartpole_state_varname_to_index('angleDD')] * (dt/float(intermediate_steps))

        # Calculates second derivatives of NEXT state
        s_next[..., cartpole_state_varnames_to_indices(['angleDD', 'positionDD'])] = np.vstack(cartpole_ode(s_next, u)).T

        s_next[..., cartpole_state_varname_to_index('angle_cos')] = np.cos(s_next[..., cartpole_state_varname_to_index('angle')])
        s_next[..., cartpole_state_varname_to_index('angle_sin')] = np.sin(s_next[..., cartpole_state_varname_to_index('angle')])


    return s_next



class predictor_ideal:
    def __init__(self, horizon, dt):

        self.normalization_info = None

        # State of the cart
        self.s = create_cartpole_state()  # s like state
        self.batch_size = 1

        self.target_position = 0.0
        self.target_position_normed = 0.0

        self.horizon = horizon

        self.dt = dt

        self.prediction_features_names = cartpole_state_indices_to_varnames(range(len(self.s)))

        self.prediction_denorm = False


    def setup(self, initial_state: np.ndarray, prediction_denorm=False):

        # The initial state is provided with not valid second derivatives
        # Batch_size > 1 allows to feed several states at once and obtain predictions parallely

        # Shape of state: (batch size x state variables)
        self.s = initial_state
        self.batch_size = np.size(self.s, 0) if self.s.ndim > 1 else 1

        self.prediction_denorm = prediction_denorm

        self.output = np.zeros((self.batch_size, self.horizon+1, len(self.prediction_features_names)+1), dtype=np.float32)


    def predict(self, Q: np.ndarray) -> np.ndarray:

        # Shape of Q: (batch size x horizon length)
        if np.size(Q, -1) != self.horizon:
            raise IndexError('Number of provided control inputs does not match the horizon')
        else:
            Q_hat = np.atleast_1d(np.asarray(Q).squeeze())

        self.output[..., :-1, -1] = Q_hat

        s_next = self.s
        # Calculate second derivatives of initial state
        s_next[..., cartpole_state_varnames_to_indices(['angleDD', 'positionDD'])] = np.vstack(cartpole_ode(s_next, Q2u(Q_hat[..., 0]))).T
        self.output[..., 0, :-1] = s_next

        for k in range(self.horizon):
            s_next = next_state(s_next, Q2u(Q_hat[..., k]), dt=self.dt, intermediate_steps=2)
            self.output[..., k+1, :-1] = s_next

        self.output = np.squeeze(self.output)
        if self.prediction_denorm:
            return self.output[..., :-1]
        else:
            columns = self.prediction_features_names + ['Q']
            return normalize_numpy_array(self.output, columns, self.normalization_info)[:, :-1]

    # @tf.function
    def update_internal_state(self, Q0):
        pass
