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

import numpy as np

from SI_Toolkit.load_and_normalize import load_normalization_info, normalize_numpy_array
from CartPole.cartpole_model import (
    L, Q2u, _cartpole_ode_numba, TrackHalfLength, next_state_numba
)

from CartPole.state_utilities import (
    ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX,
    STATE_VARIABLES
)
import yaml, os
config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'), Loader=yaml.FullLoader)

PATH_TO_NORMALIZATION_INFO = config['paths']['PATH_TO_EXPERIMENT_RECORDINGS'] + config['paths']['path_to_experiment'] + "NormalizationInfo/"
PATH_TO_NORMALIZATION_INFO += os.listdir(PATH_TO_NORMALIZATION_INFO)[0]


class predictor_ideal:
    def __init__(self, horizon, dt, intermediate_steps=1):
        try:
            self.normalization_info = load_normalization_info(PATH_TO_NORMALIZATION_INFO)
        except FileNotFoundError:
            print('Normalization info not provided.')
        
        self.batch_size = 1

        self.target_position = 0.0
        self.target_position_normed = 0.0

        self.horizon = horizon

        self.intermediate_steps = intermediate_steps
        self.t_step = dt / float(self.intermediate_steps)

        self.prediction_features_names = STATE_VARIABLES.tolist()

        self.prediction_denorm = False
        self.batch_mode = False

        self.output = None

    def setup(self, initial_state: np.ndarray, prediction_denorm=False):

        # The initial state is provided with not valid second derivatives
        # Batch_size > 1 allows to feed several states at once and obtain predictions parallely
        # Shape of state: (batch size x state variables)

        self.batch_size = np.size(initial_state, 0) if initial_state.ndim > 1 else 1
        self.batch_mode = not (self.batch_size == 1)

        if not self.batch_mode: initial_state = np.expand_dims(initial_state, 0)
        self.angleDD = self.positionDD = 0

        self.angle, self.angleD, self.position, self.positionD, self.angle_cos, self.angle_sin = (
            initial_state[:, ANGLE_IDX],
            initial_state[:, ANGLED_IDX],
            initial_state[:, POSITION_IDX],
            initial_state[:, POSITIOND_IDX],
            initial_state[:, ANGLE_COS_IDX],
            initial_state[:, ANGLE_SIN_IDX],
        )

        self.prediction_denorm = prediction_denorm

        self.u = np.zeros(shape=(self.batch_size, self.horizon), dtype=np.float32)
        self.output = np.zeros((self.batch_size, self.horizon+1, len(self.prediction_features_names)+1), dtype=np.float32)
    
    def next_state(self, k):
        """Wrapper for CartPole ODE. Given a current state (without second derivatives), returns a state after time dt
        """
        (
            self.angle, self.angleD, self.angleDD, self.position, self.positionD, self.positionDD, self.angle_cos, self.angle_sin
        ) = next_state_numba(
            angle=self.angle,
            angleD=self.angleD,
            angleDD=self.angleDD,
            angle_cos=self.angle_cos,
            angle_sin=self.angle_sin,
            position=self.position,
            positionD=self.positionD,
            positionDD=self.positionDD,
            u=self.u[:, k],
            t_step=self.t_step,
            intermediate_steps=self.intermediate_steps
        )

    def predict(self, Q: np.ndarray) -> np.ndarray:

        # Shape of Q: (batch size x horizon length)
        if np.size(Q, -1) != self.horizon:
            raise IndexError('Number of provided control inputs does not match the horizon')
        else:
            Q_hat = np.atleast_1d(np.asarray(Q).squeeze())

        # shape(u) = horizon_steps x batch_size
        self.u = Q2u(Q_hat)
        if self.u.ndim == 1: self.u = np.expand_dims(self.u, 0)

        # Calculate second derivatives of initial state
        self.angleDD, self.positionDD, self.angle_cos, self.angle_sin = _cartpole_ode_numba(
            np.cos(-self.angle),
            np.sin(-self.angle),
            self.angleD,
            self.positionD,
            self.u[:, 0]
        )
        self.write_outputs(0)

        for k in range(self.horizon):
            # State update
            self.next_state(k)
            self.write_outputs(k+1)

        # out_array = np.transpose(self.output, axes=(2,0,1))
        # if not self.batch_mode: self.output = np.squeeze(self.output)

        if self.prediction_denorm:
            return self.output[:, :, :-1] if self.batch_mode else np.squeeze(self.output[:, :, :-1])
        else:
            self.output[:, :-1, -1] = Q_hat
            columns = self.prediction_features_names + ['Q']
            out_array = self.output if self.batch_mode else np.squeeze(self.output)
            return normalize_numpy_array(out_array, columns, np.squeeze(self.normalization_info)[:, :-1])

    def write_outputs(self, iteration):
        self.output[:, iteration, ANGLE_IDX] = self.angle
        self.output[:, iteration, ANGLED_IDX] = self.angleD
        self.output[:, iteration, POSITION_IDX] = self.position
        self.output[:, iteration, POSITIOND_IDX] = self.positionD
        self.output[:, iteration, ANGLE_COS_IDX] = self.angle_cos
        self.output[:, iteration, ANGLE_SIN_IDX] = self.angle_sin

    def update_internal_state(self, Q0):
        pass
