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


import math

from numba import jit
import numpy as np

from others.globals_and_utils import Timer
from copy import deepcopy

from Modeling.load_and_normalize import load_normalization_info, normalize_numpy_array
from CartPole.cartpole_model import (
    Q2u, cartpole_ode, _angleDD, _positionDD, get_A,
    TrackHalfLength, k, M, m, g, J_fric, M_fric, L, v_max, u_max
)

from CartPole.state_utilities import (
    create_cartpole_state,
    cartpole_state_varname_to_index, cartpole_state_varnames_to_indices,
    cartpole_state_indices_to_varnames,
    ANGLE_IDX, ANGLED_IDX, ANGLEDD_IDX, POSITION_IDX, POSITIOND_IDX, POSITIONDD_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX,
    STATE_VARIABLES
)

PATH_TO_NORMALIZATION_INFO = './Modeling/NormalizationInfo/' + 'NI_2021-04-22_14-34-14.csv'
threads_per_block = 16
blocks_per_grid = math.ceil(2000/16)


@jit(nopython=True, cache=True, fastmath=True)
def edge_bounce(position, positionD):
    for i in range(position.shape[0]):
        if abs(position[i]) > TrackHalfLength: positionD[i] = -positionD[i]
    return positionD


@jit(nopython=True, cache=True, fastmath=True)
def euler_step(state, stateD, t_step):
    state += stateD * t_step
    return state


@jit(nopython=True, cache=True, fastmath=True)
def next_state_numba(angle, angleD, angleDD, angle_cos, angle_sin, position, positionD, positionDD, u, t_step, intermediate_steps):
    for _ in range(intermediate_steps):
        # Calculate NEXT state:
        # with Timer("8"):
        angle = euler_step(angle, angleD, t_step)
        # with Timer("9"):
        angleD = euler_step(angleD, angleDD, t_step)
        # with Timer("10"):
        position = euler_step(position, positionD, t_step)
        # with Timer("11"):
        positionD = euler_step(positionD, positionDD, t_step)
        
        # Simulate bouncing off edges (does not consider cart dimensions)
        # with Timer("12"):
        positionD = edge_bounce(position, positionD)

        # Calculates second derivatives of NEXT state
        # with Timer("13"):
        A = get_A(angle_cos)

        # with Timer("14"):
        angleDD = _angleDD(angleD, positionD, angle_cos, angle_sin, A, u)
        positionDD = _positionDD(angleD, positionD, angle_cos, angle_sin, A, u)

    # with Timer("15"):
    angle_cos = np.cos(angle)
    angle_sin = np.sin(angle)

    return angle, angleD, angleDD, position, positionD, positionDD, angle_cos, angle_sin

class predictor_ideal:
    def __init__(self, horizon, dt):
        try:
            self.normalization_info = load_normalization_info(PATH_TO_NORMALIZATION_INFO)
        except FileNotFoundError:
            print('Normalization info not provided.')
        # State of the cart
        self.s = create_cartpole_state()  # s like state
        self.batch_size = 1

        self.target_position = 0.0
        self.target_position_normed = 0.0

        self.horizon = horizon

        self.intermediate_steps = 1
        self.t_step = dt / float(self.intermediate_steps)

        self.prediction_features_names = cartpole_state_indices_to_varnames(range(len(self.s)))

        self.prediction_denorm = False


    def setup(self, initial_state: np.ndarray, prediction_denorm=False):

        # The initial state is provided with not valid second derivatives
        # Batch_size > 1 allows to feed several states at once and obtain predictions parallely

        initial_state = np.transpose(initial_state)
        # Shape of state: (state variables x batch size)
        if initial_state.ndim == 1: initial_state = initial_state[:, np.newaxis]
        self.s = initial_state
        self.batch_size = np.size(self.s, 1) if self.s.ndim > 1 else 1

        self.angle, self.angleD, self.angleDD, self.position, self.positionD, self.positionDD, self.angle_cos, self.angle_sin = (
            self.s[ANGLE_IDX, :],
            self.s[ANGLED_IDX, :],
            self.s[ANGLEDD_IDX, :],
            self.s[POSITION_IDX, :],
            self.s[POSITIOND_IDX, :],
            self.s[POSITIONDD_IDX, :],
            self.s[ANGLE_COS_IDX, :],
            self.s[ANGLE_SIN_IDX, :],
        )

        self.prediction_denorm = prediction_denorm

        self.A = np.zeros(shape=(self.batch_size), dtype=np.float32)
        self.u = np.zeros(shape=(self.horizon, self.batch_size), dtype=np.float32)
        self.output = np.zeros((self.horizon+1, len(self.prediction_features_names)+1, self.batch_size), dtype=np.float32)

    def next_state(self, k):
        """Wrapper for CartPole ODE. Given a current state (without second derivatives), returns a state after time dt
        """
        # # Calculates CURRENT second derivatives
        # s[cartpole_state_varnames_to_indices(['angleDD', 'positionDD'])] = cartpole_ode(s, u)
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
            u=self.u[k, :],
            t_step=self.t_step,
            intermediate_steps=self.intermediate_steps
        )
        


    def predict(self, Q: np.ndarray) -> np.ndarray:
        
        # with Timer("1"):
        # Shape of Q: (batch size x horizon length)
        if np.size(Q, -1) != self.horizon:
            raise IndexError('Number of provided control inputs does not match the horizon')
        else:
            Q_hat = np.atleast_1d(np.asarray(Q).squeeze())
        
        # with Timer("2"):
        # shape(u) = horizon_steps x batch_size
        self.u = Q2u(Q_hat.T)

        # with Timer("3"):
        # Calculate second derivatives of initial state
        self.A = get_A(self.angle_cos)
        # with Timer("4"):
        self.angleDD = _angleDD(self.angleD, self.positionD, self.angle_cos, self.angle_sin, self.A, self.u[0, :])
        self.positionDD = _positionDD(self.angleD, self.positionD, self.angle_cos, self.angle_sin, self.A, self.u[0, :])
        # with Timer("5"):
        self.write_outputs(0)

        for k in range(self.horizon):
            # with Timer("6"):
            # Inplace update of state
            self.next_state(k)
            # with Timer("6.2"):
            self.write_outputs(k+1)

        # with Timer("7"):
        self.output = np.squeeze(self.output)
        if self.prediction_denorm:
            return np.transpose(self.output[:, :-1, :], axes=(2,0,1))
        else:
            self.output[:-1, -1, :] = np.transpose(Q_hat)
            columns = self.prediction_features_names + ['Q']
            return normalize_numpy_array(np.transpose(self.output, axes=(2,0,1)), columns, np.squeeze(self.normalization_info)[:, :-1])

    def write_outputs(self, iteration):
        self.output[iteration, ANGLE_IDX, :] = self.angle
        self.output[iteration, ANGLED_IDX, :] = self.angleD
        self.output[iteration, ANGLEDD_IDX, :] = self.angleDD
        self.output[iteration, POSITION_IDX, :] = self.position
        self.output[iteration, POSITIOND_IDX, :] = self.positionD
        self.output[iteration, POSITIONDD_IDX, :] = self.positionDD
        self.output[iteration, ANGLE_COS_IDX, :] = self.angle_cos
        self.output[iteration, ANGLE_SIN_IDX, :] = self.angle_sin

    # @tf.function
    def update_internal_state(self, Q0):
        pass
