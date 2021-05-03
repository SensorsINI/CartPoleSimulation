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

from numba import cuda, vectorize, float32
import numpy as np

from others.globals_and_utils import Timer
from copy import deepcopy

from Modeling.load_and_normalize import load_normalization_info, normalize_numpy_array
from CartPole.cartpole_model import (
    Q2u, cartpole_ode, _angleDD, _positionDD, get_A,
    TrackHalfLength, k, M, m, g, J_fric, M_fric, L, v_max, u_max
)
param_k = np.array([k], dtype=np.float32)
param_M = np.array([M], dtype=np.float32)
param_m = np.array([m], dtype=np.float32)
param_g = np.array([g], dtype=np.float32)
param_J_fric = np.array([J_fric], dtype=np.float32)
param_M_fric = np.array([M_fric], dtype=np.float32)
param_L = np.array([L], dtype=np.float32)
param_TrackHalfLength = np.array([TrackHalfLength], dtype=np.float32)
param_dt = np.array([0.02], dtype=np.float32)

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

# @vectorize(['float32(float32, float32)'], target='cuda')
@cuda.jit
def edge_bounce(position, positionD, param_TrackHalfLength):
    start = cuda.grid(1)      # 1 = one dimensional thread grid
    stride = cuda.gridsize(1)
    for i in range(start, position.shape[0], stride):
        if abs(position[i]) > param_TrackHalfLength[0]: positionD[i] = -positionD[i]

# @vectorize(['float32(float32, float32, float32)'], target='cuda')
@cuda.jit
def euler_step(state, stateD, t_step):
    start = cuda.grid(1)      # 1 = one dimensional thread grid
    stride = cuda.gridsize(1)
    for i in range(start, state.shape[0], stride):
        state[i] = state[i] + stateD[i] * t_step[0]

# @vectorize(['float32(float32)'], target='cuda')
@cuda.jit
def _cos(val, cos_val):
    start = cuda.grid(1)      # 1 = one dimensional thread grid
    stride = cuda.gridsize(1)
    for i in range(start, val.shape[0], stride):
        cos_val[i] = math.cos(val[i])

# @vectorize(['float32(float32)'], target='cuda')
@cuda.jit
def _sin(val, sin_val):
    start = cuda.grid(1)      # 1 = one dimensional thread grid
    stride = cuda.gridsize(1)
    for i in range(start, val.shape[0], stride):
        sin_val[i] = math.sin(val[i])


@cuda.jit
def _generate_immut(dt, k, M, m, g, J_fric, M_fric, L, TrackHalfLength):
    i = cuda.grid(1)
    a_in = cuda.const.array_like(param_dt)
    dt[i] = a_in[i]
    a_in = cuda.const.array_like(param_k)
    k[i] = a_in[i]
    a_in = cuda.const.array_like(param_M)
    M[i] = a_in[i]
    a_in = cuda.const.array_like(param_m)
    m[i] = a_in[i]
    a_in = cuda.const.array_like(param_g)
    g[i] = a_in[i]
    a_in = cuda.const.array_like(param_J_fric)
    J_fric[i] = a_in[i]
    a_in = cuda.const.array_like(param_M_fric)
    M_fric[i] = a_in[i]
    a_in = cuda.const.array_like(param_L)
    L[i] = a_in[i]
    a_in = cuda.const.array_like(param_TrackHalfLength)
    TrackHalfLength[i] = a_in[i]



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

        self.intermediate_steps = 2
        self.param_dt = np.ones_like(param_dt / float(self.intermediate_steps))
        self.param_k = np.ones_like(param_k)
        self.param_M = np.ones_like(param_M)
        self.param_m = np.ones_like(param_m)
        self.param_g = np.ones_like(param_g)
        self.param_J_fric = np.ones_like(param_J_fric)
        self.param_M_fric = np.ones_like(param_M_fric)
        self.param_L = np.ones_like(param_L)
        self.param_TrackHalfLength = np.ones_like(param_TrackHalfLength)

        _generate_immut[1,1](self.param_dt, self.param_k, self.param_M, self.param_m, self.param_g, self.param_J_fric, self.param_M_fric, self.param_L, self.param_TrackHalfLength)

        self.prediction_features_names = cartpole_state_indices_to_varnames(range(len(self.s)))

        self.prediction_denorm = False


    def setup(self, initial_state: np.ndarray, prediction_denorm=False):

        # The initial state is provided with not valid second derivatives
        # Batch_size > 1 allows to feed several states at once and obtain predictions parallely

        initial_state = np.transpose(initial_state)
        # Shape of state: (state variables x batch size)
        if initial_state.ndim == 1: initial_state = initial_state[:, np.newaxis]
        self.s = cuda.to_device(np.ascontiguousarray(initial_state))
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
        
        # self.angleDD_out = cuda.device_array(shape=(self.batch_size,), dtype=np.float32)
        # self.positionDD_out = cuda.device_array(shape=(self.batch_size,), dtype=np.float32)

        self.prediction_denorm = prediction_denorm

        self.A = cuda.device_array(shape=(self.batch_size), dtype=np.float32)
        self.u = cuda.device_array(shape=(self.horizon, self.batch_size), dtype=np.float32)
        self.output = np.zeros((self.horizon+1, len(self.prediction_features_names)+1, self.batch_size), dtype=np.float32)

    def next_state(self, k):
        """Wrapper for CartPole ODE. Given a current state (without second derivatives), returns a state after time dt
        """
        # # Calculates CURRENT second derivatives
        # s[cartpole_state_varnames_to_indices(['angleDD', 'positionDD'])] = cartpole_ode(s, u)
        for i in range(self.intermediate_steps):
            # Calculate NEXT state:
            with Timer("8"):
                euler_step[blocks_per_grid, threads_per_block](self.angle, self.angleD, self.param_dt)
            with Timer("9"):
                euler_step[blocks_per_grid, threads_per_block](self.angleD, self.angleDD, self.param_dt)
            with Timer("10"):
                euler_step[blocks_per_grid, threads_per_block](self.position, self.positionD, self.param_dt)
            with Timer("11"):
                euler_step[blocks_per_grid, threads_per_block](self.positionD, self.positionDD, self.param_dt)
            
            cuda.synchronize()
            
            # Simulate bouncing off edges (does not consider cart dimensions)
            with Timer("12"):
                edge_bounce[blocks_per_grid, threads_per_block](self.position, self.positionD, self.param_TrackHalfLength)
            
            cuda.synchronize()

            # Calculates second derivatives of NEXT state
            with Timer("13"):
                get_A[blocks_per_grid, threads_per_block](self.angle_cos, self.A, self.param_k, self.param_M, self.param_m)
            cuda.synchronize()

            with Timer("14"):
                _angleDD[blocks_per_grid, threads_per_block](self.angleD, self.angleDD, self.positionD, self.angle_cos, self.angle_sin, self.A, self.u[k, :], self.param_m, self.param_M, self.param_g, self.param_M_fric, self.param_J_fric, self.param_L)
                _positionDD[blocks_per_grid, threads_per_block](self.angleD, self.positionD, self.positionDD, self.angle_cos, self.angle_sin, self.A, self.u[k, :], self.param_m, self.param_k, self.param_g, self.param_M_fric, self.param_J_fric, self.param_L)

        with Timer("15"):
            _cos[blocks_per_grid, threads_per_block](self.angle, self.angle_cos)
            _sin[blocks_per_grid, threads_per_block](self.angle, self.angle_sin)
        cuda.synchronize()


    def predict(self, Q: np.ndarray) -> np.ndarray:
        
        with Timer("1"):
            # Shape of Q: (batch size x horizon length)
            if np.size(Q, -1) != self.horizon:
                raise IndexError('Number of provided control inputs does not match the horizon')
            else:
                Q_hat = np.atleast_1d(np.asarray(Q).squeeze())
        
        with Timer("2"):
            # shape(u) = horizon_steps x batch_size
            self.u = cuda.to_device(Q2u(np.ascontiguousarray(Q_hat.T)))

        with Timer("3"):
            # Calculate second derivatives of initial state
            get_A[blocks_per_grid, threads_per_block](self.angle_cos, self.A, self.param_k, self.param_M, self.param_m)
        cuda.synchronize()
        with Timer("4"):
            _angleDD[blocks_per_grid, threads_per_block](self.angleD, self.angleDD, self.positionD, self.angle_cos, self.angle_sin, self.A, self.u[0, :], self.param_m, self.param_M, self.param_g, self.param_M_fric, self.param_J_fric, self.param_L)
            _positionDD[blocks_per_grid, threads_per_block](self.angleD, self.positionD, self.positionDD, self.angle_cos, self.angle_sin, self.A, self.u[0, :], self.param_m, self.param_k, self.param_g, self.param_M_fric, self.param_J_fric, self.param_L)
        with Timer("5"):
            self.output[0, :-1, :] = np.stack([self.angle, self.angleD, self.angleDD, self.position, self.positionD, self.positionDD, self.angle_cos, self.angle_sin])

        for k in range(self.horizon):
            with Timer("6"):
            # Inplace update of state
                self.next_state(k)
            with Timer("6.2"):
                self.output[k+1, :-1, :] = np.stack([self.angle, self.angleD, self.angleDD, self.position, self.positionD, self.positionDD, self.angle_cos, self.angle_sin])

        # self.output = self.output.copy_to_host()
        with Timer("7"):
            self.output = np.squeeze(self.output)
            if self.prediction_denorm:
                return np.transpose(self.output[:, :-1, :], axes=(2,0,1))
            else:
                self.output[:-1, -1, :] = np.transpose(Q_hat)
                columns = self.prediction_features_names + ['Q']
                return normalize_numpy_array(np.transpose(self.output, axes=(2,0,1)), columns, np.squeeze(self.normalization_info)[:, :-1])

    # @tf.function
    def update_internal_state(self, Q0):
        pass
