"""
This is a linear-quadratic regulator
It assumes that the input relation is u = Q*u_max (no fancy motor model) !
"""

from SI_Toolkit.computation_library import NumpyLibrary, TensorType
import numpy as np
import scipy
import yaml
import os

from CartPole.cartpole_jacobian import cartpole_jacobian

from CartPole.cartpole_parameters import u_max
from CartPole.state_utilities import create_cartpole_state
from CartPole.state_utilities import (ANGLE_IDX, ANGLED_IDX, POSITION_IDX,
                                      POSITIOND_IDX)
from Control_Toolkit.Controllers import template_controller
from others.globals_and_utils import create_rng, load_config

s0 = create_cartpole_state()
config = load_config("cartpole_physical_parameters.yml")


class controller_secloc_lqr(template_controller):
    _computation_library = NumpyLibrary()
    
    def configure(self):
        # From https://github.com/markwmuller/controlpy/blob/master/controlpy/synthesis.py#L8
        """Solve the continuous time LQR controller for a continuous time system.

        A and B are system matrices, describing the systems dynamics:
         dx/dt = A x + B u

        The controller minimizes the infinite horizon quadratic cost function:
         cost = integral (x.T*Q*x + u.T*R*u) dt

        where Q is a positive semidefinite matrix, and R is positive definite matrix.

        Returns K, X, eigVals:
        Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
        The optimal input is then computed as:
         input: u = -K*x
        """
        # ref Bertsekas, p.151

        seed = self.config_controller["seed"]
        self.rng = create_rng(self.__class__.__name__, seed if seed==None else seed*2)

        # Calculate Jacobian around equilibrium
        # Set point around which the Jacobian should be linearized
        # It can be here either pole up (all zeros) or pole down
        s = s0
        s[POSITION_IDX] = 0.0
        s[POSITIOND_IDX] = 0.0
        s[ANGLE_IDX] = 0.0
        s[ANGLED_IDX] = 0.0
        u = 0.0

        jacobian = cartpole_jacobian(s, u)
        A = jacobian[:, :-1]
        B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * u_max

        # Cost matrices for LQR controller
        self.Q = np.diag(self.config_controller["Q"]) # How much to punish x, v, theta, omega
        self.R = self.config_controller["R"]  # How much to punish Q
        print('Q:',self.Q,' - R:',self.R)
        # first, try to solve the ricatti equation
        X = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)

        # compute the LQR gain
        if np.array(self.R).ndim == 0:
            Ri = 1.0 / self.R
        else:
            Ri = np.linalg.inv(self.R)

        K = np.dot(Ri, (np.dot(B.T, X)))

        eigVals = np.linalg.eigvals(A - np.dot(B, K))

        self.K = K
        print(' K:', self.K)
        self.X = X
        self.eigVals = eigVals
        
        self.log_base = self.config_controller["log_base"]
        self.dead_ang = self.config_controller["dead_ang"]
        self.dead_pos = self.config_controller["dead_pos"]
        self.ang_last_shift = 0.0001
        self.pos_last_shift = 0.0001
        self.last_Q = 0
        self.time_last = None

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        if self.time_last is None:
            time_difference = self.config_controller["ref_period"]
        else:
            time_difference = time - self.time_last
        self.update_attributes(updated_attributes)
        
        state = np.array(
            [[s[POSITION_IDX] - self.variable_parameters.target_position], [s[POSITIOND_IDX]], [s[ANGLE_IDX]], [s[ANGLED_IDX]]])
        if (time_difference+self.config_controller["ref_period"]/20 >= self.config_controller["ref_period"]):
            spike = 0
            ang_shift_sign = 1
            ang_shift = s[ANGLE_IDX]
            pos_shift_sign = 1
            pos_shift = s[POSITION_IDX] - self.variable_parameters.target_position
            if ang_shift < 0:
                ang_shift_sign = -1
                ang_shift = -ang_shift
            if pos_shift < 0:
                pos_shift_sign = -1
                pos_shift = -pos_shift
            if (ang_shift > self.dead_ang) and (self.ang_last_shift != 0):		# ang_dead_band cant be 0.
                ang_ratio_inc = ang_shift/self.ang_last_shift
                ang_ratio_dec = 1.0/ang_ratio_inc
                if (ang_ratio_inc >= self.log_base) or (ang_ratio_dec >= self.log_base):
                    self.ang_last_shift = ang_shift
                    spike = 1
            elif (pos_shift > self.dead_pos) and (self.pos_last_shift != 0):
                pos_ratio_inc = pos_shift/self.pos_last_shift
                pos_ratio_dec = 1.0/pos_ratio_inc
                if (pos_ratio_inc >= self.log_base) or (pos_ratio_dec >= self.log_base):
                    self.pos_last_shift = pos_shift
                    spike = 1
            if (spike):
                self.time_last = time
                Q = np.dot(-self.K, state).item()
                Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)
                self.last_Q = Q
                return Q
        return self.last_Q
