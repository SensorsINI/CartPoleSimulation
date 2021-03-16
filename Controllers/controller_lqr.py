"""
This is a linear-quadratic regulator
It assumes that the input relation is u = Q*p.u_max (no fancy motor model) !
"""

import scipy
import numpy as np

from Controllers.template_controller import template_controller
from CartPole._CartPole_mathematical_helpers import create_cartpole_state, cartpole_state_varname_to_index
from CartPole.cartpole_model import cartpole_jacobian, P_GLOBALS, s0

class controller_lqr(template_controller):
    def __init__(self):
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

        # Calculate Jacobian around equilibrium
        # Set point around which the Jacobian should be linearized
        # It can be here either pole up (all zeros) or pole down
        s = s0
        s[cartpole_state_varname_to_index('position')] = 0.0
        s[cartpole_state_varname_to_index('positionD')] = 0.0
        s[cartpole_state_varname_to_index('angle')] = 0.0
        s[cartpole_state_varname_to_index('angleD')] = 0.0
        u = 0.0

        jacobian = cartpole_jacobian(P_GLOBALS, s, u)

        A = jacobian[:, :-1]
        B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * P_GLOBALS.u_max

        # Cost matrices for LQR controller
        self.Q = np.diag([10.0, 1.0, 1.0, 1.0])  # How much to punish x, v, theta, omega
        self.R = 1.0e9  # How much to punish Q

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
        self.X = X
        self.eigVals = eigVals

    def step(self, state, PositionTarget, time=None):

        s = state

        state = np.array(
            [[s[cartpole_state_varname_to_index('position')] - PositionTarget], [s[cartpole_state_varname_to_index('positionD')]], [s[cartpole_state_varname_to_index('angle')]], [s[cartpole_state_varname_to_index('angleD')]]])

        Q = np.asscalar(np.dot(-self.K, state))
        return Q
