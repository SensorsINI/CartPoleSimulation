"""
This is a linear-quadratic regulator
It assumes that the input relation is u = Q*u_max (no fancy motor model) !
"""

import scipy
import numpy as np
from numpy.random import SFC64, Generator
from datetime import datetime

from Controllers.template_controller import template_controller
from CartPole.state_utilities import ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX
from CartPole.cartpole_model import u_max, s0
from CartPole.cartpole_jacobian import cartpole_jacobian

import yaml
config = yaml.load(open("CartPoleSimulation/config.yml", "r"), Loader=yaml.FullLoader)
Q = np.diag(config["controller"]["lqr"]["Q"])
R = config["controller"]["lqr"]["R"]

p_Q = config["controller"]["lqr"]["control_noise"]

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

        SEED = config["controller"]["lqr"]["SEED"]
        if SEED == "None":
            SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds()*1000.0)  # Fully random
        self.rng_lqr = Generator(SFC64(SEED))
        self.rng_lqr = Generator(SFC64(SEED*2))

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
        self.Q = Q  # How much to punish x, v, theta, omega
        self.R = R  # How much to punish Q

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

    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):

        state = np.array(
            [[s[POSITION_IDX] - target_position], [s[POSITIOND_IDX]], [s[ANGLE_IDX]], [s[ANGLED_IDX]]])

        Q = np.dot(-self.K, state).item()

        Q = np.float32(Q * (1 + p_Q * self.rng_lqr.uniform(-1.0, 1.0)))
        # Q = self.rng_lqr.uniform(-1.0, 1.0)

        # Clip Q
        if Q > 1.0:
            Q = 1.0
        elif Q < -1.0:
            Q = -1.0
        else:
            pass

        return Q
