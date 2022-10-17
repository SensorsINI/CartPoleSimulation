"""
This is a linear-quadratic regulator
It assumes that the input relation is u = Q*u_max (no fancy motor model) !
"""

import numpy as np
import scipy
from CartPole.cartpole_jacobian import cartpole_jacobian
from CartPole.cartpole_model import s0, u_max
from CartPole.state_utilities import (ANGLE_IDX, ANGLED_IDX, POSITION_IDX,
                                      POSITIOND_IDX)
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit_ASF.Cost_Functions import cost_function_base
from gym.spaces.box import Box
from others.globals_and_utils import create_rng


class controller_lqr(template_controller):
    def __init__(
        self,
        cost_function: cost_function_base,
        seed: int,
        Q: "list[float]",
        R: "list[float]",
        action_space: Box,
        actuator_noise: float=0.1,
        **kwargs
    ):
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
        super().__init__(cost_function=cost_function, seed=seed, action_space=action_space, observation_space=None, mpc_horizon=None, num_rollouts=None, predictor_specification=None, controller_logging=False)

        self.p_Q = actuator_noise
        # ref Bertsekas, p.151

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
        self.Q = np.diag(Q) # How much to punish x, v, theta, omega
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

    def step(self, s: np.ndarray, time=None):
        state = np.array(
            [[s[POSITION_IDX] - self.cost_function.environment.target_position], [s[POSITIOND_IDX]], [s[ANGLE_IDX]], [s[ANGLED_IDX]]])

        Q = np.dot(-self.K, state).item()

        Q = np.float32(Q * (1 + self.p_Q * self.rng.uniform(self.action_low, self.action_high)))
        # Q = self.rng.uniform(-1.0, 1.0)

        # Clip Q
        if Q > 1.0:
            Q = 1.0
        elif Q < -1.0:
            Q = -1.0
        else:
            pass

        return Q
