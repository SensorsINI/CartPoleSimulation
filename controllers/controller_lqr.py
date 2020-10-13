
import scipy
import numpy as np


class controller_lqr:
    def __init__(self, A, B, Q, R):
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

        # first, try to solve the ricatti equation
        X = scipy.linalg.solve_continuous_are(A, B, Q, R)

        # compute the LQR gain
        if np.array(R).ndim == 0:
            Ri = 1.0 / R
        else:
            Ri = np.linalg.inv(R)

        K = np.dot(Ri, (np.dot(B.T, X)))

        eigVals = np.linalg.eigvals(A - np.dot(B, K))

        self.K = K
        self.X = X
        self.eigVals = eigVals

    def step(self, state):
        Q = np.asscalar(np.dot(-self.K, state))
        return Q
