import numpy as np
import scipy

from CartPole.cartpole_jacobian import cartpole_jacobian
from CartPole.cartpole_parameters import u_max
from CartPole.state_utilities import create_cartpole_state
from CartPole.state_utilities import (
    ANGLE_IDX,
    ANGLED_IDX,
    POSITION_IDX,
    POSITIOND_IDX,
)


s0 = create_cartpole_state()


class LQRSantiago:
    def __init__(self, K, X, eigVals, Q, R):
        self.K = K
        self.X = X
        self.eigVals = eigVals
        self.Q = Q
        self.R = R

    @classmethod
    def from_config(cls, config_controller):
        # Calculate the Jacobian around the upright equilibrium.
        s = s0
        s[POSITION_IDX] = 0.0
        s[POSITIOND_IDX] = 0.0
        s[ANGLE_IDX] = 0.0
        s[ANGLED_IDX] = 0.0
        u = 0.0

        jacobian = cartpole_jacobian(s, u)
        A = jacobian[:, :-1]
        B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * u_max

        Q = np.diag(config_controller["Q"])
        R = config_controller["R"]
        print("Q:", Q, " - R:", R)

        X = scipy.linalg.solve_continuous_are(A, B, Q, R)

        if np.array(R).ndim == 0:
            Ri = 1.0 / R
        else:
            Ri = np.linalg.inv(R)

        K = np.dot(Ri, (np.dot(B.T, X)))
        eigVals = np.linalg.eigvals(A - np.dot(B, K))

        print(" K:", K)
        return cls(K=K, X=X, eigVals=eigVals, Q=Q, R=R)

    def state_vector(self, s, target_position):
        return np.array(
            [
                [s[POSITION_IDX] - target_position],
                [s[POSITIOND_IDX]],
                [s[ANGLE_IDX]],
                [s[ANGLED_IDX]],
            ]
        )

    def step(self, s, target_position):
        state = self.state_vector(s, target_position)
        Q = np.dot(-self.K, state).item()
        return np.clip(Q, -1.0, 1.0, dtype=np.float32)
