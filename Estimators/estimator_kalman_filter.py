import scipy
import numpy as np
from numpy.random import SFC64, Generator
from datetime import datetime

from Estimators.template_estimator import template_estimator
from CartPole.state_utilities import ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX
from CartPole.cartpole_model import u_max, create_cartpole_state
from CartPole.cartpole_jacobian import cartpole_jacobian

import yaml
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
Q = config["estimator"]["kalman_filter"]["Q"]
Q = np.array(Q).reshape((4,4))
R = config["estimator"]["kalman_filter"]["R"]

lookahead = 1  # 0 if there is no delay in the system, 1 to estimate state dt_control intro future
dt_control = 0.02
dt_fine = dt_control/10.0
intermediate_steps = int(dt_control/dt_fine)

class estimator_kalman_filter(template_estimator):
    def __init__(self):


        SEED = config["controller"]["lqr"]["SEED"]
        if SEED == "None":
            SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds()*1000.0)  # Fully random
        self.rng_lqr = Generator(SFC64(SEED))

        # Calculate Jacobian around equilibrium
        # Set point around which the Jacobian should be linearized
        # It can be here either pole up (all zeros) or pole down
        s = create_cartpole_state()
        s[POSITION_IDX] = 0.0
        s[POSITIOND_IDX] = 0.0
        s[ANGLE_IDX] = 0.0
        s[ANGLED_IDX] = 0.0
        u = 0.0

        jacobian = cartpole_jacobian(s, u)
        self.A = jacobian[:, :-1]
        self.B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * u_max

        # Discretised
        self.Ad = np.eye(4) + dt_control * self.A
        self.Bd = dt_control * self.B

        self.Ad_fine = np.eye(4) + dt_fine * self.A
        self.Bd_fine = dt_fine * self.B

        # Noise matrices for kalman filter
        self.Q = np.array(Q)#.reshape((4,4))  # Process  noise covariance
        self.R = np.diag(R)  # Sensors noise covariance

        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # Only position and angle is measurable

        # first, try to solve the ricatti equation
        self.X = scipy.linalg.solve_discrete_are(self.Ad.T, self.H.T, self.Q, self.R)

        # compute the Kalman gain
        Y = self.H @ self.X @ self.H.T @ + self.R  # just an auxiliary variable
        if np.array(Y).ndim == 0:
            Yi = 1.0 / Y
        else:
            Yi = np.linalg.inv(Y)

        self.K = self.X @ self.H.T @ Yi

        self.z_p = np.array([[0.0], [0.0]])  # Predicted observation
        self.z_m = np.array([[0.0], [0.0]])  # Measured observation
        self.s_p = np.array([[0.0], [0.0], [0.0], [0.0]])  # Predicted state
        self.s_m = np.array([[0.0], [0.0], [0.0], [0.0]])  # State estimation after the measurement

        # Test observability:
        O = np.concatenate((self.H, self.H@self.Ad, self.H@self.Ad@self.Ad, self.H@self.Ad@self.Ad@self.Ad), axis=0)

        rank_O = np.linalg.matrix_rank(O)
        if rank_O == 4:
            print('observability_test_passed')
        else:
            raise ValueError('Observability matrix has rank {}'.format(rank_O))

    def step(self, z_m: np.ndarray, u):

        if lookahead == 1:
            # Update state based on measurement
            self.z_p = self.H @ self.s_p
            self.s_m = self.s_p + self.K @ (z_m - self.z_p)

        # Predict ahead
        self.s_p = self.s_m.copy()
        for _ in range(intermediate_steps):
            self.s_p = self.Ad_fine @ self.s_p + self.Bd_fine * u


        if lookahead == 1:
            s_estimated = self.s_p
        elif lookahead == 0:
            self.z_p = self.H @ self.s_p
            self.s_m = self.s_p + self.K @ (z_m - self.z_p)
            s_estimated = self.s_m
        else:
            raise NotImplementedError('Currently only latency of 1*dt_control or 0 latency is implemented')


        return s_estimated



if __name__ == '__main__':

    estimator = estimator_kalman_filter()


    # Set non-zero input
    s = create_cartpole_state()
    s[POSITION_IDX] = -0.02
    s[POSITIOND_IDX] = 2.87
    s[ANGLE_IDX] = -0.32
    s[ANGLED_IDX] = 0.237


    z_m = s[[POSITION_IDX, ANGLED_IDX]]
    u = -0.24

    s_estimated = estimator.step(z_m, u)

    pass