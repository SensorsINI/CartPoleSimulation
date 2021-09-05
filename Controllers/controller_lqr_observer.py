"""
This is a linear-quadratic regulator
It assumes that the input relation is u = Q*u_max (no fancy motor model) !
"""

import scipy
import numpy as np
from scipy import signal

from datetime import datetime
from numpy.random import SFC64, Generator

from Controllers.template_controller import template_controller
from CartPole.state_utilities import create_cartpole_state, cartpole_state_varname_to_index
from CartPole.cartpole_model import u_max, s0
from CartPole.cartpole_jacobian import cartpole_jacobian

import yaml
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
Q = np.diag(config["controller"]["lqr"]["Q"])
R = config["controller"]["lqr"]["R"]

class controller_lqr_observer(template_controller):
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

        self.controller_data_for_csv = {
                                        'angle_estimate': [0.0],
                                        'angleD_estimate': [0.0],
                                        'position_estimate': [0.0],
                                        'positionD_estimate': [0.0],
                                        'angle_measurement': [0.0],
                                        'position_measurement': [0.0],
                                        }

        SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)
        self.rng_lqr_observer = Generator(SFC64(SEED))

        # Calculate Jacobian around equilibrium
        # Set point around which the Jacobian should be linearized
        # It can be here either pole up (all zeros) or pole down
        s = s0
        s[cartpole_state_varname_to_index('position')] = 0.0
        s[cartpole_state_varname_to_index('positionD')] = 0.0
        s[cartpole_state_varname_to_index('angle')] = 0.0
        s[cartpole_state_varname_to_index('angleD')] = 0.0
        u = 0.0

        jacobian = cartpole_jacobian(s, u)
        self.A = jacobian[:, :-1]
        self.B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * u_max
        self.C = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0]])

        # Cost matrices for LQR controller
        self.Q = Q  # How much to punish x, v, theta, omega
        self.R = R  # How much to punish Q

        # first, try to solve the ricatti equation
        X = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

        # compute the LQR gain
        if np.array(self.R).ndim == 0:
            Ri = 1.0 / self.R
        else:
            Ri = np.linalg.inv(self.R)

        self.K = np.dot(Ri, (np.dot(self.B.T, X)))

        poles_system = np.linalg.eigvals(self.A - np.dot(self.B, self.K))
        slowest_pole_ct = poles_system[np.argmax(poles_system)]
        pole1_ct = slowest_pole_ct * 10
        pole2_ct = np.conj(pole1_ct)
        poles_obs_ct = np.array([pole1_ct, pole1_ct-1, pole2_ct, pole2_ct-1])
        print('poles obs ct', poles_obs_ct)

        # Discretize continuous time system
        Ts = 0.02 # sampling frequency in sec
        # print(np.exp(Ts*poles_system))
        poles_obs = np.exp(Ts*poles_obs_ct)
        print('poles obs dt',poles_obs)
        Ad = np.eye(4) + Ts * self.A
        Bd = Ts * self.B
        Cd = self.C
        # poles_system = np.linalg.eigvals(Ad - np.dot(Bd, self.K))
        # print(poles_system)

        # Luenberger observer gain
        obs = signal.place_poles(Ad.T, Cd.T, poles_obs.T)
        self.Kf = obs.gain_matrix.T


        self.state_estimate = np.array(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0]
            ]
        )

        self.next_state_estimate = self.state_estimate

    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):

        # True state - you are not allowed to use it!
        state_true = np.array(
            [
                [s[cartpole_state_varname_to_index('position')] - target_position],
                [s[cartpole_state_varname_to_index('positionD')]],
                [s[cartpole_state_varname_to_index('angle')]],
                [s[cartpole_state_varname_to_index('angleD')]]
            ]
        )

        # Output_measurement is what you receive at every step
        output_measurement = np.array(
            [
                [(s[cartpole_state_varname_to_index('position')] - target_position)*(1+0.05*self.rng_lqr_observer.standard_normal())],
                [s[cartpole_state_varname_to_index('angle')]*(1+0.05*self.rng_lqr_observer.standard_normal())],
            ]
        )


        # Here comes your implementation of observer
        self.state_estimate = self.next_state_estimate
        output_estimate = self.state_estimate[(0, 2), :]  # y^ = C*x^

        # TODO: IMPLEMENT THE LUENBERGER OBSERVER
        # Here comes your implementation of observer - dont' trust anything here
        self.state_estimate = self.next_state_estimate
        output_estimate = self.state_estimate[(0, 2), :]  # y^ = C*x^
        # self.next_state_estimate = np.dot((self.A - self.B * self.K), self.state_estimate) + np.dot(self.Kf, (
        #             output_measurement - output_estimate))

        # Not having the observer let me just mix the initial state with true state to have something to plot
        self.next_state_estimate = 0.9*self.state_estimate+0.1*state_true

        # Q should be calculated based on just the state_estimate
        # You can use state_true for reference solution
        Q = np.asscalar(np.dot(-self.K, state_true))
        # Q = np.asscalar(np.dot(-self.K, self.state_estimate))

        # Clip Q
        if Q > 1.0:
            Q = 1.0
        elif Q < -1.0:
            Q = -1.0
        else:
            pass

        self.controller_data_for_csv = {
                                        'position_estimate': [target_position + self.state_estimate[0, 0]],
                                        'positionD_estimate': [self.state_estimate[1, 0]],
                                        'angle_estimate': [self.state_estimate[2, 0]],
                                        'angleD_estimate': [self.state_estimate[3, 0]],
                                        'position_measurement': [target_position + output_measurement[0, 0]],
                                        'angle_measurement': [output_measurement[1, 0]],
                                        }

        return Q

if __name__ == "__main__":

    cont = controller_lqr_observer()

    # Set non-zero input
    s = s0
    s[cartpole_state_varname_to_index('position')-2] = -30.2
    s[cartpole_state_varname_to_index('positionD')-2] = 2.87
    s[cartpole_state_varname_to_index('angle')] = -0.32
    s[cartpole_state_varname_to_index('angleD')] = 0.237

    u = -0.24
