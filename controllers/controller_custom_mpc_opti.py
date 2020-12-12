"""mpc controller"""

from src.globals import *
from types import SimpleNamespace

from src.utilis import normalize_angle_rad

import matplotlib.pyplot as plt

import casadi

from copy import deepcopy


class controller_custom_mpc_opti:
    def __init__(self):

        """
        Get configured do-mpc modules:
        """

        # Physical parameters of the cart
        self.p = SimpleNamespace()  # p like parameters
        self.p.m = m_globals  # mass of pend, kg
        self.p.M = M_globals  # mass of cart, kg
        self.p.L = L_globals  # half length of pend, m
        self.p.u_max = u_max_globals  # max cart force, N
        self.p.M_fric = M_fric_globals  # cart friction, N/m/s
        self.p.J_fric = J_fric_globals  # friction coefficient on angular velocity, Nm/rad/s
        self.p.v_max = v_max_globals  # max DC motor speed, m/s, in absence of friction, used for motor back EMF model
        self.p.controlDisturbance = controlDisturbance_globals  # disturbance, as factor of u_max
        self.p.sensorNoise = sensorNoise_globals  # noise, as factor of max values
        self.p.g = g_globals  # gravity, m/s^2
        self.p.k = k_globals  # Dimensionless factor, for moment of inertia of the pend (with L being half if the length)

        # State of the cart
        self.s = SimpleNamespace()  # s like state

        self.target_position = 0.0

        self.mpc_horizon = mpc_horizon_globals
        self.dt = dt_mpc_simulation_globals

        self.yp_hat = np.zeros(self.mpc_horizon, dtype=object)  # MPC prediction of future states
        self.Q_hat = np.zeros(self.mpc_horizon)  # MPC prediction of future control inputs
        self.Q_hat0 = np.zeros(self.mpc_horizon)  # initial guess for future control inputs to be predicted
        self.Q_previous = 0.0

        self.E_kin_cart = lambda s: (s.positionD / self.p.v_max) ** 2
        self.E_kin_pol = lambda s: (s.angleD / (2 * np.pi)) ** 2
        self.E_pot_cost = lambda s: 1-np.cos(s.angle)
        self.distance_difference = lambda s: (((s.position - self.target_position) / 50.0)) ** 2

        self.Q_bounds = [(-1, 1)] * self.mpc_horizon




    def cost_function(self, Q_hat):

        # Predict future states given control_inputs Q_hat

        for k in range(0,
                       self.mpc_horizon - 1):  # We predict mpc_horizon future u, but mpc_horizon-1 y, as y0 is a current y
            if k == 0:
                self.yp_hat[0] = self.s

            cost = 0.0
            s_next = mpc_next_state(self.yp_hat[k], self.p, Q2u(Q_hat[k], self.p), dt=self.dt)

            self.yp_hat[k + 1] = s_next

        # Calculate sum of l-terms
        l_terms = 0.0
        for k in range(1, self.mpc_horizon):
            l_terms += 100 * self.distance_difference(self.yp_hat[k])*(2-self.E_pot_cost(self.yp_hat[k])) + \
                    5 * (2 - self.distance_difference(self.yp_hat[k])) * self.E_pot_cost(self.yp_hat[k]) + \
                    1.0 * (2 - self.distance_difference(self.yp_hat[k])) * self.E_kin_pol(self.yp_hat[k])

        # Calculate sum of r-terms
        r_terms = 10*(self.Q_hat[0]-self.Q_previous)**2
        for k in range(0, self.mpc_horizon - 1):
            r_terms += (self.Q_hat[k + 1] - self.Q_hat[k]) ** 2
            # normalize
            # r_terms = r_terms / self.mpc_horizon
            # Weight r-term
            r_terms = 1.0 * r_terms
            # r_terms = 0.0

        # Calculate m_term
        m_term = 5 * (self.E_kin_pol(self.yp_hat[-1]) + self.E_kin_cart(self.yp_hat[-1]) + self.E_pot_cost(self.yp_hat[-1]))

        cost += l_terms + r_terms + m_term

        return cost

    def step(self, s, target_position):

        self.s = deepcopy(s)
        self.target_position = deepcopy(target_position)

        opti = casadi.Opti();
        Q = opti.variable(len(self.Q_hat0))
        opti.minimize(self.cost_function(Q))
        opti.subject_to(Q <= 1)
        opti.subject_to(Q >= -1)
        opti.solver('ipopt', {}, {'linear_solver': 'ma27', 'print_level': 0, 'sb': 'yes'})
        try:
            sol = opti.solve()
        except:
            print(opti.debug.value(Q))
            self.Q_hat0 = np.zeros(self.mpc_horizon)
            return self.Q_previous
        self.Q_hat = sol.value(Q)

        # self.plot_prediction()

        # Prepare guess for next iteration
        self.Q_hat0 = np.hstack((self.Q_hat[1:], self.Q_hat[-1]))
        # self.Q_hat0 = np.zeros(self.mpc_horizon) # Working also in this version

        Q = self.Q_hat[0]

        return Q

    def plot_prediction(self):

        self.fig, self.axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)  # share x axis so zoom zooms all plots

        angle = []
        angleD = []
        position = []
        positionD = []
        for s in self.yp_hat:
            angle.append(s.angle)
            angleD.append(s.angleD)
            position.append(s.position)
            positionD.append(s.positionD)

        # Plot angle
        self.axs[0].set_ylabel("Angle (deg)", fontsize=18)
        self.axs[0].plot(np.array(angle) * 180.0 / np.pi, color='b',
                         markersize=12, label='Angel prediction')
        self.axs[0].tick_params(axis='both', which='major', labelsize=16)

        self.axs[1].set_ylabel("AngleD (deg/s)", fontsize=18)
        self.axs[1].plot(np.array(angleD), color='r',
                         markersize=12, label='AngelD prediction')
        self.axs[1].tick_params(axis='both', which='major', labelsize=16)

        self.axs[2].set_ylabel("Position (m)", fontsize=18)
        self.axs[2].plot(np.array(position), color='green',
                         markersize=12, label='Position prediction')
        self.axs[2].tick_params(axis='both', which='major', labelsize=16)

        self.axs[3].set_ylabel("PositionD (m/s)", fontsize=18)
        self.axs[3].plot(np.array(positionD), color='magenta',
                         markersize=12, label='PositionD prediction')
        self.axs[3].tick_params(axis='both', which='major', labelsize=16)

        self.axs[4].set_ylabel("Motor (-1,1)", fontsize=18)
        self.axs[4].plot(np.array(self.Q_hat), color='orange',
                         markersize=12, label='Force of motor')
        self.axs[4].tick_params(axis='both', which='major', labelsize=16)

        self.axs[4].set_xlabel('Time (samples)', fontsize=18)

        self.fig.align_ylabels()

        plt.show()

        return self.fig, self.axs
