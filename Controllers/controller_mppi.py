from CartPole.cartpole_model import cartpole_jacobian, cartpole_ode, p_globals, s0, Q2u
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

from copy import deepcopy
from types import SimpleNamespace

dt_mpc_simulation = 0.02  # s
mpc_horizon = 2
mc_samples = 1000


E_kin_cart = lambda s: (s.positionD / self.p.v_max) ** 2
E_kin_pol = lambda s: (s.angleD / (2 * np.pi)) ** 2
E_pot_cost = lambda s: 1 - np.cos(s.angle)
distance_difference = (
    lambda s, target_position: (((s.position - target_position) / 50.0)) ** 2
)


def state_to_array(s):
    state_list = [s.position, s.positionD, s.positionDD, s.angle, s.angleD, s.angleDD]
    return np.array(state_list)


def array_to_state(s):
    state = SimpleNamespace()
    state.position = s[0]
    state.positionD = s[1]
    state.positionDD = s[2]
    state.angle = s[3]
    state.angleD = s[4]
    state.angleDD = s[5]
    return state


@jit(nopython=True)
def trajectory_rollouts(
    s, S_tilde_k, u, delta_u, mc_samples, mpc_samples, p, dt, target_position
):
    for k in range(mc_samples):
        s_horizon = np.zeros((mpc_samples, s.size))
        s_horizon[0, :] = s
        for i in range(1, mpc_samples):
            s_last = s_horizon[i - 1, :]
            derivatives =  motion_derivatives(
                p, array_to_state(s_last), u[i] + delta_u[k, i]
            )
            s_next = s_last + derivatives * dt
            # s_next = SimpleNamespace()
            # s_next.position = s_last.position + dx * self.dt
            # s_next.positionD = s_last.positionD + ddx * self.dt
            # s_next.angle = s_last.angle + da * self.dt
            # s_next.angleD = s_last.angleD + dda * self.dt

            # s_horizon.append(s_next)
            s_horizon[i, :] = s_next
            # self.S_tilde[k, i + 1] = (
            #     self.S_tilde[k, i]
            #     + self.q(self.p, s_next, self.u[i], self.delta_u[k, i]) * self.dt
            # )
            S_tilde_k[k] += q(
                p, array_to_state(s_next), u[i], delta_u[k, i], target_position
            )
    return s_horizon, S_tilde_k


def motion_derivatives(p, s, u):
    """
    :return: The time derivative vector d/dt([x, dx/dt, theta, dtheta/dt])
    """
    dx = s[1]
    da = s[4]
    dda, ddx = cartpole_ode(p, array_to_state(s), p.u_max * u)  # cartpole_ode(p, s, Q2u(u,p))

    return np.array([dx, ddx, 0, da, dda, 0])


def q(p, s, u, delta_u, target_position):
    if np.abs(u + delta_u) > 1.0:
        return 1.0e5
    dd = 10 * distance_difference(s, target_position)
    ep = E_pot_cost(s) ** 2
    ekp = E_kin_pol(s)
    ekc = E_kin_cart(s)
    q = dd + ep + ekp + ekc
    # self.distance_differences.append(dd)
    # self.E_pots.append(ep)
    # self.E_kins_pole.append(ekp)
    # self.E_kins_cart.append(ekc)

    q += (
        0.5 * (1 - 1.0 / self.nu) * self.R * (delta_u ** 2)
        + self.R * u * delta_u
        + 0.5 * self.R * (u ** 2)
    )
    return q


class controller_mppi:
    def __init__(self):
        # Physical parameters of the cart
        self.p = p_globals

        # State of the cart
        self.s = SimpleNamespace()  # s like state

        np.random.seed(123)

        self.target_position = 0.0

        self.mpc_horizon = mpc_horizon
        self.dt = dt_mpc_simulation
        self.mpc_samples = int(self.mpc_horizon / self.dt)
        self.mc_samples = mc_samples

        self.Q_bounds = [(-1, 1)] * self.mpc_horizon

        self.Q = np.diag([10.0, 1.0, 1.0, 1.0])  # How much to punish x, v, theta, omega
        self.R = 1.0e0  # How much to punish Q
        self.l = 10  # cost parameter lambda
        self.nu = 1.0e1  # Exploration variance
        self.rho_sqrt_inv = 0.01
        self.avg_cost = []
        self.distance_differences = []
        self.E_pots = []
        self.E_kins_pole = []
        self.E_kins_cart = []
        self.num_timesteps = 0

        self.s_horizon = []  # list of states s
        self.u = np.zeros((self.mpc_samples), dtype=float)
        self.delta_u = np.zeros((self.mc_samples, self.mpc_samples), dtype=float)
        self.S_tilde = np.zeros((self.mc_samples, self.mpc_samples), dtype=float)
        self.S_tilde_k = np.zeros((self.mc_samples), dtype=float)

    def q_tilde(self):
        pass

    def cost_to_go(self):
        pass

    def reward_weighted_average(self, S_i, delta_u_i):
        rho = np.min(S_i)  # for numerical stability
        exp_s = np.exp(-1.0 / self.l * (S_i - rho))
        a = np.sum(exp_s)
        b = np.sum(np.multiply(exp_s, delta_u_i) / a)
        return b

    def step(self, s, target_position, time=None):
        self.s = deepcopy(s)
        self.target_position = deepcopy(target_position).item()

        self.num_timesteps += 1

        # self.delta_u = (
        #     np.random.normal(size=np.shape(self.delta_u))
        #     * self.rho_sqrt_inv
        #     / (np.sqrt(self.dt))
        # )  # N(mean=0, var=1/(rho*dt))
        self.delta_u = np.random.normal(size=np.shape(self.delta_u)) * 0.2
        self.S_tilde = np.zeros_like(self.S_tilde)
        self.S_tilde_k = np.zeros_like(self.S_tilde_k)

        # TODO: Parallelize loop over k
        self.s_horizon, self.S_tilde_k = trajectory_rollouts(
            state_to_array(self.s),
            self.S_tilde_k,
            self.u,
            self.delta_u,
            self.mc_samples,
            self.mpc_samples,
            self.p,
            self.dt,
            self.target_position,
        )

        self.avg_cost.append(np.mean(self.S_tilde_k, axis=0))

        for i in range(self.mpc_samples):
            # self.u[i] += self.reward_weighted_average(
            #     self.S_tilde[:, -1] - self.S_tilde[:, i], self.delta_u[:, i]
            # )
            self.u[i] += self.reward_weighted_average(
                self.S_tilde_k, self.delta_u[:, i]
            )

        Q = np.clip(self.u[0], -1, 1)
        # Q = self.u[0]

        # Index shift inputs
        self.u[:-1] = self.u[1:]
        # self.u[-1] = 0

        return Q  # normed control input in the range [-1,1]

    # Optionally: A method called after an experiment.
    # May be used to print some statistics about controller performance (e.g. number of iter. to converge)
    def controller_report(self):
        # TODO: Graph the running cost per iteration to see if the controller minimizes it
        time_axis = self.dt * 10 * np.arange(start=0, stop=len(self.avg_cost))
        plt.figure(num=2, figsize=(8, 8))
        plt.plot(time_axis, self.avg_cost)
        plt.ylabel("avg_cost")
        plt.xlabel("time")
        plt.title("Cost over iterations")
        plt.show()

        self.num_timesteps -= 1

        self.distance_differences = np.reshape(
            np.array(
                self.distance_differences[self.mc_samples * (self.mpc_samples - 1) :]
            ),
            (self.num_timesteps, self.mc_samples, self.mpc_samples - 1),
        )
        self.E_pots = np.reshape(
            np.array(self.E_pots[self.mc_samples * (self.mpc_samples - 1) :]),
            (self.num_timesteps, self.mc_samples, self.mpc_samples - 1),
        )
        self.E_kins_pole = np.reshape(
            np.array(self.E_kins_pole[self.mc_samples * (self.mpc_samples - 1) :]),
            (self.num_timesteps, self.mc_samples, self.mpc_samples - 1),
        )
        self.E_kins_cart = np.reshape(
            np.array(self.E_kins_cart[self.mc_samples * (self.mpc_samples - 1) :]),
            (self.num_timesteps, self.mc_samples, self.mpc_samples - 1),
        )

        plt.figure(num=3, figsize=(12, 8))
        plt.plot(
            np.mean(self.distance_differences[:, :, -1], axis=1),
            label="Distance difference cost",
        )
        plt.plot(np.mean(self.E_pots[:, :, -1], axis=1), label="E_pot cost")
        plt.plot(np.mean(self.E_kins_pole[:, :, -1], axis=1), label="E_kin_pole cost")
        plt.plot(np.mean(self.E_kins_cart[:, :, -1], axis=1), label="E_kin_cart cost")
        plt.title("Cost components over time")
        plt.legend()
        plt.show()

    # Optionally: reset the controller after an experiment
    # May be useful for stateful controllers, like these containing RNN,
    # To reload the hidden states e.g. if the controller went unstable in the previous run.
    # It is called after an experiment,
    # but only if the controller is supposed to be reused without reloading (e.g. in GUI)
    def controller_reset(self):
        pass
