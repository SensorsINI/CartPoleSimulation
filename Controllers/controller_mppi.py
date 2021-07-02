"""
Model Predictive Path Integral Controller
Based on Williams, Aldrich, Theodorou (2015)
"""

# Uncomment if you want to get interactive plots for MPPI in Pycharm on MacOS
# On other OS you have to chose a different interactive backend.
# from matplotlib import use
# # # use('TkAgg')
# use('macOSX')

import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from datetime import datetime

from CartPole._CartPole_mathematical_helpers import (
    conditional_decorator,
    wrap_angle_rad_inplace,
)
from CartPole.cartpole_model import TrackHalfLength
from CartPole.state_utilities import (
    ANGLE_COS_IDX,
    ANGLE_IDX,
    ANGLED_IDX,
    ANGLE_SIN_IDX,
    POSITION_IDX,
    POSITIOND_IDX,
    STATE_VARIABLES,
    STATE_INDICES,
    create_cartpole_state,
)
from matplotlib.widgets import Slider
from numba import jit
from numpy.random import SFC64, Generator
from SI_Toolkit_ApplicationSpecificFiles.predictor_ideal import predictor_ideal
from scipy.interpolate import interp1d
from SI_Toolkit.TF.TF_Functions.predictor_autoregressive_tf import (
    predictor_autoregressive_tf,
)

from Controllers.template_controller import template_controller

config = yaml.load(
    open(os.path.join("SI_Toolkit_ApplicationSpecificFiles", "config.yml"), "r"), Loader=yaml.FullLoader
)
NET_NAME = config["modeling"]["NET_NAME"]
try:
    NET_TYPE = NET_NAME.split("-")[0]
except AttributeError:  # Should get Attribute Error if NET_NAME is None
    NET_TYPE = None

config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
"""Timestep and sampling settings"""
dt = config["controller"]["mppi"]["dt"]
mpc_horizon = config["controller"]["mppi"]["mpc_horizon"]
mpc_samples = int(mpc_horizon / dt)  # Number of steps in MPC horizon
num_rollouts = config["controller"]["mppi"]["num_rollouts"]
update_every = config["controller"]["mppi"]["update_every"]
predictor_type = config["controller"]["mppi"]["predictor_type"]


"""Parameters weighting the different cost components"""
dd_weight = config["controller"]["mppi"]["dd_weight"]
ep_weight = config["controller"]["mppi"]["ep_weight"]
ekp_weight = config["controller"]["mppi"]["ekp_weight"]
ekc_weight = config["controller"]["mppi"]["ekc_weight"]
cc_weight = config["controller"]["mppi"]["cc_weight"]
ccrc_weight = config["controller"]["mppi"]["ccrc_weight"]

"""Perturbation factor"""
p_Q = config["controller"]["mppi"]["control_noise"]
dd_noise = ep_noise = ekp_noise = ekc_noise = cc_noise = config["controller"]["mppi"][
    "cost_noise"
]


dd_weight = dd_weight * (1 + dd_noise * np.random.uniform(-1.0, 1.0))
ep_weight = ep_weight * (1 + ep_noise * np.random.uniform(-1.0, 1.0))
ekp_weight = ekp_weight * (1 + ekp_noise * np.random.uniform(-1.0, 1.0))
ekc_weight = ekc_weight * (1 + ekc_noise * np.random.uniform(-1.0, 1.0))
cc_weight = cc_weight * (1 + cc_noise * np.random.uniform(-1.0, 1.0))


gui_dd = gui_ep = gui_ekp = gui_ekc = gui_cc = gui_ccrc = np.zeros(1, dtype=np.float32)


"""MPPI constants"""
R = config["controller"]["mppi"]["R"]
LBD = config["controller"]["mppi"]["LBD"]
NU = config["controller"]["mppi"]["NU"]
SQRTRHODTINV = config["controller"]["mppi"]["SQRTRHOINV"] * (1 / np.math.sqrt(dt))
GAMMA = config["controller"]["mppi"]["GAMMA"]
SAMPLING_TYPE = config["controller"]["mppi"]["SAMPLING_TYPE"]

"""Random number generator"""
rng = Generator(SFC64(int((datetime.now() - datetime(1970, 1, 1)).total_seconds())))


"""Init logging variables"""
LOGGING = config["controller"]["mppi"]["LOGGING"]
# Save average cost for each cost component
LOGS = {
    "cost_to_go": [],
    "cost_breakdown": {
        "cost_dd": [],
        "cost_ep": [],
        "cost_ekp": [],
        "cost_ekc": [],
        "cost_cc": [],
        "cost_ccrc": [],
    },
    "states": [],
    "trajectory": [],
    "target_trajectory": [],
    "inputs": [],
    "nominal_rollouts": [],
}


"""Cost function helpers"""


@jit(nopython=True, cache=True, fastmath=True)
def E_kin_cart(positionD):
    """Compute penalty for kinetic energy of cart"""
    return positionD ** 2


@jit(nopython=True, cache=True, fastmath=True)
def E_kin_pol(angleD):
    """Compute penalty for kinetic energy of pole"""
    return angleD ** 2


@jit(nopython=True, cache=True, fastmath=True)
def E_pot_cost(angle):
    """Compute penalty for not balancing pole upright (penalize large angles)"""
    return 0.25 * (1.0 - np.cos(angle)) ** 2
    # return angle ** 2


@jit(nopython=True, cache=True, fastmath=True)
def distance_difference_cost(position, target_position):
    """Compute penalty for distance of cart to the target position"""
    return ((position - target_position) / (2.0 * TrackHalfLength)) ** 2 + (
        np.abs(position) > 0.95 * TrackHalfLength
    ) * 1.0e6  # Soft constraint: Do not crash into border


@jit(nopython=True, cache=True, fastmath=True)
def control_change_rate_cost(u, u_prev):
    """Compute penalty of control jerk, i.e. difference to previous control input"""
    return (u - u_prev) ** 2


@jit(nopython=True, cache=True, fastmath=True)
def penalize_deviation(cc, u):
    """Compute penalty for producing inputs that do not fulfil input constraints"""
    # Penalize if control deviation is outside constraint set.
    I, J = cc.shape
    for i in range(I):
        for j in range(J):
            if np.abs(u[i, j]) > 1.0:
                cc[i, j] = 1.0e5
    return cc


"""Define Predictor"""
if predictor_type == "Euler":
    predictor = predictor_ideal(horizon=mpc_samples, dt=dt, intermediate_steps=1)
elif predictor_type == "NeuralNet":
    predictor = predictor_autoregressive_tf(
        horizon=mpc_samples, batch_size=num_rollouts, net_name=NET_NAME
    )


def trajectory_rollouts(
    s: np.ndarray,
    S_tilde_k: np.ndarray,
    u: np.ndarray,
    delta_u: np.ndarray,
    u_prev: np.ndarray,
    target_position: np.float32,
):
    """Sample thousands of rollouts using system model. Compute cost-weighted control update. Log states and costs if specified.

    :param s: Current state of the system
    :type s: np.ndarray
    :param S_tilde_k: Placeholder array to store the cost of each rollout trajectory
    :type S_tile_k: np.ndarray
    :param u: Vector of nominal inputs computed in previous iteration
    :type u: np.ndarray
    :param delta_u: Array containing all input perturbation samples. Shape (num_rollouts x horizon_steps)
    :type delta_u: np.ndarray
    :param u_prev: Array with nominal inputs from previous iteration. Used to compute cost of control change
    :type u_prev: np.ndarray
    :param target_position: Target position where the cart should move to
    :type target_position: np.float32

    :return: S_tilde_k - Array filled with a cost for each rollout trajectory
    """
    initial_state = np.tile(s, (num_rollouts, 1))

    predictor.setup(initial_state=initial_state, prediction_denorm=True)
    s_horizon = predictor.predict(u + delta_u)[:, :, : len(STATE_INDICES)]

    # Compute stage costs
    cost_increment, dd, ep, ekp, ekc, cc, ccrc = q(
        s_horizon[:, 1:, :], u, delta_u, u_prev, target_position
    )
    S_tilde_k = np.sum(cost_increment, axis=1)
    # Compute terminal cost
    S_tilde_k += phi(s_horizon, target_position)

    # Pass costs to GUI popup window
    global gui_dd, gui_ep, gui_ekp, gui_ekc, gui_cc, gui_ccrc
    gui_dd, gui_ep, gui_ekp, gui_ekc, gui_cc, gui_ccrc = (
        np.mean(dd),
        np.mean(ep),
        np.mean(ekp),
        np.mean(ekc),
        np.mean(cc),
        np.mean(ccrc),
    )

    if LOGGING:
        LOGS.get("cost_breakdown").get("cost_dd").append(np.mean(dd, 0))
        LOGS.get("cost_breakdown").get("cost_ep").append(np.mean(ep, 0))
        LOGS.get("cost_breakdown").get("cost_ekp").append(np.mean(ekp, 0))
        LOGS.get("cost_breakdown").get("cost_ekc").append(np.mean(ekc, 0))
        LOGS.get("cost_breakdown").get("cost_cc").append(np.mean(cc, 0))
        LOGS.get("cost_breakdown").get("cost_ccrc").append(np.mean(ccrc, 0))
        # (1 x mpc_samples)
        LOGS.get("states").append(
            np.copy(s_horizon[:, :-1, :])
        )  # num_rollouts x mpc_samples x STATE_VARIABLES

    return S_tilde_k


def q(
    s: np.ndarray,
    u: np.ndarray,
    delta_u: np.ndarray,
    u_prev: np.ndarray,
    target_position: np.float32,
) -> np.ndarray:
    """Stage cost function. Computes stage-cost elementwise for all rollouts and all trajectory steps at once.

    :param s: Current states of all rollouts
    :type s: np.ndarray
    :param u: Vector of nominal inputs
    :type u: np.ndarray
    :param delta_u: Array of perturbations
    :type delta_u: np.ndarray
    :param u_prev: Vector of nominal inputs of previous iteration
    :type u_prev: np.ndarray
    :param target_position: Target position where the cart should move to
    :type target_position: np.float32
    :return:
        - q - Summed stage cost
        - dd - Distance difference cost
        - ep - Cost to keep pole upright
        - ekp - Cost of pole kinetic energy
        - ekc - Cost of cart kinetic energy
        - cc - Control cost
        - ccrc - Control change rate cost
    """
    dd = dd_weight * distance_difference_cost(
        s[:, :, POSITION_IDX], target_position
    ).astype(np.float32)
    ep = ep_weight * E_pot_cost(s[:, :, ANGLE_IDX]).astype(np.float32)
    ekp = ekp_weight * E_kin_pol(s[:, :, ANGLED_IDX]).astype(np.float32)
    ekc = ekc_weight * E_kin_cart(s[:, :, POSITIOND_IDX]).astype(np.float32)
    cc = cc_weight * (
        0.5 * (1 - 1.0 / NU) * R * (delta_u ** 2) + R * u * delta_u + 0.5 * R * (u ** 2)
    )
    ccrc = ccrc_weight * control_change_rate_cost(u + delta_u, u_prev).astype(
        np.float32
    )
    # rterm = 1.0e4 * np.sum((delta_u[:,1:] - delta_u[:,:-1]) ** 2, axis=1, keepdims=True)

    # Penalize if control deviation is outside constraint set.
    cc[np.abs(u + delta_u) > 1.0] = 1.0e5

    q = dd + ep + ekp + ekc + cc + ccrc

    return q, dd, ep, ekp, ekc, cc, ccrc


@jit(nopython=True, cache=True, fastmath=True)
def phi(s: np.ndarray, target_position: np.float32) -> np.ndarray:
    """Calculate terminal cost of a set of trajectories

    Williams et al use an indicator function type of terminal cost in
    "Information theoretic MPC for model-based reinforcement learning"

    TODO: Try a quadratic terminal cost => Use the LQR terminal cost term obtained
    by linearizing the system around the unstable equilibrium.

    :param s: Reference to numpy array of states of all rollouts
    :type s: np.ndarray
    :param target_position: Target position to move the cart to
    :type target_position: np.float32
    :return: One terminal cost per rollout
    :rtype: np.ndarray
    """
    terminal_states = s[:, -1, :]
    terminal_cost = 10000 * (
        (np.abs(terminal_states[:, ANGLE_IDX]) > 0.2)
        | (
            np.abs(terminal_states[:, POSITION_IDX] - target_position)
            > 0.1 * TrackHalfLength
        )
    )
    return terminal_cost


@jit(nopython=True, cache=True, fastmath=True)
def reward_weighted_average(S: np.ndarray, delta_u: np.ndarray):
    """Average the perturbations delta_u based on their desirability

    :param S: Array of rollout costs
    :type S: np.ndarray
    :param delta_u: Array of perturbations
    :type delta_u: np.ndarray
    :return: Gain to update the vector of nominal inputs by. Vector of length (horizon_steps)
    :rtype: np.ndarray
    """
    rho = np.min(S)  # for numerical stability
    exp_s = np.exp(-1.0 / LBD * (S - rho))
    a = np.sum(exp_s)
    b = np.sum(np.multiply(np.expand_dims(exp_s, 1), delta_u) / a, axis=0)
    return b


@jit(nopython=True, cache=True, fastmath=True)
def update_inputs(u: np.ndarray, S: np.ndarray, delta_u: np.ndarray):
    """Reward-weighted in-place update of nominal control inputs according to the MPPI method.

    :param u: Sampling mean / warm started control inputs of size (,mpc_samples)
    :type u: np.ndarray
    :param S: Cost array of size (num_rollouts)
    :type S: np.ndarray
    :param delta_u: The input perturbations that had been used, shape (num_rollouts x mpc_samples)
    :type delta_u: np.ndarray
    """
    u += reward_weighted_average(S, delta_u)


class controller_mppi(template_controller):
    """Controller implementing the Model Predictive Path Integral method (Williams et al. 2015)

    :param template_controller: Superclass describing the basic controller interface
    :type template_controller: abc.ABC
    """

    def __init__(self):
        # State of the cart
        self.s = create_cartpole_state()

        self.target_position = 0.0

        self.rho_sqrt_inv = 0.01

        self.iteration = -1
        self.control_enabled = True

        self.s_horizon = np.zeros((), dtype=np.float32)
        self.u = np.zeros((mpc_samples), dtype=np.float32)
        self.u_prev = np.zeros_like(self.u, dtype=np.float32)
        self.delta_u = np.zeros((num_rollouts, mpc_samples), dtype=np.float32)
        self.S_tilde_k = np.zeros((num_rollouts), dtype=np.float32)

        self.warm_up_len = 100
        self.warm_up_countdown = self.warm_up_len
        try:
            from Controllers.controller_lqr import controller_lqr

            self.auxiliary_controller_available = True
            self.auxiliary_controller = controller_lqr()
        except ModuleNotFoundError:
            self.auxiliary_controller_available = False
            self.auxiliary_controller = None

    def initialize_perturbations(
        self, stdev: float = 1.0, sampling_type: str = None
    ) -> np.ndarray:
        """Sample an array of control perturbations delta_u. Samples for two distinct rollouts are always independent

        :param stdev: standard deviation of samples if Gaussian, defaults to 1.0
        :type stdev: float, optional
        :param sampling_type: defaults to None, can be one of
            - "random_walk" - The next horizon step's perturbation is correlated with the previous one
            - "uniform" - Draw uniformly distributed samples between -1.0 and 1.0
            - "repeated" - Sample only one perturbation per rollout, apply it repeatedly over the course of the rollout
            - "interpolated" - Sample a new independent perturbation every 10th MPC horizon step. Interpolate in between the samples
            - "iid" - Sample independent and identically distributed samples of a Gaussian distribution
        :type sampling_type: str, optional
        :return: Independent perturbation samples of shape (num_rollouts x horizon_steps)
        :rtype: np.ndarray
        """
        """
        Return a numpy array with the perturbations delta_u.
        If random_walk is false, initialize with independent Gaussian samples
        If random_walk is true, each row represents a 1D random walk with Gaussian steps.
        """
        if sampling_type == "random_walk":
            delta_u = np.empty((num_rollouts, mpc_samples), dtype=np.float32)
            delta_u[:, 0] = stdev * rng.standard_normal(
                size=(num_rollouts,), dtype=np.float32
            )
            for i in range(1, mpc_samples):
                delta_u[:, i] = delta_u[:, i - 1] + stdev * rng.standard_normal(
                    size=(num_rollouts,), dtype=np.float32
                )
        elif sampling_type == "uniform":
            delta_u = np.empty((num_rollouts, mpc_samples), dtype=np.float32)
            for i in range(0, mpc_samples):
                delta_u[:, i] = rng.uniform(
                    low=-1.0, high=1.0, size=(num_rollouts,)
                ).astype(np.float32)
        elif sampling_type == "repeated":
            delta_u = np.tile(
                stdev * rng.standard_normal(size=(num_rollouts, 1), dtype=np.float32),
                (1, mpc_samples),
            )
        elif sampling_type == "interpolated":
            step = 10
            range_stop = int(np.ceil((mpc_samples) / step) * step) + 1
            t = np.arange(start=0, stop=range_stop, step=step)
            t_interp = np.arange(start=0, stop=range_stop, step=1)
            t_interp = np.delete(t_interp, t)
            delta_u = np.zeros(shape=(num_rollouts, range_stop), dtype=np.float32)
            delta_u[:, t] = stdev * rng.standard_normal(
                size=(num_rollouts, t.size), dtype=np.float32
            )
            f = interp1d(t, delta_u[:, t])
            delta_u[:, t_interp] = f(t_interp)
            delta_u = delta_u[:, :mpc_samples]
        else:
            delta_u = stdev * rng.standard_normal(
                size=(num_rollouts, mpc_samples), dtype=np.float32
            )

        return delta_u

    def step(self, s: np.ndarray, target_position: np.float64, time=None):
        """Perform controller step

        :param s: State passed to controller after system has evolved for one step
        :type s: np.ndarray
        :param target_position: Target position where the cart should move to
        :type target_position: np.float64
        :param time: Time in seconds that has passed in the current experiment, defaults to None
        :type time: float, optional
        :return: A normed control value in the range [-1.0, 1.0]
        :rtype: np.float32
        """
        self.s = s
        self.target_position = np.float32(target_position)

        self.iteration += 1

        # Adjust horizon if changed in GUI while running
        # FIXME: For this to work with NeuralNet predictor we need to build a setter,
        #  which also reinitialize arrays which size depends on horizon
        predictor.horizon = mpc_samples
        if mpc_samples != self.u.size:
            self.update_control_vector()

        if self.iteration % update_every == 0:
            # Initialize perturbations and cost arrays
            self.delta_u = self.initialize_perturbations(
                # stdev=0.1 * (1 + 1 / (self.iteration + 1)),
                stdev=SQRTRHODTINV,
                sampling_type=SAMPLING_TYPE,
            )  # du ~ N(mean=0, var=1/(rho*dt))
            self.S_tilde_k = np.zeros_like(self.S_tilde_k, dtype=np.float32)

            # Run parallel trajectory rollouts for different input perturbations
            self.S_tilde_k = trajectory_rollouts(
                self.s,
                self.S_tilde_k,
                self.u,
                self.delta_u,
                self.u_prev,
                self.target_position,
            )

            # Update inputs with weighted perturbations
            update_inputs(self.u, self.S_tilde_k, self.delta_u)

            # Log states and costs incurred for plotting later
            if LOGGING:
                LOGS.get("cost_to_go").append(np.copy(self.S_tilde_k))
                LOGS.get("inputs").append(np.copy(self.u))

                # Simulate nominal rollout to plot the trajectory the controller wants to make
                # Compute one rollout of shape (mpc_samples + 1) x s.size
                if predictor_type == "Euler":
                    predictor.setup(
                        initial_state=np.copy(self.s), prediction_denorm=True
                    )
                    rollout_trajectory = predictor.predict(self.u)
                elif predictor_type == "NeuralNet":
                    predictor.setup(
                        initial_state=np.tile(self.s, (num_rollouts, 1)),
                        prediction_denorm=True,
                    )
                    # This is a lot of unnecessary calculation, but a stateful RNN in TF has frozen batch size
                    rollout_trajectory = predictor.predict(
                        np.tile(self.u, (num_rollouts, 1))
                    )[0, ...]
                LOGS.get("nominal_rollouts").append(np.copy(rollout_trajectory[:-1, :]))

        if LOGGING:
            LOGS.get("trajectory").append(np.copy(self.s))
            LOGS.get("target_trajectory").append(np.copy(target_position))

        if (
            self.warm_up_countdown > 0
            and self.auxiliary_controller_available
            and (NET_TYPE == "GRU" or NET_TYPE == "LSTM" or NET_TYPE == "RNN")
            and predictor_type == "NeuralNet"
        ):
            self.warm_up_countdown -= 1
            Q = self.auxiliary_controller.step(s, target_position)
        else:
            Q = self.u[0]

        # A snippet of code to switch on and off the controller to cover better the statespace with experimental data
        # It stops controller when Pole is well stabilized (starting inputing random input)
        # And re-enables it when angle exceedes 90 deg.
        # if (abs(self.s[[ANGLE_IDX]]) < 0.01
        #     and abs(self.s[[POSITION_IDX]]-self.target_position < 0.02)
        #         and abs(self.s[[ANGLED_IDX]]) < 0.1
        #             and abs(self.s[[POSITIOND_IDX]]) < 0.05):
        #     self.control_enabled = False
        # elif abs(self.s[[ANGLE_IDX]]) > np.pi/2:
        #     self.control_enabled = True
        #
        # if self.control_enabled is True:
        #     Q = self.u[0]
        # else:
        #     Q = np.random.uniform(-1.0, 1.0)

        # Add noise on top of the calculated Q value to better explore state space
        Q = np.float32(Q * (1 + p_Q * np.random.uniform(-1.0, 1.0)))
        # Clip inputs to allowed range
        Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)

        # Preserve current series of inputs
        self.u_prev = np.copy(self.u)

        # Index-shift inputs
        self.u[:-1] = self.u[1:]
        self.u[-1] = 0
        # self.u = zeros_like(self.u)

        # Prepare predictor for next timestep
        Q_update = np.tile(Q, (num_rollouts, 1))
        predictor.update_internal_state(Q_update)

        return Q  # normed control input in the range [-1,1]

    def update_control_vector(self):
        """
        MPPI stores a vector of best-guess-so-far control inputs for future steps.
        When adjusting the horizon length, need to adjust this vector too.
        Init with zeros when lengthening, and slice when shortening horizon.
        """
        update_length = min(mpc_samples, self.u.size)
        u_new = np.zeros((mpc_samples), dtype=np.float32)
        u_new[:update_length] = self.u[:update_length]
        self.u = u_new
        self.u_prev = np.copy(self.u)

    def controller_report(self):
        if LOGGING:
            ### Plot the average state cost per iteration
            ctglgs = np.stack(
                LOGS.get("cost_to_go"), axis=0
            )  # ITERATIONS x num_rollouts
            NUM_ITERATIONS = np.shape(ctglgs)[0]
            time_axis = update_every * dt * np.arange(start=0, stop=np.shape(ctglgs)[0])
            plt.figure(num=2, figsize=(16, 9))
            plt.plot(time_axis, np.mean(ctglgs, axis=1))
            plt.ylabel("Average Running Cost")
            plt.xlabel("time (s)")
            plt.title("Cost-to-go per Timestep")
            plt.show()

            ### Graph the different cost components per iteration
            LOGS["cost_breakdown"]["cost_dd"] = np.stack(
                LOGS.get("cost_breakdown").get("cost_dd"), axis=0
            )  # ITERATIONS x mpc_samples
            LOGS["cost_breakdown"]["cost_ep"] = np.stack(
                LOGS.get("cost_breakdown").get("cost_ep"), axis=0
            )
            LOGS["cost_breakdown"]["cost_ekp"] = np.stack(
                LOGS.get("cost_breakdown").get("cost_ekp"), axis=0
            )
            LOGS["cost_breakdown"]["cost_ekc"] = np.stack(
                LOGS.get("cost_breakdown").get("cost_ekc"), axis=0
            )
            LOGS["cost_breakdown"]["cost_cc"] = np.stack(
                LOGS.get("cost_breakdown").get("cost_cc"), axis=0
            )
            LOGS["cost_breakdown"]["cost_ccrc"] = np.stack(
                LOGS.get("cost_breakdown").get("cost_ccrc"), axis=0
            )
            time_axis = update_every * dt * np.arange(start=0, stop=NUM_ITERATIONS)

            plt.figure(num=3, figsize=(16, 9))
            plt.plot(
                time_axis,
                np.sum(LOGS.get("cost_breakdown").get("cost_dd"), axis=-1),
                label="Distance difference cost",
            )
            plt.plot(
                time_axis,
                np.sum(LOGS.get("cost_breakdown").get("cost_ep"), axis=-1),
                label="E_pot cost",
            )
            plt.plot(
                time_axis,
                np.sum(LOGS.get("cost_breakdown").get("cost_ekp"), axis=-1),
                label="E_kin_pole cost",
            )
            plt.plot(
                time_axis,
                np.sum(LOGS.get("cost_breakdown").get("cost_ekc"), axis=-1),
                label="E_kin_cart cost",
            )
            plt.plot(
                time_axis,
                np.sum(LOGS.get("cost_breakdown").get("cost_cc"), axis=-1),
                label="Control cost",
            )
            plt.plot(
                time_axis,
                np.sum(LOGS.get("cost_breakdown").get("cost_ccrc"), axis=-1),
                label="Control change rate cost",
            )

            plt.ylabel("total horizon cost")
            plt.xlabel("time (s)")
            plt.title("Cost component breakdown")
            plt.legend()
            plt.show()

            ### Draw the trajectory rollouts simulated by MPPI
            def draw_rollouts(
                angles: np.ndarray,
                positions: np.ndarray,
                ax_position: plt.Axes,
                ax_angle: plt.Axes,
                costs: np.ndarray,
                iteration: int,
            ):
                mc_rollouts = np.shape(angles)[0]
                horizon_length = np.shape(angles)[1]
                # Loop over all MC rollouts
                for i in range(0, 2000, 5):
                    ax_position.plot(
                        (update_every * iteration + np.arange(0, horizon_length)) * dt,
                        positions[i, :],
                        linestyle="-",
                        linewidth=1,
                        color=(
                            0.0,
                            (1 - 0.3 * costs[i]) ** 2,
                            0.0,
                            0.02 * (1 - 0.3 * costs[i]) ** 2,
                        ),
                    )
                    ax_angle.plot(
                        (update_every * iteration + np.arange(0, horizon_length)) * dt,
                        angles[i, :] * 180.0 / np.pi,
                        linestyle="-",
                        linewidth=1,
                        color=(
                            0.0,
                            (1 - 0.3 * costs[i]) ** 2,
                            0.0,
                            0.02 * (1 - 0.3 * costs[i]) ** 2,
                        ),
                    )

            # Prepare data
            # shape(slgs) = ITERATIONS x num_rollouts x mpc_samples x STATE_VARIABLES
            slgs = np.stack(LOGS.get("states"), axis=0)
            wrap_angle_rad_inplace(slgs[:, :, :, ANGLE_IDX])
            # shape(iplgs) = ITERATIONS x mpc_horizon
            iplgs = np.stack(LOGS.get("inputs"), axis=0)
            # shape(nrlgs) = ITERATIONS x mpc_horizon x STATE_VARIABLES
            nrlgs = np.stack(LOGS.get("nominal_rollouts"), axis=0)
            wrap_angle_rad_inplace(nrlgs[:, :, ANGLE_IDX])
            # shape(trjctlgs) = (update_every * ITERATIONS) x STATE_VARIABLES
            trjctlgs = np.stack(LOGS.get("trajectory")[:-1], axis=0)
            wrap_angle_rad_inplace(trjctlgs[:, ANGLE_IDX])
            # shape(trgtlgs) = ITERATIONS x [position]
            trgtlgs = np.stack(LOGS.get("target_trajectory")[:-1], axis=0)
            # For each rollout, calculate what the nominal trajectory would be using the known true model
            # This can uncover if the model used makes inaccurate predictions
            # shape(true_nominal_rollouts) = ITERATIONS x mpc_horizon x [position, positionD, angle, angleD]
            predictor_true_equations = predictor_ideal(
                horizon=mpc_samples, dt=dt, intermediate_steps=10
            )
            predictor_true_equations.setup(
                np.copy(nrlgs[:, 0, :]), prediction_denorm=True
            )
            true_nominal_rollouts = predictor_true_equations.predict(iplgs)[:, :-1, :]
            wrap_angle_rad_inplace(true_nominal_rollouts[:, :, ANGLE_IDX])

            # Create figure
            fig, (ax1, ax2) = plt.subplots(
                nrows=2,
                ncols=1,
                num=5,
                figsize=(16, 9),
                sharex=True,
                gridspec_kw={"bottom": 0.15, "left": 0.1, "right": 0.84, "top": 0.95},
            )

            # Create time slider
            slider_axis = plt.axes([0.15, 0.02, 0.7, 0.03])
            slider = Slider(
                slider_axis, "timestep", 1, np.shape(slgs)[0], valinit=1, valstep=1
            )

            # Normalize cost to go to use as opacity in plot
            # shape(ctglgs) = ITERATIONS x num_rollouts
            ctglgs = np.divide(ctglgs.T, np.max(np.abs(ctglgs), axis=1)).T

            # This function updates the plot when a new iteration is selected
            def update_plot(i):
                # Clear previous iteration plot
                ax1.clear()
                ax2.clear()

                # Plot Monte Carlo rollouts
                draw_rollouts(
                    slgs[i - 1, :, :, ANGLE_IDX],
                    slgs[i - 1, :, :, POSITION_IDX],
                    ax1,
                    ax2,
                    ctglgs[i - 1, :],
                    i - 1,
                )

                # Plot the realized trajectory
                ax1.plot(
                    np.arange(0, np.shape(trjctlgs)[0]) * dt,
                    trjctlgs[:, POSITION_IDX],
                    alpha=1.0,
                    linestyle="-",
                    linewidth=1,
                    color="g",
                    label="realized trajectory",
                )
                ax2.plot(
                    np.arange(0, np.shape(trjctlgs)[0]) * dt,
                    trjctlgs[:, ANGLE_IDX] * 180.0 / np.pi,
                    alpha=1.0,
                    linestyle="-",
                    linewidth=1,
                    color="g",
                    label="realized trajectory",
                )
                # Plot target positions
                ax1.plot(
                    np.arange(0, np.shape(trgtlgs)[0]) * dt,
                    trgtlgs,
                    alpha=1.0,
                    linestyle="--",
                    linewidth=1,
                    color="k",
                    label="target position",
                )
                # Plot trajectory planned by MPPI (= nominal trajectory)
                ax1.plot(
                    (update_every * (i - 1) + np.arange(0, np.shape(nrlgs)[1])) * dt,
                    nrlgs[i - 1, :, POSITION_IDX],
                    alpha=1.0,
                    linestyle="-",
                    linewidth=1,
                    color="r",
                    label="nominal trajectory\n(under trained model)",
                )
                ax2.plot(
                    (update_every * (i - 1) + np.arange(0, np.shape(nrlgs)[1])) * dt,
                    nrlgs[i - 1, :, ANGLE_IDX] * 180.0 / np.pi,
                    alpha=1.0,
                    linestyle="-",
                    linewidth=1,
                    color="r",
                    label="nominal trajectory\n(under trained model)",
                )
                # Plot the trajectory of rollout with cost-averaged nominal inputs if model were ideal
                ax1.plot(
                    (
                        update_every * (i - 1)
                        + np.arange(0, np.shape(true_nominal_rollouts)[1])
                    )
                    * dt,
                    true_nominal_rollouts[i - 1, :, POSITION_IDX],
                    alpha=1.0,
                    linestyle="--",
                    linewidth=1,
                    color="r",
                    label="nominal trajectory\n(under true model)",
                )
                ax2.plot(
                    (
                        update_every * (i - 1)
                        + np.arange(0, np.shape(true_nominal_rollouts)[1])
                    )
                    * dt,
                    true_nominal_rollouts[i - 1, :, ANGLE_IDX] * 180.0 / np.pi,
                    alpha=1.0,
                    linestyle="--",
                    linewidth=1,
                    color="r",
                    label="nominal trajectory\n(under true model)",
                )
                # Set axis limits
                ax1.set_xlim(0, np.shape(trjctlgs)[0] * dt)
                ax1.set_ylim(-TrackHalfLength * 1.05, TrackHalfLength * 1.05)
                ax2.set_ylim(-180.0, 180.0)

                # Set axis labels
                ax1.set_ylabel("position (m)")
                ax2.set_ylabel("angle (deg)")
                ax2.set_xlabel("time (s)", loc="right")
                ax1.set_title("Monte Carlo Rollouts")

                # Set axis legends
                ax1.legend(
                    loc="upper left", fontsize=12, bbox_to_anchor=(1, 0, 0.16, 1)
                )
                ax2.legend(
                    loc="upper left", fontsize=12, bbox_to_anchor=(1, 0, 0.16, 1)
                )

            # Draw first iteration
            update_plot(1)

            # Update plot on slider click
            slider.on_changed(update_plot)

            # Show plot
            plt.show()

    # Optionally: reset the controller after an experiment
    # May be useful for stateful controllers, like these containing RNN,
    # To reload the hidden states e.g. if the controller went unstable in the previous run.
    # It is called after an experiment,
    # but only if the controller is supposed to be reused without reloading (e.g. in GUI)
    def controller_reset(self):
        try:
            self.warm_up_countdown = self.warm_up_len
            # TODO: Not sure if this works for predictor autoregressive tf
            predictor.net.reset_states()
        except:
            pass
