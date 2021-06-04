"""
Model Predictive Path Integral Controller
Based on Williams, Aldrich, Theodorou (2015)
"""

# Uncomment if you want to get interactive plots for MPPI in Pycharm on MacOS
# On other OS you have to chose a different interactive backend.
# from matplotlib import use
# # # use('TkAgg')
# use('macOSX')

from Controllers.template_controller import template_controller
from CartPole.cartpole_model import TrackHalfLength
from CartPole.state_utilities import (
    create_cartpole_state,
    cartpole_state_varname_to_index,
    ANGLE_IDX,
    ANGLED_IDX,
    POSITION_IDX,
    POSITIOND_IDX,
)

from CartPole._CartPole_mathematical_helpers import (
    conditional_decorator,
    wrap_angle_rad_inplace,
)

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from numba import jit
import numpy as np
from numpy.random import SFC64, Generator
from scipy.interpolate import interp1d

from others.globals_and_utils import Timer
from SI_Toolkit.TF.TF_Functions.predictor_autoregressive_tf import (
    predictor_autoregressive_tf,
)
from Predictores.predictor_ideal import predictor_ideal

import yaml

config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

"""Timestep and sampling settings"""
dt = config["controller"]["mppi"]["dt"]
mpc_horizon = config["controller"]["mppi"]["mpc_horizon"]
mpc_samples = int(mpc_horizon / dt)  # Number of steps in MPC horizon
mc_samples = config["controller"]["mppi"]["mc_samples"]
update_every = config["controller"]["mppi"]["update_every"]
predictor_type = config["controller"]["mppi"]["predictor_type"]


"""Parameters weighting the different cost components"""
dd_weight = config["controller"]["mppi"]["dd_weight"]
ep_weight = config["controller"]["mppi"]["ep_weight"]
ekp_weight = config["controller"]["mppi"]["ekp_weight"]
ekc_weight = config["controller"]["mppi"]["ekc_weight"]
cc_weight = config["controller"]["mppi"]["cc_weight"]
ccrc_weight = config["controller"]["mppi"]["ccrc_weight"]


gui_dd = gui_ep = gui_ekp = gui_ekc = gui_cc = gui_ccrc = np.zeros(1, dtype=np.float32)


"""MPPI constants"""
R = config["controller"]["mppi"]["R"]
LBD = config["controller"]["mppi"]["LBD"]
NU = config["controller"]["mppi"]["NU"]
GAMMA = config["controller"]["mppi"]["GAMMA"]
SAMPLING_TYPE = config["controller"]["mppi"]["SAMPLING_TYPE"]

"""Random number generator"""
rng = Generator(SFC64(123))


"""Init logging variables"""
LOGGING = config["controller"]["mppi"]["LOGGING"]
# Save average cost for each cost component
COST_TO_GO_LOGS = []
COST_BREAKDOWN_LOGS = []
STATE_LOGS = []
TRAJECTORY_LOGS = []
TARGET_TRAJECTORY_LOGS = []
INPUT_LOGS = []
NOMINAL_ROLLOUT_LOGS = []


"""Cost function helpers"""


@jit(nopython=True, cache=True, fastmath=True)
def E_kin_cart(positionD):
    return positionD ** 2


@jit(nopython=True, cache=True, fastmath=True)
def E_kin_pol(angleD):
    return angleD ** 2


@jit(nopython=True, cache=True, fastmath=True)
def E_pot_cost(angle):
    return 0.25 * (1.0 - np.cos(angle)) ** 2
    # return angle ** 2


@jit(nopython=True, cache=True, fastmath=True)
def distance_difference_cost(position, target_position):
    return ((position - target_position) / (2.0 * TrackHalfLength)) ** 2 + (
        TrackHalfLength - np.abs(position) < 0.005 * TrackHalfLength
    ) * 1.0e3  # Hard constraint: Do not crash into border


@jit(nopython=True, cache=True, fastmath=True)
def control_change_rate_cost(u, u_prev):
    return (u - u_prev) ** 2


@jit(nopython=True, cache=True, fastmath=True)
def penalize_deviation(cc, u):
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
    predictor = predictor_autoregressive_tf(horizon=mpc_samples, batch_size=mc_samples)


def trajectory_rollouts(
    s: np.ndarray,
    S_tilde_k: np.ndarray,
    u: np.ndarray,
    delta_u: np.ndarray,
    u_prev: np.ndarray,
    target_position: np.ndarray,
):
    s_horizon = np.zeros((mc_samples, mpc_samples + 1, s.size), dtype=np.float32)
    s_horizon[:, 0, :] = np.tile(s, (mc_samples, 1))

    predictor.setup(initial_state=s_horizon[:, 0, :], prediction_denorm=True)
    s_horizon = predictor.predict(u + delta_u)

    cost_increment, dd, ep, ekp, ekc, cc, ccrc = q(
        s_horizon[:, 1:, :], u, delta_u, u_prev, target_position
    )
    S_tilde_k = np.sum(cost_increment, axis=1)

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
        cost_logs_internal = np.swapaxes(np.array([dd, ep, ekp, ekc, cc, ccrc]), 0, 1)
        # (mc_samples x 6 x mpc_samples)

        return S_tilde_k, cost_logs_internal, s_horizon[:, :-1, :]
    return S_tilde_k, None, None


def q(s, u, delta_u, u_prev, target_position):
    """Cost function per iteration"""
    # TODO: "weight"
    dd = dd_weight * distance_difference_cost(
        s[:, :, POSITION_IDX], target_position
    ).astype(np.float32)
    ep = ep_weight * E_pot_cost(s[:, :, ANGLE_IDX]).astype(
        np.float32
    )  # Frederik had 1.0e3
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
def reward_weighted_average(S, delta_u):
    """Average the perturbations delta_u based on their desirability"""
    rho = np.min(S)  # for numerical stability
    exp_s = np.exp(-1.0 / LBD * (S - rho))
    a = np.sum(exp_s)
    b = np.sum(np.multiply(np.expand_dims(exp_s, 1), delta_u) / a, axis=0)
    return b


@jit(nopython=True, cache=True, fastmath=True)
def update_inputs(u: np.ndarray, S: np.ndarray, delta_u: np.ndarray):
    """
    :param u: Sampling mean / warm started control inputs of size (,mpc_samples)
    :param S: Cost array of size (mc_samples)
    :param delta_u: The input perturbations that had been used, size (mc_samples, mpc_samples)
    """
    u += reward_weighted_average(S, delta_u)


class controller_mppi(template_controller):
    def __init__(self):
        # State of the cart
        self.s = create_cartpole_state()

        self.target_position = 0.0

        self.rho_sqrt_inv = 0.01

        self.iteration = -1

        self.s_horizon = np.zeros((), dtype=np.float32)
        self.u = np.zeros((mpc_samples), dtype=np.float32)
        self.u_prev = np.zeros_like(self.u, dtype=np.float32)
        self.delta_u = np.zeros((mc_samples, mpc_samples), dtype=np.float32)
        self.S_tilde_k = np.zeros((mc_samples), dtype=np.float32)

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
        """
        Return a numpy array with the perturbations delta_u.
        If random_walk is false, initialize with independent Gaussian samples
        If random_walk is true, each row represents a 1D random walk with Gaussian steps.
        """
        if sampling_type == "random_walk":
            delta_u = np.empty((mc_samples, mpc_samples), dtype=np.float32)
            delta_u[:, 0] = stdev * rng.standard_normal(
                size=(mc_samples,), dtype=np.float32
            )
            for i in range(1, mpc_samples):
                delta_u[:, i] = delta_u[:, i - 1] + stdev * rng.standard_normal(
                    size=(mc_samples,), dtype=np.float32
                )
        elif sampling_type == "uniform":
            delta_u = np.empty((mc_samples, mpc_samples), dtype=np.float32)
            for i in range(0, mpc_samples):
                delta_u[:, i] = rng.uniform(
                    low=-1.0, high=1.0, size=(mc_samples,)
                ).astype(np.float32)
        elif sampling_type == "repeated":
            delta_u = np.tile(
                stdev * rng.standard_normal(size=(mc_samples, 1), dtype=np.float32),
                (1, mpc_samples),
            )
        elif sampling_type == "interpolated":
            step = 10
            range_stop = int(np.ceil((mpc_samples) / step) * step) + 1
            t = np.arange(start=0, stop=range_stop, step=step)
            t_interp = np.arange(start=0, stop=range_stop, step=1)
            t_interp = np.delete(t_interp, t)
            delta_u = np.zeros(shape=(mc_samples, range_stop), dtype=np.float32)
            delta_u[:, t] = stdev * rng.standard_normal(
                size=(mc_samples, t.size), dtype=np.float32
            )
            f = interp1d(t, delta_u[:, t])
            delta_u[:, t_interp] = f(t_interp)
            delta_u = delta_u[:, :mpc_samples]
        else:
            delta_u = stdev * rng.standard_normal(
                size=(mc_samples, mpc_samples), dtype=np.float32
            )

        return delta_u

    def step(self, s, target_position, time=None):
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
                stdev=NU * 1e-4,
                sampling_type=SAMPLING_TYPE,
            )  # du ~ N(mean=0, var=1/(rho*dt))
            self.S_tilde_k = np.zeros_like(self.S_tilde_k, dtype=np.float32)

            # Run parallel trajectory rollouts for different input perturbations
            self.S_tilde_k, cost_logs_internal, s_horizon = trajectory_rollouts(
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
                COST_TO_GO_LOGS.append(np.copy(self.S_tilde_k))
                COST_BREAKDOWN_LOGS.append(np.copy(np.mean(cost_logs_internal, axis=0)))
                STATE_LOGS.append(s_horizon[:, :, [POSITION_IDX, ANGLE_IDX]])
                INPUT_LOGS.append(np.copy(self.u))
                # Simulate nominal rollout to plot the trajectory the controller wants to make
                # Compute one rollout of shape (mpc_samples + 1) x s.size
                if predictor_type == "Euler":
                    predictor.setup(
                        initial_state=np.copy(self.s), prediction_denorm=True
                    )
                    rollout_trajectory = predictor.predict(self.u)
                elif predictor_type == "NeuralNet":
                    predictor.setup(
                        initial_state=np.tile(self.s, (mc_samples, 1)), prediction_denorm=True
                    )
                    # This is a lot of unnecessary calculation, but a stateful RNN in TF has frozen batch size
                    rollout_trajectory = predictor.predict(
                        np.tile(self.u, (mc_samples, 1))
                    )[0, ...]
                NOMINAL_ROLLOUT_LOGS.append(np.copy(rollout_trajectory[:-1, :8]))

        if LOGGING:
            TRAJECTORY_LOGS.append(np.copy(self.s[[POSITION_IDX, ANGLE_IDX]]))
            TARGET_TRAJECTORY_LOGS.append(np.copy(target_position))

        if self.warm_up_countdown > 0 and self.auxiliary_controller_available:
            self.warm_up_countdown -= 1
            Q = self.auxiliary_controller.step(s, target_position)
        else:
            Q = self.u[0]

        # Clip inputs to allowed range
        Q = np.clip(Q, -1.0, 1.0)

        # Preserve current series of inputs
        self.u_prev = np.copy(self.u)

        # Index-shift inputs
        self.u[:-1] = self.u[1:]
        self.u[-1] = 0
        # self.u = zeros_like(self.u)

        # Prepare predictor for next timestep
        Q_update = np.tile(Q, (mc_samples, 1))
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
            ctglgs = np.stack(COST_TO_GO_LOGS, axis=0)  # ITERATIONS x mc_samples
            time_axis = update_every * dt * np.arange(start=0, stop=np.shape(ctglgs)[0])
            plt.figure(num=2, figsize=(16, 9))
            plt.plot(time_axis, np.mean(ctglgs, axis=1))
            plt.ylabel("avg_cost")
            plt.xlabel("time (s)")
            plt.title("Cost-to-go per Timestep")
            plt.show()

            ### Graph the different cost components per iteration
            clgs = np.stack(COST_BREAKDOWN_LOGS, axis=0)  # ITERATIONS x 6 x mpc_samples
            time_axis = update_every * dt * np.arange(start=0, stop=np.shape(clgs)[0])

            plt.figure(num=3, figsize=(16, 9))
            plt.plot(
                time_axis,
                np.sum(clgs[:, 0, :], axis=-1),
                label="Distance difference cost",
            )
            plt.plot(time_axis, np.sum(clgs[:, 1, :], axis=-1), label="E_pot cost")
            plt.plot(time_axis, np.sum(clgs[:, 2, :], axis=-1), label="E_kin_pole cost")
            plt.plot(time_axis, np.sum(clgs[:, 3, :], axis=-1), label="E_kin_cart cost")
            plt.plot(time_axis, np.sum(clgs[:, 4, :], axis=-1), label="Control cost")
            plt.plot(
                time_axis,
                np.sum(clgs[:, 5, :], axis=-1),
                label="Control change rate cost",
            )

            plt.ylabel("total horizon cost")
            plt.xlabel("time (s)")
            plt.title("Cost component breakdown")
            plt.legend()
            plt.show()

            ### Draw the trajectory rollouts simulated by MPPI
            def draw_rollouts(
                states: np.ndarray,
                ax_position: plt.Axes,
                ax_angle: plt.Axes,
                costs: np.ndarray,
                iteration: int,
            ):
                mc_rollouts = np.shape(states)[0]
                horizon_length = np.shape(states)[1]
                # Loop over all MC rollouts
                for i in range(mc_rollouts):
                    ax_position.plot(
                        (update_every * iteration + np.arange(0, horizon_length)) * dt,
                        states[i, :, 0],
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
                        states[i, :, 1] * 180.0 / np.pi,
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
            # shape(slgs) = ITERATIONS x mc_samples x mpc_samples x [position, angle]
            slgs = np.stack(STATE_LOGS, axis=0)
            wrap_angle_rad_inplace(slgs[:, :, :, 1])
            # shape(iplgs) = ITERATIONS x mpc_horizon
            iplgs = np.stack(INPUT_LOGS, axis=0)
            # shape(nrlgs) = ITERATIONS x mpc_horizon x [position, angle]
            nrlgs = np.stack(NOMINAL_ROLLOUT_LOGS, axis=0)[
                :, :, [POSITION_IDX, ANGLE_IDX]
            ]
            wrap_angle_rad_inplace(nrlgs[:, :, 1])
            # shape(trjctlgs) = (update_every * ITERATIONS) x [position, angle]
            trjctlgs = np.stack(TRAJECTORY_LOGS[:-1], axis=0)
            wrap_angle_rad_inplace(trjctlgs[:, 1])
            # shape(trgtlgs) = ITERATIONS x [position]
            trgtlgs = np.stack(TARGET_TRAJECTORY_LOGS[:-1], axis=0)
            # For each rollout, calculate what the nominal trajectory would be using the known true model
            # This can uncover if the model used makes inaccurate predictions
            # shape(true_nominal_rollouts) = ITERATIONS x mpc_horizon x [position, positionD, angle, angleD]
            predictor_true_equations = predictor_ideal(horizon=mpc_samples, dt=dt, intermediate_steps=10)
            true_nominal_rollouts = np.stack(NOMINAL_ROLLOUT_LOGS, axis=0)
            wrap_angle_rad_inplace(true_nominal_rollouts[:, :, ANGLE_IDX])
            predictor_true_equations.setup(
                np.copy(true_nominal_rollouts[:, 0, :]), prediction_denorm=True
            )
            true_nominal_rollouts = predictor_true_equations.predict(iplgs)[
                :, :-1, [POSITION_IDX, ANGLE_IDX]
            ]
            wrap_angle_rad_inplace(true_nominal_rollouts[:, :, 1])

            # Create figure
            fig, (ax1, ax2) = plt.subplots(
                nrows=2,
                ncols=1,
                num=4,
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
            # shape(ctglgs) = ITERATIONS x mc_samples
            ctglgs = np.divide(ctglgs.T, np.max(np.abs(ctglgs), axis=1)).T

            # This function updates the plot when a new iteration is selected
            def update_plot(i):
                # Clear previous iteration plot
                ax1.clear()
                ax2.clear()

                # Plot Monte Carlo rollouts
                draw_rollouts(slgs[i - 1, :, :, :], ax1, ax2, ctglgs[i - 1, :], i - 1)

                # Plot the realized trajectory
                ax1.plot(
                    np.arange(0, np.shape(trjctlgs)[0]) * dt,
                    trjctlgs[:, 0],
                    alpha=1.0,
                    linestyle="-",
                    linewidth=1,
                    color="g",
                    label="realized trajectory",
                )
                ax2.plot(
                    np.arange(0, np.shape(trjctlgs)[0]) * dt,
                    trjctlgs[:, 1] * 180.0 / np.pi,
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
                    nrlgs[i - 1, :, 0],
                    alpha=1.0,
                    linestyle="-",
                    linewidth=1,
                    color="r",
                    label="nominal trajectory\n(under trained model)",
                )
                ax2.plot(
                    (update_every * (i - 1) + np.arange(0, np.shape(nrlgs)[1])) * dt,
                    nrlgs[i - 1, :, 1] * 180.0 / np.pi,
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
                    true_nominal_rollouts[i - 1, :, 0],
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
                    true_nominal_rollouts[i - 1, :, 1] * 180.0 / np.pi,
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
                ax1.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1, 0, 0.16, 1))
                ax2.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1, 0, 0.16, 1))

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
