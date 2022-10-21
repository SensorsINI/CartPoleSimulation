import os

from Control_Toolkit.Cost_Functions import cost_function_base
from others.globals_and_utils import load_config

from CartPole.cartpole_model import TrackHalfLength
from CartPole.state_utilities import ANGLE_IDX, POSITION_IDX

# load constants from config file
config = load_config(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"))

dd_weight = config["CartPole"]["quadratic_boundary"]["dd_weight"]
cc_weight = config["CartPole"]["quadratic_boundary"]["cc_weight"]
ep_weight = config["CartPole"]["quadratic_boundary"]["ep_weight"]
R = config["CartPole"]["quadratic_boundary"]["R"]
ccrc_weight = config["CartPole"]["quadratic_boundary"]["ccrc_weight"]


class quadratic_boundary(cost_function_base):
    # cost for distance from track edge
    def distance_difference_cost(self, position):
        """Compute penalty for distance of cart to the target position"""
        return (
            (position - self.controller.target_position) / (2.0 * TrackHalfLength)
        ) ** 2 + self.lib.cast(
            self.lib.abs(position) > 0.95 * TrackHalfLength, self.lib.float32
        ) * 1e9 * (
            (self.lib.abs(position) - 0.95 * TrackHalfLength) / (0.05 * TrackHalfLength)
        ) ** 2  # Soft constraint: Do not crash into border

    # cost for difference from upright position
    def E_pot_cost(self, angle):
        """Compute penalty for not balancing pole upright (penalize large angles)"""
        return self.controller.target_equilibrium * 0.25 * (1.0 - self.lib.cos(angle)) ** 2

    # actuation cost
    def CC_cost(self, u):
        return R * self.lib.sum(u**2, 2)

    # final stage cost
    def get_terminal_cost(self, s):
        """Calculate terminal cost of a set of trajectories

        Williams et al use an indicator function type of terminal cost in
        "Information theoretic MPC for model-based reinforcement learning"

        TODO: Try a quadratic terminal cost => Use the LQR terminal cost term obtained
        by linearizing the system around the unstable equilibrium.

        :param s: Reference to numpy array of states of all rollouts
        :type s: np.ndarray
        :return: One terminal cost per rollout
        :rtype: np.ndarray
        """
        terminal_states = s[:, -1, :]
        terminal_cost = 10000 * self.lib.cast(
            (self.lib.abs(terminal_states[:, ANGLE_IDX]) > 0.2)
            | (
                self.lib.abs(terminal_states[:, POSITION_IDX] - self.controller.target_position)
                > 0.1 * TrackHalfLength
            ),
            self.lib.float32,
        )
        return terminal_cost

    # cost of changeing control to fast
    def control_change_rate_cost(self, u, u_prev):
        """Compute penalty of control jerk, i.e. difference to previous control input"""
        u_prev_vec = self.lib.concat(
            (self.lib.ones((u.shape[0], 1, u.shape[2])) * u_prev, u[:, :-1, :]), 1
        )
        return self.lib.sum((u - u_prev_vec) ** 2, 2)

    # all stage costs together
    def get_stage_cost(self, s, u, u_prev):
        dd = dd_weight * self.distance_difference_cost(s[:, :, POSITION_IDX])
        ep = ep_weight * self.E_pot_cost(s[:, :, ANGLE_IDX])
        cc = cc_weight * self.CC_cost(u)
        ccrc = 0
        if u_prev is not None:
            ccrc = ccrc_weight * self.control_change_rate_cost(u, u_prev)
        stage_cost = dd + ep + cc + ccrc
        return stage_cost
