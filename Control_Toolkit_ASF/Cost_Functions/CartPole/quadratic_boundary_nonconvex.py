import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base

from others.globals_and_utils import load_config

from CartPole.cartpole_model import TrackHalfLength
from CartPole.state_utilities import ANGLE_IDX, POSITION_IDX

# load constants from config file
config = load_config(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"))

dd_weight = config["CartPole"]["quadratic_boundary_nonconvex"]["dd_weight"]
cc_weight = config["CartPole"]["quadratic_boundary_nonconvex"]["cc_weight"]
ep_weight = config["CartPole"]["quadratic_boundary_nonconvex"]["ep_weight"]
R = config["CartPole"]["quadratic_boundary_nonconvex"]["R"]

ccrc_weight = config["CartPole"]["quadratic_boundary_nonconvex"]["cem_ccrc_weight"]


class quadratic_boundary_nonconvex(cost_function_base):
    # cost for distance from track edge
    def _distance_difference_cost(self, position):
        """Compute penalty for distance of cart to the target position"""
        return (
            ((position - self.controller.target_position) / (2.0 * TrackHalfLength)) ** 2
            - 0.15
            * (
                self.lib.cos(
                    4
                    * 2
                    * self.lib.pi
                    * (position - self.controller.target_position)
                    / (2.0 * TrackHalfLength)
                )
                - 1.0
            )
            + self.lib.cast(self.lib.abs(position) > 0.95 * TrackHalfLength, self.lib.float32)
            * 1e9
            * ((self.lib.abs(position) - 0.95 * TrackHalfLength) / (0.05 * TrackHalfLength))
            ** 2
        )  # Soft constraint: Do not crash into border

    # cost for difference from upright position
    def _E_pot_cost(self, angle):
        """Compute penalty for not balancing pole upright (penalize large angles)"""
        return self.controller.target_equilibrium * 0.25 * (1.0 - self.lib.cos(angle)) ** 2

    # actuation cost
    def _CC_cost(self, u):
        return R * self.lib.sum(u**2, 2)

    # final stage cost
    def get_terminal_cost(self, terminal_states: TensorType):
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
        terminal_cost = 10000 * self.lib.cast(
            (self.lib.abs(terminal_states[:, ANGLE_IDX]) > 0.2)
            | (
                self.lib.abs(terminal_states[:, POSITION_IDX] - self.controller.target_position)
                > 0.1 * TrackHalfLength
            ),
            self.lib.float32,
        )
        return terminal_cost

    # cost of changing control to fast
    def _control_change_rate_cost(self, u, u_prev):
        """Compute penalty of control jerk, i.e. difference to previous control input"""
        u_prev_vec = self.lib.concat(
            (self.lib.ones((u.shape[0], 1, u.shape[2])) * u_prev, u[:, :-1, :]), 1
        )
        return self.lib.sum((u - u_prev_vec) ** 2, 2)

    # all stage costs together
    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):
        dd = dd_weight * self._distance_difference_cost(states[:, :, POSITION_IDX])
        ep = ep_weight * self._E_pot_cost(states[:, :, ANGLE_IDX])
        cc = cc_weight * self._CC_cost(inputs)
        ccrc = 0
        if previous_input is not None:
            ccrc = ccrc_weight * self._control_change_rate_cost(inputs, previous_input)
        stage_cost = dd + ep + cc + ccrc
        return stage_cost
