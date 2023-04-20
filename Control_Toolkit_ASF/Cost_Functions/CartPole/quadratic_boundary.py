import os
from yaml import safe_load

from SI_Toolkit.computation_library import TensorType

from Control_Toolkit.Cost_Functions import cost_function_base
from others.globals_and_utils import load_config

from CartPole.cartpole_model import TrackHalfLength
from CartPole.state_utilities import ANGLE_IDX, POSITION_IDX
from CartPole.cartpole_model import u_max

# load constants from config file
config = safe_load(open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"))

dd_weight = config["CartPole"]["quadratic_boundary"]["dd_weight"]
cc_weight = config["CartPole"]["quadratic_boundary"]["cc_weight"]
ep_weight = config["CartPole"]["quadratic_boundary"]["ep_weight"]
R = config["CartPole"]["quadratic_boundary"]["R"]
ccrc_weight = config["CartPole"]["quadratic_boundary"]["ccrc_weight"]


class quadratic_boundary(cost_function_base):
    MAX_COST = dd_weight * 1.0e7 + ep_weight + cc_weight * R * (u_max ** 2) + ccrc_weight * 4 * (u_max ** 2)
    
    # cost for distance from track edge
    def _distance_difference_cost(self, position):
        """Compute penalty for distance of cart to the target position"""
        return (
            (position - self.variable_parameters.target_position) / (2.0 * TrackHalfLength)
        ) ** 2 + self.lib.cast(
            self.lib.abs(position) > 0.95 * TrackHalfLength, self.lib.float32
        ) * 1e9 * (
            (self.lib.abs(position) - 0.95 * TrackHalfLength) / (0.05 * TrackHalfLength)
        ) ** 2  # Soft constraint: Do not crash into border

    # cost for difference from upright position
    def _E_pot_cost(self, angle):
        """Compute penalty for not balancing pole upright (penalize large angles)"""
        return self.variable_parameters.target_equilibrium * 0.25 * (1.0 - self.lib.cos(angle)) ** 2

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

        :param terminal_states: Reference to numpy array of terminal states of all rollouts
        :type terminal_states: np.ndarray
        :return: One terminal cost per rollout
        :rtype: np.ndarray
        """
        terminal_cost = 10000 * self.lib.cast(
            (self.lib.abs(terminal_states[:, ANGLE_IDX]) > 0.2)
            | (
                self.lib.abs(terminal_states[:, POSITION_IDX] - self.variable_parameters.target_position)
                > 0.1 * TrackHalfLength
            ),
            self.lib.float32,
        )
        return self.lib.reshape(terminal_cost, (-1, 1))

    # cost of changing control to fast
    def _control_change_rate_cost(self, u, u_prev):
        """Compute penalty of control jerk, i.e. difference to previous control input"""
        u_prev_vec = self.lib.concat(
            (self.lib.ones((u.shape[0], 1, u.shape[2])) * u_prev, u[:, :-1, :]), 1
        )
        return self.lib.sum((u - u_prev_vec) ** 2, 2)

    # all stage costs together
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):
        dd = dd_weight * self._distance_difference_cost(states[:, :, POSITION_IDX])
        ep = ep_weight * self._E_pot_cost(states[:, :, ANGLE_IDX])
        cc = cc_weight * self._CC_cost(inputs)
        ccrc = 0
        if previous_input is not None:
            ccrc = ccrc_weight * self._control_change_rate_cost(inputs, previous_input)
        stage_cost = dd + ep + cc + ccrc
        return stage_cost
