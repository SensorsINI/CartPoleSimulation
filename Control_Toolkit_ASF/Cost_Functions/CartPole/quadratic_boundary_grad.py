import os
from yaml import safe_load

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base

from others.globals_and_utils import load_config

from CartPole.cartpole_parameters import TrackHalfLength, u_max
from CartPole.state_utilities import ANGLE_IDX, ANGLED_IDX, POSITION_IDX

import numpy as np




class quadratic_boundary_grad(cost_function_base):

    def __init__(self, variable_parameters, lib):
        super().__init__(variable_parameters, lib)
        # load constants from config file - these are quite ugly two lines which needs to be manually changed for each cost function
        config, config_path = load_config(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"),
                                          return_path=True)
        self.config = config["CartPole"]["quadratic_boundary_grad"]

        self.config_path = config_path

        print('\nConfig cost function:')
        for key, value in self.config.items():
            if key == "admissible_angle":
                setattr(self, key, self.lib.to_variable(self.lib.pi*value/180.0, self.lib.float32))
            else:
                setattr(self, key, self.lib.to_variable(value, self.lib.float32))
            print('{}: {}'.format(key, value))
        print()



    def reload_cost_parameters_from_config(self):
        for key, value in self.config.items():
            var = getattr(self, key)
            if key == "admissible_angle":
                self.lib.assign(var, self.lib.to_variable(self.lib.pi*value/180.0, self.lib.float32))
            else:
                self.lib.assign(var, self.lib.to_variable(value, self.lib.float32))

    # cost for distance from track edge
    def _distance_difference_cost_quadratic(self, position):
        """Compute penalty for distance of cart to the target position"""

        target_distance_cost = (
            (position - self.variable_parameters.target_position) / (2*TrackHalfLength)) ** 2

        return target_distance_cost

    def _distance_difference_cost_linear(self, position):
        """Compute penalty for distance of cart to the target position"""
        target_distance_cost = self.lib.abs(
            (position - self.variable_parameters.target_position) / (2.0 * TrackHalfLength))

        return target_distance_cost

    def _boundary_approach_cost(self, position):
        """Compute penalty for approaching the boundary"""
        # Soft constraint: Do not crash into border
        too_near_to_the_boundary = self.lib.cast(self.lib.abs(position) > self.permissible_track_fraction * TrackHalfLength, self.lib.float32)
        return too_near_to_the_boundary * (
                (self.lib.abs(position) - self.permissible_track_fraction * TrackHalfLength) / (
                    (1 - self.permissible_track_fraction) * TrackHalfLength)
        ) ** 2

    # cost for difference from upright position
    def _E_pot_cost(self, angle):
        """Compute penalty for not balancing pole upright (penalize large angles)"""
        return 0.25 * (2.0 - self.lib.cos(angle + (1.0-self.variable_parameters.target_equilibrium)*self.lib.pi/2.0)) ** 2

    def _E_kin_cost(self, angleD):
        """Compute penalty for not balancing pole upright (penalize large angles)"""
        return angleD ** 2

    # actuation cost
    def _CC_cost(self, u):
        return self.R * self.lib.sum(u**2, 2)

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
        # terminal_cost = 10000 * self.lib.cast(
        #     (self.lib.abs(terminal_states[:, ANGLE_IDX]) > 0.2)
        #     | (
        #         self.lib.abs(terminal_states[:, POSITION_IDX] - self.variable_parameters.target_position)
        #         > 0.1 * TrackHalfLength
        #     ),
        #     self.lib.float32,
        # )
        terminal_cost = self.lib.zeros_like(terminal_states[:, ANGLE_IDX])
        return self.lib.reshape(terminal_cost, (-1, 1))

    # cost of changing control to fast
    def _control_change_rate_cost(self, u, u_prev):
        """Compute penalty of control jerk, i.e. difference to previous control input"""
        u_prev_vec = self.lib.concat(
            (self.lib.ones((u.shape[0], 1, u.shape[2])) * u_prev, u[:, :-1, :]), 1
        )
        return self.lib.sum((u - u_prev_vec) ** 2, 2)

    # all stage costs together
    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):
        dd_quadratic = self.dd_quadratic_weight * self._distance_difference_cost_quadratic(
            states[:, :, POSITION_IDX]
        )
        dd_linear = self.dd_linear_weight * self._distance_difference_cost_linear(
            states[:, :, POSITION_IDX]
        )
        db = self.db_weight * self._boundary_approach_cost(states[:, :, POSITION_IDX])
        ep = self.ep_weight * self._E_pot_cost(states[:, :, ANGLE_IDX])
        ekp = self.ekp_weight * self._E_kin_cost(states[:, :, ANGLED_IDX])
        cc = self.cc_weight * self._CC_cost(inputs)
        ccrc = self.ccrc_weight * self._control_change_rate_cost(inputs, previous_input)
        stage_cost = dd_linear + dd_quadratic + db + ep + ekp + cc + ccrc
        return stage_cost

    def q_debug(self, s, u, u_prev):
        dd_quadratic = self.dd_quadratic_weight * self._distance_difference_cost_quadratic(
            s[:, :, POSITION_IDX]
        )
        ep = self.ep_weight * self._E_pot_cost(s[:, :, ANGLE_IDX])
        cc = self.cc_weight * self._CC_cost(u)
        ccrc = 0
        if u_prev is not None:
            ccrc = self.ccrc_weight * self._control_change_rate_cost(u, u_prev)
        stage_cost = dd_quadratic + ep + cc + ccrc
        return stage_cost, dd_quadratic, ep, cc, ccrc
