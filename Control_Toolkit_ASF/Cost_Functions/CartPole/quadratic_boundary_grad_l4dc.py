"""L4DC 2025 A-NC teacher cost for pole length and pole mass variation.

This is the 16 Nov 2024 `quadratic_boundary_grad` (commit 5c94adde), kept as its
own class so today's up/down / energy-shaped cost is not overwritten.

Important: the 2024 yaml key `dd_weight` was implemented as *linear*
|position - target| / (2 L_track), not as a squared distance.
`ekp` is angleD**2. Potential uses the 2024 cosine formula.
"""

import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from others.globals_and_utils import load_config

from CartPole.cartpole_parameters import TrackHalfLength, u_max
from CartPole.state_utilities import ANGLE_IDX, ANGLED_IDX, POSITION_IDX


class quadratic_boundary_grad_l4dc(cost_function_base):

    def __init__(self, variable_parameters, lib):
        super().__init__(variable_parameters, lib)
        config, config_path = load_config(
            os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"),
            return_path=True,
        )
        self.config = config["CartPole"]["quadratic_boundary_grad_l4dc"]
        self.config_path = config_path

        print("\nConfig cost function quadratic_boundary_grad_l4dc (pole L & m teacher):")
        for key, value in self.config.items():
            if key == "admissible_angle":
                setattr(self, key, self.lib.to_variable(self.lib.pi * value / 180.0, self.lib.float32))
            else:
                setattr(self, key, self.lib.to_variable(value, self.lib.float32))
            print("{}: {}".format(key, value))
        print()

        self.MAX_COST = (
            float(self.config["dd_weight"]) * 1.0e7
            + float(self.config["ep_weight"])
            + float(self.config["ekp_weight"]) * 25.0
            + float(self.config["cc_weight"]) * float(self.config["R"]) * (u_max ** 2)
            + float(self.config["ccrc_weight"]) * 4 * (u_max ** 2)
        )

    def reload_cost_parameters_from_config(self):
        for key, value in self.config.items():
            var = getattr(self, key)
            if key == "admissible_angle":
                self.lib.assign(var, self.lib.to_variable(self.lib.pi * value / 180.0, self.lib.float32))
            else:
                self.lib.assign(var, self.lib.to_variable(value, self.lib.float32))

    def _distance_difference_cost(self, position):
        """Linear |x - x_target| / (2 TrackHalfLength). This is what 2024 called dd_weight."""
        return self.lib.abs(
            (position - self.variable_parameters.target_position) / (2.0 * TrackHalfLength)
        )

    def _boundary_approach_cost(self, position):
        too_near_to_the_boundary = self.lib.cast(
            self.lib.abs(position) > self.permissible_track_fraction * TrackHalfLength,
            self.lib.float32,
        )
        return too_near_to_the_boundary * (
            (self.lib.abs(position) - self.permissible_track_fraction * TrackHalfLength)
            / ((1 - self.permissible_track_fraction) * TrackHalfLength)
        ) ** 2

    def _E_pot_cost(self, angle):
        return 0.25 * (
            2.0
            - self.lib.cos(
                angle + (1.0 - self.variable_parameters.target_equilibrium) * self.lib.pi / 2.0
            )
        ) ** 2

    def _E_kin_cost(self, angleD):
        return angleD ** 2

    def _CC_cost(self, u):
        return self.R * self.lib.sum(u ** 2, 2)

    def get_terminal_cost(self, terminal_states: TensorType):
        terminal_cost = self.lib.zeros_like(terminal_states[:, ANGLE_IDX])
        return self.lib.reshape(terminal_cost, (-1, 1))

    def _control_change_rate_cost(self, u, u_prev):
        u_prev_vec = self.lib.concat(
            (self.lib.ones((u.shape[0], 1, u.shape[2])) * u_prev, u[:, :-1, :]), 1
        )
        return self.lib.sum((u - u_prev_vec) ** 2, 2)

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):
        dd = self.dd_weight * self._distance_difference_cost(states[:, :, POSITION_IDX])
        db = self.db_weight * self._boundary_approach_cost(states[:, :, POSITION_IDX])
        ep = self.ep_weight * self._E_pot_cost(states[:, :, ANGLE_IDX])
        ekp = self.ekp_weight * self._E_kin_cost(states[:, :, ANGLED_IDX])
        cc = self.cc_weight * self._CC_cost(inputs)
        ccrc = self.ccrc_weight * self._control_change_rate_cost(inputs, previous_input)
        return dd + db + ep + ekp + cc + ccrc
