import os
from yaml import safe_load

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base

from others.globals_and_utils import load_config

from CartPole.cartpole_parameters import TrackHalfLength, u_max
from CartPole.state_utilities import ANGLE_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX

import numpy as np




class quadratic_boundary_grad_minimal(cost_function_base):

    def __init__(self, variable_parameters, lib):
        super().__init__(variable_parameters, lib)
        # load constants from config file - these are quite ugly two lines which needs to be manually changed for each cost function
        config, config_path = load_config(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"),
                                          return_path=True)
        self.config = config["CartPole"]["quadratic_boundary_grad"]

        self.config_path = config_path

        self.target_angular_speed_sqr_max_correction = self.lib.to_variable(0.0, self.lib.float32)

        print('\nConfig cost function:')
        for key, value in self.config.items():
            if key == "admissible_angle":
                setattr(self, key, self.lib.to_variable(self.lib.pi*value/180.0, self.lib.float32))
            else:
                setattr(self, key, self.lib.to_variable(value, self.lib.float32))
            print('{}: {}'.format(key, value))
        print()

        self.stage_cost, self.dd_linear, self.dd_quadratic, self.db, self.ep, self.ekp, self.cc, self.ccrc = [
            self.lib.to_variable((0.0,), self.lib.float32) for _ in range(8)
        ]
        self.set_logged_attributes({
            "cost_component_total_stage_cost": lambda: float(self.stage_cost),
            "cost_component_dd_liear": lambda: float(self.dd_linear),
            "cost_component_dd_quadratic": lambda: float(self.dd_quadratic),
            "cost_component_db": lambda: float(self.db),
            "cost_component_ep": lambda: float(self.ep),
            "cost_component_ekp": lambda: float(self.ekp),
            "cost_component_cc": lambda: float(self.cc),
            "cost_component_ccrc": lambda: float(self.ccrc),
        })



    def reload_cost_parameters_from_config(self):
        for key, value in self.config.items():
            var = getattr(self, key)
            if key == "admissible_angle":
                self.lib.assign(var, self.lib.to_variable(self.lib.pi*value/180.0, self.lib.float32))
            else:
                self.lib.assign(var, self.lib.to_variable(value, self.lib.float32))

    # cost for distance from track edge
    def _distance_difference_cost_quadratic(self, position, target_position):
        """Compute penalty for distance of cart to the target position"""

        target_distance_cost = (
            (position - target_position) / (2*TrackHalfLength)) ** 2

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
        return (1.0 - self.variable_parameters.target_equilibrium*self.lib.cos(angle)) ** 2

    def _E_kin_cost(self, angleD):
        """Compute penalty for not balancing pole upright (penalize large angles)"""
        return angleD ** 2

    def _CC_cost(self, u):
        return self.R * self.lib.sum(u**2, 2)

    # final stage cost
    def get_terminal_cost(self, terminal_states: TensorType):
        terminal_cost = self.lib.zeros_like(terminal_states[:, ANGLE_IDX])
        return self.lib.reshape(terminal_cost, (-1, 1))

    def weights(self):
        return (self.dd_quadratic_weight_up, self.db_weight_up,
                self.ep_weight_up, self.ekp_weight_up,
                self.cc_weight_up,
                )

    def stage_cost_components(self, states: TensorType, inputs: TensorType, previous_input: TensorType):


        dd_quadratic_weight, db_weight, ep_weight, ekp_weight, cc_weight = self.weights()

        dd_quadratic = dd_quadratic_weight * self._distance_difference_cost_quadratic(
            states[:, :, POSITION_IDX],
            self.variable_parameters.target_position
        )

        db = db_weight * self._boundary_approach_cost(states[:, :, POSITION_IDX])
        ep = ep_weight * self._E_pot_cost(states[:, :, ANGLE_IDX])
        ekp = ekp_weight * self._E_kin_cost(states[:, :, ANGLED_IDX])

        cc = cc_weight * self._CC_cost(inputs)

        return dd_quadratic, db, ep, ekp, cc

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):
        dd_quadratic, db, ep, ekp, cc = self.stage_cost_components(states, inputs, previous_input)
        stage_cost =  dd_quadratic + db + ep + ekp + cc
        return stage_cost

    def get_summed_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):

        dd_quadratic, db, ep, ekp, cc = self.stage_cost_components(states, inputs, previous_input)

        self.lib.assign(self.dd_quadratic, self.lib.sum(dd_quadratic, 1))
        self.lib.assign(self.db, self.lib.sum(db, 1))
        self.lib.assign(self.ep, self.lib.sum(ep, 1))
        self.lib.assign(self.ekp, self.lib.sum(ekp, 1))
        self.lib.assign(self.cc, self.lib.sum(cc, 1))

        self.lib.assign(self.stage_cost, self.dd_quadratic + self.db + self.ep + self.ekp + self.cc)
        return self.stage_cost

