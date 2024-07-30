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

    def _distance_difference_cost_linear(self, position, target_position):
        """Compute penalty for distance of cart to the target position"""
        target_distance_cost = self.lib.abs(
            (position - target_position) / (2.0 * TrackHalfLength))

        return target_distance_cost

    def get_target_position_efficient(self, position):
        target_position_clipped = self.lib.clip(self.variable_parameters.target_position, -self.permissible_track_fraction * TrackHalfLength, self.permissible_track_fraction * TrackHalfLength)

        # Compute the vector from position to target_position
        vector_to_target = target_position_clipped - position[:, 0]

        # Calculate the current distance to the target
        distance_to_target = self.lib.norm(vector_to_target, axis=0)

        # Determine the clipping factor
        clipping_factor = self.lib.min(1.0, self.admissible_target_distance * 2 * TrackHalfLength / distance_to_target)

        # Clip the vector to target position to make sure it does not exceed the admissible distance
        target_position_efficient = self.lib.reshape(position[:, 0] + vector_to_target * clipping_factor, (-1, 1))

        return self.lib.stop_gradient(target_position_efficient)


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
        # return 0.25 * (2.0 - self.variable_parameters.target_equilibrium*self.lib.cos(angle)) ** 2

    # def _E_kin_cost(self, angleD):
    #     """Compute penalty for not balancing pole upright (penalize large angles)"""
    #     return angleD ** 2

    def _E_kin_cost(self, angle_cos, angleD):

        te = self.variable_parameters.target_equilibrium

        # The lines below would set the target angular speed to a value
        # dictated by conservation of energy (target_angular_speed_sqr_max) for target equilibrium up
        # and to 0 to target equilibrium down
        target_angular_speed_sqr_max = 120.0
        default_target_angular_speed_sqr_max = (target_angular_speed_sqr_max*(1.0+self.variable_parameters.target_equilibrium)/2.0)

        # There is additional correction on top of this default speed
        # For equilibrium up it accounts for the very rough estimation of target_angular_speed_sqr_max
        # For equilibrium down it encourages the pole to leave the upright position
        # (which would be a minimum otherwise)
        target_angular_speed_sqr_max = self.lib.abs(default_target_angular_speed_sqr_max + self.target_angular_speed_sqr_max_correction)

        # We still have problem with this cost causing jitter while down

        basic_scaling_minus = self.lib.stop_gradient((1.0-te*angle_cos)/2)  # From energy conservation
        condition = te * (angle_cos - self.lib.cos(self.admissible_angle)) > 0  # setting to 0 small angles
        scaling = self.lib.stop_gradient(self.lib.where(condition, 0.0, basic_scaling_minus))

        target_angular_speed_sqr = self.lib.stop_gradient(
            target_angular_speed_sqr_max * basic_scaling_minus
        )
        argument = (angleD**2 - target_angular_speed_sqr)  # *basic_scaling
        # return argument ** 2
        return self.lib.abs(argument)

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

    def weights(self):

        def weights_up():
            self.lib.assign(self.target_angular_speed_sqr_max_correction, self.target_angular_speed_sqr_max_correction_up)
            return (self.dd_quadratic_weight_up, self.dd_linear_weight_up, self.db_weight_up,
                    self.ep_weight_up, self.ekp_weight_up,
                    self.cc_weight_up, self.ccrc_weight_up)

        def weights_down():
            self.lib.assign(self.target_angular_speed_sqr_max_correction, self.target_angular_speed_sqr_max_correction_down)
            return (self.dd_quadratic_weight_down, self.dd_linear_weight_down, self.db_weight_down,
                    self.ep_weight_down, self.ekp_weight_down,
                    self.cc_weight_down, self.ccrc_weight_down)

        return self.lib.cond(
            self.lib.equal(self.variable_parameters.target_equilibrium, 1.0),
            true_fn=weights_up,
            false_fn=weights_down
        )

    # This is the old/standard cost function, below second one, but adapted for easier logging
    # all stage costs together

    def stage_cost_components(self, states: TensorType, inputs: TensorType, previous_input: TensorType):


        dd_quadratic_weight, dd_linear_weight, db_weight, ep_weight, ekp_weight, cc_weight, ccrc_weight = self.weights()

        # target_position_efficient = self.get_target_position_efficient(states[:, :, POSITION_IDX])
        # dd_quadratic = dd_quadratic_weight * self._distance_difference_cost_quadratic(
        #     states[:, :, POSITION_IDX],
        #     target_position_efficient
        # )
        # dd_linear = dd_linear_weight * self._distance_difference_cost_linear(
        #     states[:, :, POSITION_IDX],
        #     target_position_efficient
        # )

        dd_quadratic = dd_quadratic_weight * self._distance_difference_cost_quadratic(
            states[:, :, POSITION_IDX],
            self.variable_parameters.target_position
        )
        dd_linear = dd_linear_weight * self._distance_difference_cost_linear(
            states[:, :, POSITION_IDX],
            self.variable_parameters.target_position
        )

        db = db_weight * self._boundary_approach_cost(states[:, :, POSITION_IDX])
        ep = ep_weight * self._E_pot_cost(states[:, :, ANGLE_IDX])
        ekp = ekp_weight * self._E_kin_cost(states[:, :, ANGLE_COS_IDX], states[:, :, ANGLED_IDX])
        # ekp = ekp_weight * self._E_kin_cost(states[:, :, ANGLED_IDX])
        cc = cc_weight * self._CC_cost(inputs)
        ccrc = ccrc_weight * self._control_change_rate_cost(inputs, previous_input)

        return dd_quadratic, dd_linear, db, ep, ekp, cc, ccrc

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):
        dd_quadratic, dd_linear, db, ep, ekp, cc, ccrc = self.stage_cost_components(states, inputs, previous_input)
        stage_cost = dd_linear + dd_quadratic + db + ep + ekp + cc +  ccrc
        return stage_cost

    def get_summed_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):

        dd_quadratic, dd_linear, db, ep, ekp, cc, ccrc = self.stage_cost_components(states, inputs, previous_input)

        self.lib.assign(self.dd_quadratic, self.lib.sum(dd_quadratic, 1))
        self.lib.assign(self.dd_linear, self.lib.sum(dd_linear, 1))
        self.lib.assign(self.db, self.lib.sum(db, 1))
        self.lib.assign(self.ep, self.lib.sum(ep, 1))
        self.lib.assign(self.ekp, self.lib.sum(ekp, 1))
        self.lib.assign(self.cc, self.lib.sum(cc, 1))
        self.lib.assign(self.ccrc, self.lib.sum(ccrc, 1))

        self.lib.assign(self.stage_cost, self.dd_linear + self.dd_quadratic + self.db + self.ep + self.ekp + self.cc + self.ccrc)
        return self.stage_cost

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
