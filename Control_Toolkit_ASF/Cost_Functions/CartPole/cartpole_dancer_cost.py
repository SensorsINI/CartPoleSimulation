import os

import numpy as np
import setuptools
import tensorflow
from torch import TensorType
from yaml import safe_load

from CartPole import state_utilities
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.globals_and_utils import get_logger
import Control_Toolkit_ASF.Cost_Functions.CartPole.cartpole_trajectory_generator
from GUI import gui_default_params
from SI_Toolkit.computation_library import TensorType, ComputationLibrary
from Control_Toolkit.Cost_Functions import cost_function_base

from others.globals_and_utils import load_config, load_or_reload_config_if_modified

from CartPole.cartpole_model import TrackHalfLength, L, m_pole
from CartPole.state_utilities import NUM_STATES, ANGLE_IDX, POSITION_IDX, POSITIOND_IDX, ANGLED_IDX

log = get_logger(__name__)



# (config, _) = load_or_reload_config_if_modified(os.path.join("Control_Toolkit_ASF", "config_cost_functions.yml"),
#                                                        every=1)
# 
# weights=config.CartPole.cartpole_dancer_cost
# log.info(f'starting MPC cartpole trajectory cost weights={weights}')  # only runs once

import tensorflow as tf


class cartpole_dancer_cost(cost_function_base):
    """ Computes the rollout costs for cartpole dancer.
    The state trajectories are computed from the current timestep according to desired 'step" type.
    The costs are weighted according to config_cost_functions.yml values for various state components, e.g. cart position, pole swing velocity, etc.

    """

    def __init__(self, controller: template_controller, ComputationLib: "type[ComputationLibrary]",
                 config: dict = None) -> None:
        """ makes a new cost function

        :param controller: the controller
        :param ComputationLib: the library, e.g. python, tensorflow
        :param config: the dict of configuration for this cost function.  The caller can modify the config to change behavior during runtime.

         """
        super().__init__(controller, ComputationLib)
        self.new_target_trajectory = None
        dist_metric = 'abs'
        if dist_metric == 'rmse':
            self.dist = lambda x: self.lib.sqrt(self.lib.pow(x, 2))
        elif dist_metric == 'mse':
            self.dist = lambda x: self.lib.pow(x, 2)
        elif dist_metric == 'abs':
            self.dist = lambda x: self.lib.abs(x)
        else:
            raise Exception(f'unknown distance metric for cost "{dist_metric}"')

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType, time: float = None):
        """
        Computes costs of all stages, i.e., costs of all rollouts for each timestep.

        :param states: Tensor of states [rollout, timestep, state].
            rollout dimension is the batch dimension of the rollouts, e.g. 1000 parallel rollouts.
            timestep dimension are the timesteps of each rollout, e.g. 50 timesteps.
            state is the state vector, defined by the STATE_INDICES in [state_utilities.py](../CartPole/state_utilities.py)
        :param inputs: control input, e.g. cart acceleration
        :param previous_input: previous control input, can be used to compute cost of changing control input
        :param time: the time in seconds, used for generating target trajectory now

        :returns: Tensor of [rollout, timestamp] over all rollouts and horizon timesteps
        """

        control_change_cost = 0  # compute the control change cost

        # position=states[:, :, POSITION_IDX]
        # positionD=states[:, :, POSITIOND_IDX]
        # angle=states[:, :, ANGLE_IDX]
        # angleD=states[:,:,ANGLED_IDX]

        gui_target_equilibrium = self.target_equilibrium  # GUI switch +1 or -1 to make pole target up or down position

        input_shape = self.lib.shape(states)
        num_rollouts = input_shape[0]
        num_timesteps = input_shape[1]
        target_trajectory = self.target_trajectory

        stage_costs = self.lib.zeros((num_rollouts, num_timesteps))
        if not self.lib.equal(self.policy_number, 2):  # if not spin from 'spin2' as policy
            # states are "angle", "angleD", "angle_cos", "angle_sin", "position", "positionD"
            cost_weights = (
            self.pole_angle_weight, self.pole_swing_weight, self.pole_angle_weight, self.pole_angle_weight,
            self.cart_pos_weight, self.cart_vel_weight)
            for i in range(NUM_STATES):
                if not self.lib.any(
                        self.lib.isnan(target_trajectory[i, 0])):  # to skip a state, the first timestep is NaN
                    state_i = states[:, :, i]  # matrix [rollout,timestep] for this state element
                    trajectory_i = target_trajectory[i, :]  # timestep vector for this state element
                    # if use_terminal_state_only==1:
                    #     zerodiffs=self.lib.zeros((num_rollouts, num_timesteps-1))
                    #     terminaldiffs=state_i[:,-1]-trajectory_i[-1]
                    #     terminaldiffs_unsqueezed=self.lib.unsqueeze(terminaldiffs,1)
                    #     diff=self.lib.concat((zerodiffs,terminaldiffs_unsqueezed),1) # -1 is terminal state, subtracts time vector trajectory_i from each time row i of state
                    # else:
                    diff = state_i - trajectory_i

                    # make it unsigned error matrix [rollout, timestep], in this case just last timestep of rollout if use_terminal_state_only==1
                    # diffabs=self.lib.zeros_like(diff) # needed for tf.compile
                    # # dist_norm=self.distance_norm
                    # def abs(x): return tf.abs(x)
                    # def mse(x): return tf.pow(x,2)
                    # def rmse(x): return tf.sqrt(tf.pow(x,2))
                    # branch_fns={0:abs, 1: rmse, 2: mse}
                    # dist_branch=tf.constant(self.distance_norm, tf.int32)
                    # tf.switch_case(dist_branch, branch_fns)
                    diffabs = self.dist(diff)

                    # don't do sum here, it is done in get_trajectory_cost() caller
                    # sums=self.lib.sum(diff2,1) # sums over the time dimension, leaving column vector of rollouts
                    cost_i = cost_weights[i] * diffabs
                    stage_costs += cost_i
        else:  # spin cost is based on total pole energy plus cart position plus rotation speed in cw or ccw direction
            upright_pole_energy=2*self.pole_energy_potential(0) # energy of pole upright compared with hanging down (factor of 2), means pole can swing around all the way if total energy is larger than this
            angleD_states = states[:, :, state_utilities.ANGLED_IDX]
            cosangle_states = states[:, :, state_utilities.ANGLE_COS_IDX]
            # compute the total pole energy, which is sum of kinetic and potential energy of pole
            spin_cost_energy = self.pole_energy_kinetic(angleD_states) + self.pole_energy_potential(cosangle_states)
            # count the spin direction cost only if the total pole energy is some multiple of the upright pole energy
            # spin direction cost is weighted by pole_swing_weight
            # and measures the spin direction cost by the absolute pole angle derivative,
            # but only for trajectories that have total energy larger than the upright pole energy by the multiple UPRIGHT_POLE_ENERGY_MULTIPLE_TO_COUNT_SPIN_DIRECTION_COST
            less_than_spin_energy_mask = float(
                spin_cost_energy < self.upright_pole_energy_multiple_to_count_spin_direction_cost * upright_pole_energy)
            more_than_spin_energy_mask=1-less_than_spin_energy_mask
            spin_cost_spin_direction =   -spin_cost_energy* more_than_spin_energy_mask\
                            *float(self.lib.sign(angleD_states*float(self.spin_dir_number) * gui_target_equilibrium))
            # only count energy of pole for trajectories that don't have sufficient energy to spin
            spin_cost_energy = spin_cost_energy * less_than_spin_energy_mask
            # compute the cart position cost as distance from desired cart position
            pos_states = states[:, :, state_utilities.POSITION_IDX]
            gui_target_position = self.target_position  # GUI slider position
            pos_cost =  self.dist(pos_states - gui_target_position)*more_than_spin_energy_mask
            # total stage cost is sum of energy of pole spin, direction of spin, and cart position
            stage_costs =  -self.pole_energy_total_weight * spin_cost_energy + -self.pole_swing_weight *spin_cost_spin_direction + self.cart_pos_weight * pos_cost

        if previous_input is not None:
            control_change_cost = self.control_cost_change_weight * self._control_change_rate_cost(inputs, previous_input)
        control_cost = self.control_cost_weight * self._CC_cost(inputs)  # compute the cart acceleration control cost

        stage_cost = stage_costs + control_cost + control_change_cost  # hack to bring in GUI stuff to cost

        return stage_cost * self.stage_cost_factor

    # final stage cost
    def get_terminal_cost(self, terminal_states: TensorType):
        """Calculate terminal cost of a set of trajectories

        For the cartpole dancer, we ignore terminal cost for trajectory
        and compute only track edge avoidance cost here, so that any rollout that ends with the cart hitting the track edge is culled.

        Williams et al use an indicator function type of terminal cost the penelizes crashes in
       Williams, Grady, Nolan Wagener, Brian Goldfain, Paul Drews, James M. Rehg, Byron Boots, and Evangelos A. Theodorou. 2017.
       “Information Theoretic MPC for Model-Based Reinforcement Learning.”
       In 2017 IEEE International Conference on Robotics and Automation (ICRA), 1714–21.
       https://doi.org/10.1109/ICRA.2017.7989202, but our terminal cost does not use it.

        :param terminal_states: tensor of terminal states of all rollouts, ordered by [rollout, state], e.g. [320,6]
        :type terminal_states: np.ndarray
        :return: One terminal cost per rollout
        :rtype: np.ndarray
        """

        # gui_target_position=self.target_position # GUI slider position
        gui_target_equilibrium = self.target_equilibrium  # GUI switch +1 or -1 to make pole target up or down position

        terminal_cost = self.lib.zeros((terminal_states.shape[0],))  # make a zero vector with length number of rollouts
        # target_trajectory_terminal_state_vector = self.target_trajectory[:, -1]  # take final state as target
        # if not self.lib.equal(self.policy_number, 2):  # if not spin cost
        #     # "angle", "angleD", "angle_cos", "angle_sin", "position", "positionD"
        #     cost_weights = (
        #     self.pole_angle_weight, self.pole_swing_weight, self.pole_angle_weight, self.pole_angle_weight,
        #     self.cart_pos_weight, self.cart_vel_weight)
        #     for i in range(NUM_STATES):
        #         terminal_state_i = terminal_states[:,
        #                            i]  # matrix [rollout,timestep] for this state element, so this results in vector of e.g. N rollout terminals state angles
        #         target_terminal_state_i = target_trajectory_terminal_state_vector[
        #             i]  # timestep vector for this state element
        #         if not self.lib.isnan(target_terminal_state_i):  # to skip a state, the first timestep is NaN
        #             diff = terminal_state_i - target_terminal_state_i
        #
        #             diffabs = self.dist(diff)  # self.lib.pow(diff,2)
        #
        #             cost_i = cost_weights[i] * diffabs
        #             terminal_cost += cost_i
        #         else:
        #             terminal_cost += 0.
        # else:  # spin cost
        #     terminal_state_angled = terminal_states[:, state_utilities.ANGLED_IDX]
        #     terminal_state_cosangle = terminal_states[:, state_utilities.ANGLE_COS_IDX]
        #     terminal_cost_energy = -self.pole_energy_total_weight * (
        #                 self.pole_energy_kinetic(terminal_state_angled) + self.pole_energy_potential(
        #             terminal_state_cosangle))
        #     terminal_cost_spin = gui_target_equilibrium * self.pole_swing_weight * self.dist(terminal_state_angled)
        #     # log.debug(f'spin costs. energy= {tf.math.reduce_mean(terminal_cost_energy):.1g} spin= {tf.math.reduce_mean(terminal_cost_spin):.1g}')
        #     terminal_cost = terminal_cost_energy + terminal_cost_spin


        terminal_traj_costs = terminal_cost * self.terminal_cost_factor
        terminal_traj_edge_barrier_cost =  self.terminal_cost_factor * self._track_edge_barrier(terminal_states[:, state_utilities.POSITION_IDX])
        return (terminal_traj_costs + terminal_traj_edge_barrier_cost)

    # cost for distance from cart track edge
    def _track_edge_barrier(self, position):
        """Compute penalty for distance of cart to the target position"""
        return self.lib.cast(
            self.lib.abs(position) > 0.9 * TrackHalfLength, self.lib.float32
        ) * 1.0e9  # Soft constraint: Do not crash into border

    # actuation cost
    def _CC_cost(self, u):
        return self.lib.sum(u ** 2, 2)

    # cost of changing control too fast
    def _control_change_rate_cost(self, u, u_prev):
        """Compute penalty of control jerk, i.e. difference to previous control input

        :poram u: the current control input (cart acceleration in dimensionless scale)
        :param u_prev: the previous timesteps control input

        :returns: the cost of changing control
        """
        u_prev_vec = self.lib.concat(
            (self.lib.ones((u.shape[0], 1, u.shape[2])) * u_prev, u[:, :-1, :]), 1
        )
        return self.lib.sum((u - u_prev_vec) ** 2, 2)

    def pole_energy_potential(self, cosangle):
        """Compute potential energy of pole

        :param cosangle: the cosine of pole angle (1 when upright)

        :returns: the energy
        """
        return ((1+cosangle)/2) * m_pole * L * 9.8 * 0.5  # hopefully this is height of COM potential gravitational energy

    def pole_energy_kinetic(self, angleD):
        """Compute kinetic energy of pole

         :param angleD: time derivative of pole angle in rad/s

         :returns: the kinetic energy
         """
        return (1. / 3.) * m_pole * L * L * (angleD ** 2)
