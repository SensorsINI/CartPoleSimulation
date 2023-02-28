import numpy as np

from CartPole import state_utilities
from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.computation_library import TensorType, ComputationLibrary
from Control_Toolkit.Cost_Functions import cost_function_base

from CartPole.cartpole_model import TrackHalfLength, L, m_pole
from CartPole.state_utilities import NUM_STATES

import tensorflow as tf

from Control_Toolkit.others.get_logger import get_logger
log = get_logger(__name__)



# (config, _) = load_or_reload_config_if_modified(os.path.join("Control_Toolkit_ASF", "config_cost_functions.yml"),
#                                                        every=1)
# 
# weights=config.CartPole.cartpole_dancer_cost
# log.info(f'starting MPC cartpole trajectory cost weights={weights}')  # only runs once


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
        dist_metric = 'mse'
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
            states[:,0,:] are num_rollouts copies of current state.
            rollout dimension is the batch dimension of the rollouts, e.g. 1000 parallel rollouts.
            timestep dimension are the timesteps of each rollout, e.g. 50 timesteps.
            state is the state vector, defined by the STATE_INDICES in [state_utilities.py](../CartPole/state_utilities.py)
        :param inputs: control input, e.g. cart acceleration
        :param previous_input: previous control input, can be used to compute cost of changing control input
        :param time: the time in seconds, used for generating target trajectory now

        :returns: Tensor of [rollout, timestep] over all rollouts and horizon timesteps
        """

        control_change_cost = 0  # compute the control change cost

        # position=states[:, :, POSITION_IDX]
        # positionD=states[:, :, POSITIOND_IDX]
        # angle=states[:, :, ANGLE_IDX]
        # angleD=states[:,:,ANGLED_IDX]

        gui_target_equilibrium = self.target_equilibrium  # GUI switch +1 or -1 to make pole target up or down position, set by previous call that update_attributes this field

        input_shape = self.lib.shape(states)
        num_rollouts = input_shape[0]
        mpc_horizon = input_shape[1]
        num_states=input_shape[2]

        # allocate a zero stage costs tensor of dimension [num_rollouts, horizon], using the input states tensor that has the correct dimension
        # squeeze is to remove the single state dimension to go from 3d to 2d tensor
        # this 2d stage_costs tensor is immutable and is replaced every time modify it
        stage_costs = tf.squeeze(tf.zeros_like(states[:,:,0]))



        # dance step policies are: dance0 (follow csv file)   balance1 spin2 shimmy3 cartonly4

        # states are "angle", "angleD", "angle_cos", "angle_sin", "position", "positionD"
        # stage_costs = self.lib.zeros((num_rollouts, num_timesteps))
        target_trajectory = tf.transpose(
            self.target_trajectory)  # [horizon,num_states] array tensor computed by cartpole_trajectory_generator.py
        broadcasted_target_trajectory = tf.broadcast_to(target_trajectory,
                                                        [num_rollouts, mpc_horizon, NUM_STATES])
        cost_weights = self.effective_traj_cost_weights

        traj_dist = self.dist(
            states - broadcasted_target_trajectory)  # difference for each rollout of predicted to target trajectory, has dim [num_rollouts, horizon, num_states]
        stage_costs = tf.multiply(traj_dist, cost_weights)  # multiply states dim by cost_weights
        stage_costs = self.stage_cost_factor * tf.reduce_sum(stage_costs,
                                                             axis=2)  # sum costs across state dimension to leave costs as [num_rollouts, mpc_horizon]
        stage_costs = tf.reshape(stage_costs, [num_rollouts, mpc_horizon])

        if self.lib.equal(self.policy_number, 2): # spin
            # The cost tensor we will return.  we allocate one less than num_timesteps to be all zeros because
            # we concatenate the terminal cost in this branch
            # stage_costs = self.lib.zeros((num_rollouts, num_timesteps - 1))
            # stage_costs = tf.zeros_like(states)
            # stage_costs=stage_costs[:,-1]

            # spin is special cost aimed to first get energy into pole, then make it spin in correct direction.
            # It is totally based on terminal state at end of rollout.
            # Spin cost is based on total pole energy plus cart position plus rotation speed in cw or ccw direction.
            # When the pole has insufficient energy for a spin, we follow objective to maximize the total pole energy.
            # If the pole has sufficient energy for full rotations, then we follow objective to just maximize the signed kinetic energy in the cw or ccw direction.
            upright_pole_energy=self.pole_energy_potential(0) # energy of pole upright compared with hanging down (factor of 2), means pole can swing around all the way if total energy is larger than this
            current_pole_angle=states[0,0,state_utilities.ANGLE_IDX] # rad
            current_pole_angleD=states[0,0,state_utilities.ANGLED_IDX] # rad/s

            # compute the total pole energy, which is sum of kinetic and potential energy of pole
            current_pole_potential_energy=self.pole_energy_potential(current_pole_angle)
            current_pole_kinetic_energy=self.pole_energy_kinetic(current_pole_angleD)
            current_pole_total_energy=current_pole_potential_energy+current_pole_kinetic_energy

            # now find the final state at end of horizon and try to get it to either have more energy or spinning faster in particular direction
            # each of following is [num_rollouts] vector of angleD, cos_angle, and cart position
            angleD_states = states[:, -1, state_utilities.ANGLED_IDX]
            cosangle_states = states[:, -1, state_utilities.ANGLE_COS_IDX]
            pos_states = states[:, -1, state_utilities.POSITION_IDX]

            gui_target_position = self.target_position  # GUI slider position
            pos_cost =  self.dist(pos_states - gui_target_position) # num_rollouts vector

            # compute the cost of spinning, the more energy, the lower the cost
            spin_cost_energy = -(self.pole_energy_kinetic(angleD_states) + self.pole_energy_potential(cosangle_states))

            if current_pole_total_energy>self.upright_pole_energy_multiple_to_count_spin_direction_cost * upright_pole_energy:
                # pole is already spinning, multiply the spin energy by the actual direction of spinning and desired spin direction
                actual_spin_dirs=self.lib.sign(angleD_states)
                desired_spin_dir= -float(self.spin_dir) * gui_target_equilibrium # minus sign to get correct cost sign for cw/ccw rotation
                spin_cost_energy=spin_cost_energy*desired_spin_dir*actual_spin_dirs

            # compute the cart position cost as distance from desired cart position
            # total stage cost is sum of energy of pole spin, direction of spin, and cart position
            # the returned stage_costs is a tensor [num_rollouts, horizon] that has concatenated the terminal costs at end of horizon to the zero stage costs along the horizon
            term_costs=self.spin_energy_weight *spin_cost_energy + self.cart_pos_weight * pos_cost
            term_costs=tf.expand_dims(term_costs,-1) # make it, e.g. with num_rollouts=700, from [700] to [700,1] tensor
            stage_costs =  self.lib.concat([stage_costs[:,:-1],term_costs],1) # now concatenate the terminal costs (all we count for spin) to the zero stage costs replacing the last horizon step
            stage_costs = tf.reshape(stage_costs, [num_rollouts, mpc_horizon])

        if previous_input is not None:
            control_change_cost = self.control_cost_change_weight * self._control_change_rate_cost(inputs, previous_input)
        control_cost = self.control_cost_weight * self._CC_cost(inputs)  # compute the cart acceleration control cost

        terminal_traj_edge_barrier_cost = self.track_edge_barrier_cost * self.track_edge_barrier(states[:, :, state_utilities.POSITION_IDX])

        stage_costs = stage_costs + terminal_traj_edge_barrier_cost + control_cost + control_change_cost   # result has dimension [num_rollouts, horizon]

        return stage_costs

    # final stage cost
    def get_terminal_cost(self, terminal_states: TensorType):
        """ returns 0; terminal cost and barrier cost computed in get_stage_cost()"""
        return self.lib.zeros_like(terminal_states)

    # cost for distance from cart track edge
    def track_edge_barrier(self, position):
        """Soft constraint: Do not crash into border; returns a 1 for any rollout cart position state that exceeds the track edge boundary"""
        return self.lib.cast(
            self.lib.abs(position) > self.track_length_fraction * TrackHalfLength, self.lib.float32
        )

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

        :returns: the pole potential energy in Joules, it is zero when pole is hanging down
        """
        return ((1+cosangle)/2) * m_pole * (L/2) * 9.8  # hopefully this is height of COM potential gravitational energy

    def pole_energy_kinetic(self, angleD):
        """Compute kinetic energy of pole

         :param angleD: time derivative of pole angle in rad/s

         :returns: the kinetic energy in Joules
         """
        return (1. / 6.) * m_pole * L * L * (angleD ** 2)
