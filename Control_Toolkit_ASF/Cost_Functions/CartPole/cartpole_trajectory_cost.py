import os

import tensorflow
from yaml import safe_load

from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.globals_and_utils import get_logger
from SI_Toolkit.computation_library import ComputationLibrary, TensorType
from Control_Toolkit.Cost_Functions import cost_function_base

from others.globals_and_utils import load_config, load_or_reload_config_if_modified

from CartPole.cartpole_model import TrackHalfLength
from CartPole.state_utilities import NUM_STATES, ANGLE_IDX, POSITION_IDX, POSITIOND_IDX, ANGLED_IDX
log=get_logger(__name__)

# (config, _) = load_or_reload_config_if_modified(os.path.join("Control_Toolkit_ASF", "config_cost_functions.yml"),
#                                                        every=1)
# 
# weights=config.CartPole.cartpole_trajectory_cost
# log.info(f'starting MPC cartpole trajectory cost weights={weights}')  # only runs once



class cartpole_trajectory_cost(cost_function_base):

    def __init__(self, controller: template_controller, ComputationLib: "type[ComputationLibrary]",
                 config: dict = None) -> None:
        """ makes a new cost function

        :param controller: the controller
        :param ComputationLib: the library, e.g. python, tensorflow
        :param config: the dict of configuration for this cost function.  The caller can modify the config to change behavior during runtime.

         """

        super().__init__(controller, ComputationLib)


    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):
        """
        Computes costs of all stages, i.e., costs of all rollouts for each timestep.

        :param states: Tensor of states [rollout, timestep, state].
            rollout dimension is the batch dimension of the rollouts, e.g. 1000 parallel rollouts.
            timestep dimension are the timesteps of each rollout, e.g. 50 timesteps.
            state is the state vector, defined by the STATE_INDICES in [state_utilities.py](../CartPole/state_utilities.py)
        :param inputs: control input, e.g. cart acceleration
        :param previous_input: previous control input, can be used to compute cost of changing control input

        :returns: Tensor of [rollout, timestamp] over all rollouts and horizon timesteps
        """
        # print("test") # only prints once since TF graph is only built once, and not rebuilt if arguments do not change

        dd = 0 # self.dd_weight * self._distance_difference_cost(states[:, :, POSITION_IDX]) # compute cart position target distance cost
        control_cost = 0 # self.cc_weight * self.R*self._CC_cost(inputs) # compute the cart acceleration control cost
        control_change_cost = 0 # compute the control change cost
        ep=0 # pole angle cost
        position=states[:, :, POSITION_IDX]
        positionD=states[:, :, POSITIOND_IDX]
        angle=states[:, :, ANGLE_IDX]
        angleD=states[:,:,ANGLED_IDX]
        gui_target_position=self.controller.target_position # GUI slider position
        gui_target_equilibrium=self.controller.target_equilibrium # GUI switch +1 or -1 to make pole target up or down position
        input_shape=self.lib.shape(states)
        num_rollouts=input_shape[0]
        num_timesteps=input_shape[1]
        trajectory_cost=self.lib.zeros((num_rollouts,num_timesteps))
        target_trajectory=self.controller.target_trajectory # [state, timestep]
        # "angle", "angleD", "angle_cos", "angle_sin", "position", "positionD"
        cost_weights=(self.pole_angle_weight, self.pole_swing_weight, self.pole_angle_weight, self.pole_angle_weight, self.cart_pos_weight, self.cart_vel_weight)
        for i in range(NUM_STATES):
            if not self.lib.any(self.lib.isnan(target_trajectory[i,0])): # to skip a state, the first timestep is NaN
                state_i=states[:,:,i] # matrix [rollout,timestep] for this state element
                trajectory_i=target_trajectory[i,:] # timestep vector for this state element
                zerodiffs=self.lib.zeros((num_rollouts, num_timesteps-1))
                terminaldiffs=state_i[:,-1]-trajectory_i[-1]
                terminaldiffs_unsqueezed=self.lib.unsqueeze(terminaldiffs,1)
                diff=self.lib.concat((zerodiffs,terminaldiffs_unsqueezed),1) # -1 is terminal state, subtracts time vector trajectory_i from each time row i of state
                diffabs=self.lib.abs(diff) # make it unsigned error matrix [rollout, timestep], in this case just last timestep of rollout
                # don't do sum here, it is done in get_trajectory_cost() caller
                # sums=self.lib.sum(diff2,1) # sums over the time dimension, leaving column vector of rollouts
                trajectory_cost+=cost_weights[i]*diffabs
        # control_cost_weight: 1.0
        # control_cost_change_weight: 1.0

        # ep = self.ep_weight * angleD # make pole spin
        if previous_input is not None:
            control_change_cost = self.control_cost_change_weight * self._control_change_rate_cost(inputs, previous_input)
        control_cost = self.control_cost_weight *self._CC_cost(inputs) # compute the cart acceleration control cost
        # if self.controller.target_positions_vector[0] > 0: # TODO why is potential cost positive for this case and negative otherwise?
        #     stage_cost = dd + ep + cc + ccrc
        # else:
        stage_cost = trajectory_cost +control_cost + control_change_cost # hack to bring in GUI stuff to cost

        return stage_cost


    # cost for distance from cart track edge
    def _distance_difference_cost(self, position):
        """Compute penalty for distance of cart to the target position"""
        return (
            (position - self.controller.target_position) / (2.0 * TrackHalfLength)
        ) ** 2 + self.lib.cast(
            self.lib.abs(position) > 0.90 * TrackHalfLength, self.lib.float32
        ) * 1.0e7  # Soft constraint: Do not crash into border

    # cost for difference from upright position
    def _E_pot_cost(self, angle):
        """Compute penalty for not balancing pole upright (penalize large angles)"""
        return self.controller.target_equilibrium * 0.25 * (1.0 - self.lib.cos(angle)) ** 2

    # actuation cost
    def _CC_cost(self, u):
        return self.lib.sum(u**2, 2)

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
        #         self.lib.abs(terminal_states[:, POSITION_IDX] - self.controller.target_position)
        #         > 0.1 * TrackHalfLength
        #     ),
        #     self.lib.float32,
        # )
        return 0

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

