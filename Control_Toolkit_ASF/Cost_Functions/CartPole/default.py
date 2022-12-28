import os
from yaml import safe_load

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base

from others.globals_and_utils import load_config, load_or_reload_config_if_modified

from CartPole.cartpole_model import TrackHalfLength
from CartPole.state_utilities import ANGLE_IDX, POSITION_IDX, ANGLED_IDX

class default(cost_function_base):

    # all stage costs together
    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):
        """
        Computes costs of all stages, i.e., costs of all rollouts for each timestep.

        :param states: Tensor of states [rollout, timestep, state].
            rollout dimension is the batch dimension of the rollouts, e.g. 1000 parallel rollouts.
            timestep dimension are the timesteps of each rollout, e.g. 50 timesteps.
            state is the state vector, defined by the STATE_INDICES in [state_utilities.py](../CartPole/state_utilities.py)
        :param inputs: control input, e.g. cart acceleration
        :param previous_input: previous control input, can be used to compute cost of changing control input
        :return: Tensor of [rollout, timestamp] over all rollouts and horizon timesteps
        """
        (config,reloaded)=load_or_reload_config_if_modified(os.path.join("Control_Toolkit_ASF", "config_cost_functions.yml"),every=1)
        self.dd_weight = config["CartPole"]["default"]["dd_weight"]
        self.cc_weight = config["CartPole"]["default"]["cc_weight"]
        self.ep_weight = config["CartPole"]["default"]["ep_weight"]
        self.ccrc_weight = config["CartPole"]["default"]["ccrc_weight"]
        self.R = config["CartPole"]["default"]["R"]

        dd = self.dd_weight * self._distance_difference_cost(states[:, :, POSITION_IDX]) # compute cart position target distance cost
        cc = self.cc_weight * self.R*self._CC_cost(inputs) # compute the cart acceleration control cost
        ccrc = 0 # compute the control change cost
        angle=states[:, :, ANGLE_IDX]
        angleD=states[:,:,ANGLED_IDX]
        ep = self.ep_weight * self._E_pot_cost(angle)
        # if previous_input is not None:
        #     ccrc = ccrc_weight * self._control_change_rate_cost(inputs, previous_input)

        # if self.controller.target_positions_vector[0] > 0: # TODO why is potential cost positive for this case and negative otherwise?
        #     stage_cost = dd + ep + cc + ccrc
        # else:
        stage_cost = dd + ep + cc + ccrc

        return stage_cost


    # cost for distance from track edge
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
        terminal_cost = 10000 * self.lib.cast(
            (self.lib.abs(terminal_states[:, ANGLE_IDX]) > 0.2)
            | (
                self.lib.abs(terminal_states[:, POSITION_IDX] - self.controller.target_position)
                > 0.1 * TrackHalfLength
            ),
            self.lib.float32,
        )
        return terminal_cost

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

