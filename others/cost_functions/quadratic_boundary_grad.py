import os
import tensorflow as tf

from CartPole.cartpole_model import TrackHalfLength

from CartPole.state_utilities import (
    ANGLE_IDX,
    ANGLE_SIN_IDX,
    ANGLE_COS_IDX,
    ANGLED_IDX,
    POSITION_IDX,
    POSITIOND_IDX,
    create_cartpole_state,
)
import yaml


#load constants from config file\
try:
    config = yaml.load(open("CartPoleSimulation/config.yml", "r"), Loader=yaml.FullLoader)
except FileNotFoundError:
    config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

dd_weight = config["controller"]["mppi"]["dd_weight"]
cc_weight = tf.convert_to_tensor(config["controller"]["mppi"]["cc_weight"])
ep_weight = config["controller"]["mppi"]["ep_weight"]
ekp_weight = config["controller"]["mppi"]["ekp_weight"]
R = config["controller"]["mppi"]["R"]

ccrc_weight = config["controller"]["mppi"]["ccrc_weight"]


class quadratic_boundary_grad:
    def __init__(self, environment) -> None:
        self.env_mock = environment

    # cost for distance from track edge
    def distance_difference_cost(self, position):
        """Compute penalty for distance of cart to the target position"""
        return (
            (position - self.env_mock.target_position) / (2.0 * TrackHalfLength)
        ) ** 2 + tf.cast(
            tf.abs(position) > 0.95 * TrackHalfLength, tf.float32
        ) * 1e9 * (
            (tf.abs(position) - 0.95 * TrackHalfLength) / (0.05 * TrackHalfLength)
        ) ** 2  # Soft constraint: Do not crash into border

    # cost for difference from upright position
    def E_pot_cost(self, angle):
        """Compute penalty for not balancing pole upright (penalize large angles)"""
        return 0.25 * (1.0 - tf.cos(angle)) ** 2
    
    def E_kin_cost(self, angleD):
        """Compute penalty for not balancing pole upright (penalize large angles)"""
        return angleD ** 2

    # actuation cost
    def CC_cost(self, u):
        return R * tf.reduce_sum(u**2, axis=2)

    # final stage cost
    def get_terminal_cost(self, s):
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
        terminal_states = s[:, -1, :]
        terminal_cost = 10000 * tf.cast(
            (tf.abs(terminal_states[:, ANGLE_IDX]) > 0.2)
            | (
                tf.abs(terminal_states[:, POSITION_IDX] - self.env_mock.CartPoleInstance.target_position)
                > 0.1 * TrackHalfLength
            ),
            tf.float32,
        )
        return terminal_cost

    # cost of changeing control to fast
    def control_change_rate_cost(self, u, u_prev):
        """Compute penalty of control jerk, i.e. difference to previous control input"""
        u_prev_vec = tf.concat(
            (tf.ones((u.shape[0], 1, u.shape[2])) * u_prev, u[:, :-1, :]), axis=1
        )
        return tf.reduce_sum((u - u_prev_vec) ** 2, axis=2)

    # all stage costs together
    def get_stage_cost(self, s, u, u_prev):
        dd = dd_weight * self.distance_difference_cost(
            s[:, :, POSITION_IDX]
        )
        ep = ep_weight * self.E_pot_cost(s[:, :, ANGLE_IDX])
        ekp = ekp_weight * self.E_kin_cost(s[:, :, ANGLED_IDX])
        cc = cc_weight * self.CC_cost(u)
        ccrc = ccrc_weight * self.control_change_rate_cost(u, u_prev)
        stage_cost = dd + ep + ekp + cc + ccrc
        return stage_cost

    def q_debug(self, s, u, u_prev):
        dd = dd_weight * self.distance_difference_cost(
            s[:, :, POSITION_IDX]
        )
        ep = ep_weight * self.E_pot_cost(s[:, :, ANGLE_IDX])
        cc = cc_weight * self.CC_cost(u)
        ccrc = 0
        if u_prev is not None:
            ccrc = ccrc_weight * self.control_change_rate_cost(u, u_prev)
        stage_cost = dd + ep + cc + ccrc
        return stage_cost, dd, ep, cc, ccrc

    # total cost of the trajectory
    def get_trajectory_cost(self, s_hor, u, u_prev=None):
        stage_cost = self.get_stage_cost(s_hor[:, 1:, :], u, u_prev)
        total_cost = tf.math.reduce_sum(stage_cost, axis=1)
        total_cost = total_cost + self.get_terminal_cost(s_hor)
        return total_cost
