import tensorflow as tf

from CartPole.cartpole_model import TrackHalfLength

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state
import yaml


#load constants from config file
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

dd_weight = config["controller"]["mppi"]["dd_weight"]
cc_weight = tf.convert_to_tensor(config["controller"]["mppi"]["cc_weight"])
ep_weight = config["controller"]["mppi"]["ep_weight"]
R = config["controller"]["mppi"]["R"]

ccrc_weight = config["controller"]["mppi"]["ccrc_weight"]

#cost for distance from track edge
def distance_difference_cost(position, target_position):
    """Compute penalty for distance of cart to the target position"""
    return ((position - target_position) / (2.0 * TrackHalfLength)) ** 2 + tf.cast(
        tf.abs(position) > 0.95 * TrackHalfLength
    , tf.float32) * 1e9*((tf.abs(position) - 0.95*TrackHalfLength)/(0.05*TrackHalfLength))**2  # Soft constraint: Do not crash into border

#cost for difference from upright position
def E_pot_cost(angle):
    """Compute penalty for not balancing pole upright (penalize large angles)"""
    return 0.25 * (1.0 - tf.cos(angle)) ** 2

#actuation cost
def CC_cost(u):
    return R * (u ** 2)

#final stage cost
def phi(s, target_position):
    """Calculate terminal cost of a set of trajectories

    Williams et al use an indicator function type of terminal cost in
    "Information theoretic MPC for model-based reinforcement learning"

    TODO: Try a quadratic terminal cost => Use the LQR terminal cost term obtained
    by linearizing the system around the unstable equilibrium.

    :param s: Reference to numpy array of states of all rollouts
    :type s: np.ndarray
    :param target_position: Target position to move the cart to
    :type target_position: np.float32
    :return: One terminal cost per rollout
    :rtype: np.ndarray
    """
    terminal_states = s[:, -1, :]
    terminal_cost = 10000 * tf.cast(
        (tf.abs(terminal_states[:, ANGLE_IDX]) > 0.2)
        | (
            tf.abs(terminal_states[:, POSITION_IDX] - target_position)
            > 0.1 * TrackHalfLength
        )
    , tf.float32)
    return terminal_cost

#cost of changeing control to fast
def control_change_rate_cost(u, u_prev):
    """Compute penalty of control jerk, i.e. difference to previous control input"""
    u_prev_vec = tf.concat((tf.ones((u.shape[0],1))*u_prev,u[:,:-1]),axis=-1)
    return (u - u_prev_vec) ** 2

#all stage costs together
def q(s,u,target_position, u_prev):
    dd = dd_weight * distance_difference_cost(
        s[:, :, POSITION_IDX], target_position
    )
    ep = ep_weight * E_pot_cost(s[:, :, ANGLE_IDX])
    cc = cc_weight * CC_cost(u)
    ccrc = ccrc_weight * control_change_rate_cost(u,u_prev)
    stage_cost = dd+ep+cc+ccrc
    return stage_cost

def q_debug(s,u,target_position, u_prev):
    dd = dd_weight * distance_difference_cost(
        s[:, :, POSITION_IDX], target_position
    )
    ep = ep_weight * E_pot_cost(s[:, :, ANGLE_IDX])
    cc = cc_weight * CC_cost(u)
    ccrc = ccrc_weight * control_change_rate_cost(u,u_prev)
    stage_cost = dd+ep+cc+ccrc
    return stage_cost, dd, ep, cc, ccrc

#total cost of the trajectory
def cost(s_hor ,u,target_position,u_prev):
    stage_cost = q(s_hor[:,1:,:],u,target_position,u_prev)
    total_cost = tf.math.reduce_sum(stage_cost,axis=1)
    total_cost = total_cost + phi(s_hor,target_position)
    return total_cost
