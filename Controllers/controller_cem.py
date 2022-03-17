import scipy
import numpy as np
from numpy.random import SFC64, Generator
from datetime import datetime
from numba import jit

from Controllers.template_controller import template_controller
from CartPole.cartpole_model import TrackHalfLength

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state
from CartPole.cartpole_model import u_max, s0
from CartPole.cartpole_jacobian import cartpole_jacobian

import yaml

from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf

config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

dt = config["controller"]["mppi"]["dt"]
cem_horizon = config["controller"]["mppi"]["mpc_horizon"]
num_rollouts = config["controller"]["mppi"]["num_rollouts"]
dd_weight = config["controller"]["mppi"]["dd_weight"]
cc_weight = config["controller"]["mppi"]["cc_weight"]
ep_weight = config["controller"]["mppi"]["ep_weight"]
R = config["controller"]["mppi"]["R"]
cem_samples = int(cem_horizon / dt)  # Number of steps in MPC horizon

predictor = predictor_ODE(horizon=cem_samples, dt=dt, intermediate_steps=10)



@jit(nopython=True, cache=True, fastmath=True)
def distance_difference_cost(position, target_position):
    """Compute penalty for distance of cart to the target position"""
    return ((position - target_position) / (2.0 * TrackHalfLength)) ** 2 + (
        np.abs(position) > 0.95 * TrackHalfLength
    ) * 1.0e6  # Soft constraint: Do not crash into border

@jit(nopython=True, cache=True, fastmath=True)
def E_pot_cost(angle):
    """Compute penalty for not balancing pole upright (penalize large angles)"""
    return 0.25 * (1.0 - np.cos(angle)) ** 2

@jit(nopython=True,cache = True, fastmath = True)
def CC_cost(u):
    return R * (u ** 2)

@jit(nopython=True, cache=True, fastmath=True)
def phi(s: np.ndarray, target_position: np.float32) -> np.ndarray:
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
    terminal_cost = 10000 * (
        (np.abs(terminal_states[:, ANGLE_IDX]) > 0.2)
        | (
            np.abs(terminal_states[:, POSITION_IDX] - target_position)
            > 0.1 * TrackHalfLength
        )
    )
    return terminal_cost


def q(s :np.ndarray,u:np.ndarray,target_position: np.float32):
    dd = dd_weight * distance_difference_cost(
        s[:, :, POSITION_IDX], target_position
    ).astype(np.float32)
    ep = ep_weight * E_pot_cost(s[:, :, ANGLE_IDX]).astype(np.float32)
    cc = cc_weight * CC_cost(u)

    stage_cost = dd+ep+cc
    return stage_cost

def cost(s_hor :np.ndarray,u:np.ndarray,target_position: np.float32):
    stage_cost = q(s_hor[:,1:,:],u,target_position)
    total_cost = np.sum(stage_cost,axis=1)
    total_cost+= phi(s_hor,target_position)
    return total_cost

class controller_cem(template_controller):
    def __init__(self):
        SEED = config["controller"]["mppi"]["SEED"]
        if SEED == "None":
            SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)  # Fully random
        self.rng_cem = Generator(SFC64(SEED))

        self.dist_mue = np.zeros([1,cem_samples])
        self.dist_var = 0.5*np.ones([1,cem_samples])
        self.stdev = np.sqrt(self.dist_var)


    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):
        Q = np.tile(self.dist_mue,(num_rollouts,1))+ np.multiply(self.rng_cem.standard_normal(
            size=(num_rollouts, cem_samples), dtype=np.float32),self.stdev)
        Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)
        rollout_trajectory = predictor.predict(np.copy(s), Q[:,:, np.newaxis])
        traj_cost = cost(rollout_trajectory, Q, target_position)
        sorted_cost = np.argsort(traj_cost)
        best_idx = sorted_cost[0:20]
        elite_Q = Q[best_idx,:]
        self.dist_mue = np.mean(elite_Q,axis = 0)
        self.stdev = np.std(elite_Q,axis=0)[np.newaxis,:]
        self.stdev = np.append(self.stdev[1:], 0.4).astype(np.float32)
        self.dist_mue = np.append(self.dist_mue[1:], 0).astype(np.float32)
        u = self.dist_mue[0]
        return u




if __name__ == '__main__':
    ctrl = controller_cem()


    import timeit

    s0 = create_cartpole_state()
    # Set non-zero input
    s = s0
    s[POSITION_IDX] = -30.2
    s[POSITIOND_IDX] = 2.87
    s[ANGLE_IDX] = -0.32
    s[ANGLED_IDX] = 0.237
    u = -0.24

    ctrl.step(s0, 0)
    f_to_measure = 'ctrl.step(s0,0)'
    number = 1  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 100  # Gives how many times timeit should be repeated
    timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
    min_time = min(timings) / float(number)
    max_time = max(timings) / float(number)
    average_time = np.mean(timings) / float(number)
    print()
    print('----------------------------------------------------------------------------------')
    print('Min time to evaluate is {} ms'.format(min_time * 1.0e3))  # ca. 5 us
    print('Average time to evaluate is {} ms'.format(average_time * 1.0e3))  # ca 5 us
    # The max is of little relevance as it is heavily influenced by other processes running on the computer at the same time
    print('Max time to evaluate is {} ms'.format(max_time * 1.0e3))  # ca. 100 us
    print('----------------------------------------------------------------------------------')
    print()