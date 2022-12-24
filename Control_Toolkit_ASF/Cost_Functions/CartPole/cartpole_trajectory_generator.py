import numpy as np

from CartPole import state_utilities
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Cost_Functions import cost_function_base
from Control_Toolkit.others.globals_and_utils import get_logger
from GUI import gui_default_params
from SI_Toolkit.computation_library import TensorType
import tensorflow as tf

log=get_logger(__name__)

def generate_cartpole_trajectory(time: float, state: np.ndarray, controller:template_controller, cost_function: cost_function_base) -> TensorType:
    """ Computes the desired future state trajectory at this time.

    :param time: the scalar time in seconds
    :param horizon: the number of horizon steps
    :param dt: the timestep in seconds
    :param state: the current state of the cartpole as 1d vector

    :returns: the target state trajectory of cartpole.
    It is an np.ndarray[states,timesteps] with NaN as at least first entries of each state for don't care states, and otherwise the desired future state values.

    """
    dt = gui_default_params.controller_update_interval
    mpc_horizon = controller.optimizer.mpc_horizon
    gui_target_position = cost_function.target_position  # GUI slider position
    gui_target_equilibrium = cost_function.target_equilibrium  # GUI switch +1 or -1 to make pole target up or down position

    traj = np.full(shape=(state_utilities.NUM_STATES, mpc_horizon),
                   fill_value=np.nan)  # use numpy not tf, too hard to work with immutable tensors

    policy = cost_function.policy
    if policy is None:
        raise RuntimeError(f'set policy in config_self.controller.cost_function_wrapper.cost_functions.yml')

    if policy == 'spin':  # spin pole CW or CCW depending on target_equilibrium up or down
        endtime = float(mpc_horizon) * dt
        times = np.linspace(0, endtime, num=mpc_horizon)
        s_per_rev_target = cost_function.spin_rev_period_sec
        rad_per_s_target = gui_target_equilibrium * 2 * np.pi / s_per_rev_target # note direction of spin from target_equilibrium
        rad_per_dt = rad_per_s_target * dt
        current_angle = state[state_utilities.ANGLE_IDX]
        angle_trajectory=current_angle +  times * rad_per_dt
        traj[state_utilities.POSITION_IDX] = gui_target_position
        # traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(angle_trajectory)
        # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(angle_trajectory)
        # traj[state_utilities.ANGLE_IDX, :] = angle_trajectory
        traj[state_utilities.ANGLED_IDX, :] = rad_per_s_target # 1000 rad/s is arbitrary, not sure if this is best target
        # traj[state_utilities.POSITIOND_IDX, :] = 0
    elif policy == 'balance':  # balance upright or down at desired cart position
        target_angle = np.pi * (1 - gui_target_equilibrium) / 2  # either 0 for up and pi for down
        traj[state_utilities.POSITION_IDX] = gui_target_position
        traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(target_angle)
        # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
        # traj[state_utilities.ANGLE_IDX, :] = target_angle
        # traj[state_utilities.ANGLED_IDX, :] = 0
        # traj[state_utilities.POSITIOND_IDX, :] = 0
    elif policy == 'shimmy':  # cart follows a desired cart position shimmy while keeping pole up or down
        per = cost_function.shimmy_per  # seconds
        amp = cost_function.shimmy_amp  # meters
        endtime = time + mpc_horizon * dt
        times = np.linspace(time, endtime, num=mpc_horizon)
        cartpos = amp * np.sin((2 * np.pi / per) * times)
        cartvel = np.gradient(cartpos, dt)
        target_angle = np.pi * (1 - gui_target_equilibrium) / 2  # either 0 for up and pi for down
        traj[state_utilities.POSITION_IDX] = gui_target_position + cartpos
        traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(target_angle)
        # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
        # traj[state_utilities.ANGLE_IDX, :] = target_angle
        # traj[state_utilities.ANGLED_IDX, :] = 0
        traj[state_utilities.POSITIOND_IDX, :] = cartvel
    elif policy == 'cartonly':  # cart follows the trajectory, pole ignored
        per = cost_function.cartonly_per  # seconds
        amp = cost_function.cartonly_amp  # meters
        endtime = time + mpc_horizon * dt
        times = np.linspace(time, endtime, num=mpc_horizon)
        from scipy.signal import sawtooth
        cartpos = amp * sawtooth((2 * np.pi / per) * times,
                                 width=cost_function.cartonly_duty_cycle)  # width=.5 makes triangle https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html
        cartvel = np.gradient(cartpos, dt)
        traj[state_utilities.POSITION_IDX] = gui_target_position + cartpos
        # target_angle=np.pi * (1-gui_target_equilibrium)/2 # either 0 for up and pi for down
        # traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(target_angle)
        # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
        # traj[state_utilities.ANGLE_IDX, :] = target_angle
        # traj[state_utilities.ANGLED_IDX, :] = 0
        traj[state_utilities.POSITIOND_IDX, :] = cartvel
    else:
        log.error(f'cost policy "{policy}" is unknown')

    return traj
