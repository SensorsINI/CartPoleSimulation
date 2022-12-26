import numpy as np

from CartPole import state_utilities
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Cost_Functions import cost_function_base
from Control_Toolkit.others.globals_and_utils import get_logger
from Control_Toolkit_ASF.Cost_Functions.CartPole.cartpole_dancer import cartpole_dancer
from GUI import gui_default_params
from SI_Toolkit.computation_library import TensorType
import tensorflow as tf

log=get_logger(__name__)
class cartpole_trajectory_generator:

    def __init__(self):
        self.cartpole_dancer=cartpole_dancer()
        self._prev_policy=None

    def generate_cartpole_trajectory(self, time: float, state: np.ndarray, controller:template_controller, cost_function: cost_function_base) -> TensorType:
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
        dancer_current_step=None
        if policy is None:
            raise RuntimeError(f'set policy in config_self.controller.cost_function_wrapper.cost_functions.yml')

        elif policy=='dance':
            if self._prev_policy is None or self._prev_policy!=policy:
                self.cartpole_dancer.start(time)
            self.cartpole_dancer.step(time=time)

            self._prev_policy=policy
            policy=self.cartpole_dancer.policy
            cost_function.cartpos=self.cartpole_dancer.cartpos
            if policy=='spin':
                cost_function.spin_dir=self.cartpole_dancer.option
                cost_function.spin_freq_hz=self.cartpole_dancer.freq
            elif policy=='balance':
                cost_function.balance_dir=self.cartpole_dancer.option
            elif policy=='shimmy':
                cost_function.shimmy_freq_hz=self.cartpole_dancer.freq
                cost_function.shimmy_amp=self.cartpole_dancer.amp
            elif policy=='cartonly':
                cost_function.cartonly_freq_hz=self.cartpole_dancer.freq
                cost_function.cartonly_amp=self.cartpole_dancer.amp
                cost_function.cartonly_duty_cycle=float(self.cartpole_dancer.option)


        if policy == 'spin':  # spin pole CW or CCW depending on target_equilibrium up or down
            spin_dir_factor=1
            if cost_function.spin_dir=='cw':
                spin_dir_factor=1
            elif cost_function.spin_dir=='ccw':
                spin_dir_factor=-1
            else:
                log.warning(f'spin_dir value of "{cost_function.spin_dir} must be "cw" or "ccw"')
            endtime = float(mpc_horizon) * dt
            times = np.linspace(0, endtime, num=mpc_horizon)
            s_per_rev_target = spin_dir_factor/cost_function.spin_freq_hz
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
            up_down=1
            if cost_function.balance_dir=='up':
                up_down=1
            elif cost_function.balance_dir=='down':
                up_down=-1
            else:
                log.warning(f'balance_dir value of "{cost_function.balance_dir} must be "up" or "down"')
            target_angle = np.pi * (1 - gui_target_equilibrium*up_down) / 2  # either 0 for up and pi for down
            traj[state_utilities.POSITION_IDX] = gui_target_position
            traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(target_angle)
            # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
            # traj[state_utilities.ANGLE_IDX, :] = target_angle
            # traj[state_utilities.ANGLED_IDX, :] = 0
            # traj[state_utilities.POSITIOND_IDX, :] = 0
        elif policy == 'shimmy':  # cart follows a desired cart position shimmy while keeping pole up or down
            per = 1/cost_function.shimmy_freq_hz  # seconds
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
            per = 1./cost_function.cartonly_freq_hz  # seconds
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
