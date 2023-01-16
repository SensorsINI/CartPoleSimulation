import numpy as np

from CartPole import state_utilities
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Cost_Functions import cost_function_base
from Control_Toolkit.others.globals_and_utils import get_logger
from Control_Toolkit_ASF.Cost_Functions.CartPole.cartpole_dancer import cartpole_dancer
from GUI import gui_default_params, MainWindow
from SI_Toolkit.computation_library import TensorType
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

log=get_logger(__name__)
class cartpole_trajectory_generator:

    def __init__(self):
        self.cartpole_dancer=cartpole_dancer()
        self._prev_policy=None
        self._prev_dance_policy=None
        self._time_policy_changed=None # when the new dance step started
        self.shimmy_function=None # used for ramp of shimmy
        self._policy_changed=False

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

        policy:str = cost_function.policy
        dancer_current_step=None
        if policy is None:
            raise RuntimeError(f'set policy in config_self.controller.cost_function_wrapper.cost_functions.yml')

        if self._prev_policy is None or self._prev_policy!=policy:
            self._time_policy_changed=time
            self._policy_changed=True
        else:
            self._policy_changed=False
        self._prev_policy=policy

        if policy=='dance':
            if self._policy_changed:
                self.cartpole_dancer.start(time)
            self.cartpole_dancer.step(time=time)

            policy=self.cartpole_dancer.policy
            self._prev_dance_policy=policy


            cost_function.policy_number=self.cartpole_dancer.policy_number
            gui_target_position=self.cartpole_dancer.cartpos
            if policy=='spin':
                cost_function.spin_dir=self.cartpole_dancer.option
                cost_function.spin_freq_hz=self.cartpole_dancer.freq
            elif policy=='balance':
                cost_function.balance_dir=self.cartpole_dancer.option
            elif policy=='shimmy':
                cost_function.shimmy_freq_hz=self.cartpole_dancer.freq
                cost_function.shimmy_amp=self.cartpole_dancer.amp
                cost_function.shimmy_freq2_hz=self.cartpole_dancer.freq2
                cost_function.shimmy_amp2=self.cartpole_dancer.amp2
                cost_function.shimmy_duration=self.cartpole_dancer.duration


            elif policy=='cartonly':
                cost_function.cartonly_freq_hz=self.cartpole_dancer.freq
                cost_function.cartonly_amp=self.cartpole_dancer.amp
                cost_function.cartonly_duty_cycle=float(self.cartpole_dancer.option)


        if policy== 'spin':  # spin pole CW or CCW depending on target_equilibrium up or down
            spin_dir_factor=1
            if cost_function.spin_dir=='cw':
                spin_dir_factor=-1
            elif cost_function.spin_dir=='ccw':
                spin_dir_factor=+1
            else:
                log.warning(f'spin_dir value of "{cost_function.spin_dir} must be "cw" or "ccw"')
            horizon_endtime = float(mpc_horizon) * dt
            times = np.linspace(0, horizon_endtime, num=mpc_horizon)
            rad_per_s_target = spin_dir_factor*gui_target_equilibrium * 2 * np.pi * cost_function.spin_freq_hz # note direction of spin from target_equilibrium
            rad_per_dt = rad_per_s_target * dt
            # current_angle = state[state_utilities.ANGLE_IDX]
            # angle_trajectory=current_angle +  times * rad_per_dt
            traj[state_utilities.POSITION_IDX] = gui_target_position
            # traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(angle_trajectory)
            # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(angle_trajectory)
            # traj[state_utilities.ANGLE_IDX, :] = angle_trajectory
            traj[state_utilities.ANGLED_IDX, :] = rad_per_s_target # 1000 rad/s is arbitrary, not sure if this is best target
            # traj[state_utilities.POSITIOND_IDX, :] = 0
        elif policy=='balance':  # balance upright or down at desired cart position
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
        elif policy=='shimmy':  # cart follows a desired cart position shimmy while keeping pole up or down
            # shimmy is an (optional) ramp from freq to freq2 and amp to amp2
            if time>self._time_policy_changed+cost_function.shimmy_duration:
                self._time_policy_changed=time # reset shimmy and start over if doing it from fixed shimmy policy in yml
                log.debug(f'shimmy restarted at time={time}')
            shimmy_starttime=self._time_policy_changed
            shimmy_endtime=self._time_policy_changed+cost_function.shimmy_duration
            # compute times from current time to end of horizon
            horizon_endtime = time-shimmy_starttime + mpc_horizon * dt
            # time for shimmy must be relative to start of shimmy step for freq ramp to make sense
            times = np.linspace(time-shimmy_starttime, horizon_endtime, num=mpc_horizon)
            time_frac=times/cost_function.shimmy_duration
            f0 = cost_function.shimmy_freq_hz  # seconds
            a0 = cost_function.shimmy_amp  # meters
            f1 = cost_function.shimmy_freq2_hz  # seconds
            a1 = cost_function.shimmy_amp2  # meters

            # compute the shimmmy trajectory over the horizon
            amps=a0+time_frac*(a1-a0)
            freqs=f0+time_frac*(f1-f0)

            # print(f'abs_time={time:.2f}, rel_time={times[0]:.2f} time_frac={time_frac[0]:.3f} amp={amps[0]:.3f} freq={freqs[0]:.2f}')

            cartpos=amps*np.sin(2*np.pi*freqs*times)
            cartpos_d=np.gradient(cartpos,dt)

            # if cost_function.shimmy_plot==1:
            #     # matplotlib.use('Qt5Agg')
            #     f=plt.figure('shimmy')
            #     plt.plot(times,cartpos, times,cartpos_d)
            #     plt.legend(['cart pos','cart vel'])
            #     plt.xlabel('time (s)')
            #     plt.ylabel('m or m/s')
            #     # plt.draw()
            #     plt.show()

            target_angle = np.pi * (1 - gui_target_equilibrium) / 2  # either 0 for up and pi for down
            traj[state_utilities.POSITION_IDX] = gui_target_position + cartpos
            traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(target_angle)
            # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
            # traj[state_utilities.ANGLE_IDX, :] = target_angle
            # traj[state_utilities.ANGLED_IDX, :] = 0
            traj[state_utilities.POSITIOND_IDX, :] = cartpos_d
        elif policy=='cartonly':  # cart follows the trajectory, pole ignored
            per = 1./cost_function.cartonly_freq_hz  # seconds
            amp = cost_function.cartonly_amp  # meters
            horizon_endtime = time + mpc_horizon * dt
            times = np.linspace(time, horizon_endtime, num=mpc_horizon)
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

        self.set_status_text(time,state,cost_function)
        return traj

    def set_status_text(self, time: float, state: np.ndarray, cost_function: cost_function_base) -> None:
        policy:str = cost_function.policy
        if policy=='dance':
            return # status of cartpole GUI set by dancer
        gui_target_position = float(cost_function.target_position)  # GUI slider position
        gui_target_equilibrium = float(cost_function.target_equilibrium)  # GUI switch +1 or -1 to make pole target up or down position

        if policy=='balance':
            dir=self.decode_string(cost_function.balance_dir)
            s= f'Policy: balance/{dir} pos={gui_target_position:.1f}m'
        elif policy=='spin':
            dir=self.decode_string(cost_function.spin_dir)
            s= f'Policy: spin/{dir} pos={gui_target_position:.1f}m freq={float(cost_function.spin_freq_hz):.1f}Hz'
        elif policy=='shimmy':
            s= f'Policy: shimmy pos={gui_target_position:.1f}m freq={float(cost_function.shimmy_freq_hz):.1f}Hz amp={float(cost_function.shimmy_amp):.1f}Hz'
        elif policy=='cartonly':
            s= f'Policy: cartonly pos={gui_target_position:.1f}m freq={float(cost_function.cartonly_freq_hz):.1f}Hz amp={float(cost_function.cartonly_amp):.1f}Hz'
        else:
            s= f'unknown/not implemented string'

        MainWindow.set_status_text(s)

    def decode_string(self,tfstring:tf.Variable):
        return tfstring.numpy().decode('utf-8')