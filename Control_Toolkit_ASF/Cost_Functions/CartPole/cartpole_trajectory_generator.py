import time
from pydoc import text

import numpy as np

from CartPole import state_utilities
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Cost_Functions import cost_function_base
from CartPole import is_physical_cartpole_running_and_control_enabled
from Control_Toolkit_ASF.Cost_Functions.CartPole.cartpole_dancer import cartpole_dancer
from GUI import gui_default_params, CartPoleMainWindow
from SI_Toolkit.computation_library import TensorType
import tensorflow as tf

from Control_Toolkit.others.get_logger import get_logger
from others.globals_and_utils import update_attributes
from others.p_globals import CARTPOLE_PHYSICAL_CONSTANTS

log = get_logger(__name__)



class cartpole_trajectory_generator:

    POLICIES=('balance','spin','shimmy','cartonly','cartwheel')
    POLICIES_NUMBERED=('balance0','spin1','shimmy2','cartonly3','cartwheel4') # for tensorflow scalar BS
    CARTWHEEL_STATES={'before','during','after'}

    def __init__(self):
        self._prev_policy=None
        self._prev_dance_policy=None
        self._time_policy_changed=None # when the new dance step started
        self.shimmy_function=None # used for ramp of shimmy
        self._policy_changed=False
        self.controller=None
        self.cost_function=None # set by each generate_trajectory, used by contained classes, e.g. cartpole_dancer, to access config values
        self.cartpole_dancer=cartpole_dancer()
        self.traj:np.ndarray=None # stores the latest trajectory, for later reference, e.g. for measuring model mismatch
        self.start_time=0
        # we cannot determine if we are running physical cartpole with following method since control is not running at start time.time()
        # if is_physical_cartpole_running_and_control_enabled() else 0 # subtracted from all times for running physical cartpole to avoid numerical overflow
        # log.debug(f'self.start_time={self.start_time:.1f}s')

        self.cartwheel_state=None
        self.cartwheel_time_state_changed = None
        self.set_cartwheel_state('before',self.start_time)
        self.cartwheel_cycles_to_do=0
        self.cartwheel_starttime = 0 # time that this particular cartwheel started
        self.cartwheel_direction=None # 1 for cw, -1 for ccw

    def reset(self):
        """ stops dance if it is going
        """
        self.cartpole_dancer.stop()

    def generate_cartpole_trajectory(self, time: float, state: np.ndarray, controller:template_controller, cost_function: cost_function_base) -> TensorType:
        """ Computes the desired future state trajectory at this time.

        :param time: the scalar time in seconds
        :param horizon: the number of horizon steps
        :param dt: the timestep in seconds
        :param state: the current state of the cartpole as 1d vector

        :returns: the target state trajectory of cartpole, valid for steps that use this trajectory (spin does not use it)
        It is np.ndarray[states,timesteps] with NaN as at least first entries of each state for don't care states, and otherwise the desired future state values.
        The state components are ordered as in CartPole.state_utilities.STATE_VARIABLES

        """
        self.controller=controller
        self.cost_function=cost_function
        dt = gui_default_params.controller_update_interval # TODO fix for physical-cartpole which has CONTROL_PERIOD_MS in globals.py
        # dt=CartPoleMainWindow.CartPoleMainWindowInstance.dt_simulation # todo figure out way to get from single place
        mpc_horizon = controller.optimizer.mpc_horizon
        gui_target_position = cost_function.target_position  # GUI slider position
        gui_target_equilibrium = cost_function.target_equilibrium  # GUI switch +1 or -1 to make pole target up or down position
        # log.debug(f'dt={dt:.3f} mpc_horizon={mpc_horizon}')
        time=time-self.start_time # account for 1970's time returned when running physical-cartpol

        # initialize desired trajectory with NaN entries
        traj = np.full(shape=(state_utilities.NUM_STATES, mpc_horizon),
                       fill_value=np.nan)  # use numpy not tf, too hard to work with immutable tensors

        policy:str = bytes.decode(cost_function.policy.numpy()) # we really want a python str out of the policy so we can compare it using python previous policy
        dancer_current_step=None
        if policy is None:
            raise RuntimeError(f'set policy in config_self.controller.cost_function_wrapper.cost_functions.yml')

        if self._prev_policy is None or policy!=self._prev_policy:
            self._time_policy_changed=time
            self._policy_changed=True
        else:
            self._policy_changed=False
        self._prev_policy=policy

        if self._policy_changed and self._prev_policy=='dance' and policy!='dance':
            self.cartpole_dancer.stop() # stop music

        if policy=='dance':
            if self._policy_changed:
                self.cartpole_dancer.start(time,  self)
            self.cartpole_dancer.step(time=time)

            policy=self.cartpole_dancer.policy
            self._prev_dance_policy=policy


            # must assign for TF/XLA to see the policy number in the compiled cost function. BS...
            cost_function.lib.assign(getattr(cost_function, 'policy_number'),
                                  cost_function.lib.to_variable(self.cartpole_dancer.policy_number, cost_function.lib.int32))
            gui_target_position=self.cartpole_dancer.cartpos
            if policy=='spin':
                cost_function.spin_dir=self.cartpole_dancer.option
                spin_dir_number=1 if cost_function.spin_dir=='cw' else -1
                # hack to get int value for TF cost function use
                cost_function.lib.assign(getattr(cost_function, 'spin_dir_number'),
                                         cost_function.lib.to_variable(spin_dir_number,
                                                                       cost_function.lib.int32))
                # cost_function.spin_freq_hz=self.cartpole_dancer.freq
            elif policy=='balance':
                cost_function.balance_dir=self.cartpole_dancer.option
            elif policy=='shimmy':
                cost_function.shimmy_freq_hz=self.cartpole_dancer.freq
                cost_function.shimmy_amp=self.cartpole_dancer.amp
                cost_function.shimmy_freq2_hz=self.cartpole_dancer.freq2
                cost_function.shimmy_amp2=self.cartpole_dancer.amp2
                cost_function.shimmy_duration=self.cartpole_dancer.endtime
                cost_function.shimmy_dir=self.cartpole_dancer.option

            elif policy=='cartonly':
                cost_function.cartonly_freq_hz=self.cartpole_dancer.freq
                cost_function.cartonly_amp=self.cartpole_dancer.amp
                cost_function.cartonly_duty_cycle=float(self.cartpole_dancer.option)

            elif policy=='cartwheel':
                self.cartwheel_cycles=self.cartpole_dancer.amp


        if policy== 'spin':  # spin pole CW or CCW depending on target_equilibrium up or down
            # spin is handled entirely by a special cost function in cartpole_dancer_cost.py
            pass
            # spin_dir_factor=1
            # if cost_function.spin_dir=='cw':
            #     spin_dir_factor=-1
            # elif cost_function.spin_dir=='ccw':
            #     spin_dir_factor=+1
            # else:
            #     log.warning(f'spin_dir value of "{cost_function.spin_dir} must be "cw" or "ccw"')
            # horizon_endtime = float(mpc_horizon) * dt
            # times = np.linspace(0, horizon_endtime, num=mpc_horizon)
            # rad_per_s_target = spin_dir_factor*gui_target_equilibrium * 2 * np.pi * cost_function.spin_freq_hz # note direction of spin from target_equilibrium
            # rad_per_dt = rad_per_s_target * dt
            # # current_angle = state[state_utilities.ANGLE_IDX]
            # # angle_trajectory=current_angle +  times * rad_per_dt
            # traj[state_utilities.POSITION_IDX] = gui_target_position
            # # traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(angle_trajectory)
            # # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(angle_trajectory)
            # # traj[state_utilities.ANGLE_IDX, :] = angle_trajectory
            # traj[state_utilities.ANGLED_IDX, :] = rad_per_s_target # 1000 rad/s is arbitrary, not sure if this is best target
            # # traj[state_utilities.POSITIOND_IDX, :] = 0
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
            traj[state_utilities.ANGLED_IDX, :] = 0
            traj[state_utilities.POSITIOND_IDX, :] = 0
        elif policy=='shimmy':  # cart follows a desired cart position shimmy while keeping pole up or down
            # # shimmy is an (optional) ramp from freq to freq2 and amp to amp2
            # up_down=1
            # if cost_function.shimmy_dir=='up':
            #     up_down=1
            # elif cost_function.shimmy_dir=='down':
            #     up_down=-1
            # else:
            #     log.warning(f'balance_dir value of "{cost_function.balance_dir} must be "up" or "down"')
            if time>self._time_policy_changed+cost_function.shimmy_duration:
                self._time_policy_changed=time # reset shimmy and start over if doing it from fixed shimmy policy in yml
                log.debug(f'shimmy restarted at time={time}')
            shimmy_starttime=self._time_policy_changed
            # shimmy_endtime=self._time_policy_changed+cost_function.shimmy_duration
            # compute times from current time to end of horizon
            horizon_endtime = time-shimmy_starttime + mpc_horizon * dt
            # time for shimmy must be relative to start of shimmy step for freq ramp to make sense
            times = np.linspace(time-shimmy_starttime, horizon_endtime, num=mpc_horizon)
            # time_frac=times/cost_function.shimmy_duration
            f0 = cost_function.shimmy_freq_hz  # seconds
            a0 = cost_function.shimmy_amp  # meters
            # f1 = cost_function.shimmy_freq2_hz  # seconds
            # a1 = cost_function.shimmy_amp2  # meters

            # compute the shimmmy trajectory over the horizon
            # amps=a0+time_frac*(a1-a0)
            # freqs=f0+time_frac*(f1-f0)

            # print(f'abs_time={time:.2f}, rel_time={times[0]:.2f} time_frac={time_frac[0]:.3f} amp={amps[0]:.3f} freq={freqs[0]:.2f}')

            cartpos=a0*np.sin(2*np.pi*f0*times)
            cartpos_d=np.gradient(cartpos,dt)
            angle=-np.arcsin(cartpos/(2*CARTPOLE_PHYSICAL_CONSTANTS.L)) # compute the angle towards the center of shimmy to keep head of pole fixed as well as possible. L is half of pole length.
            angle_d=np.gradient(angle,dt)

            traj[state_utilities.POSITION_IDX] = gui_target_position + cartpos
            traj[state_utilities.ANGLE_IDX, :] = angle # keep pole pointing towards center of shimmy to minimize pole tip movement
            # traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(target_angle)
            # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
            traj[state_utilities.ANGLED_IDX, :] = angle_d
            traj[state_utilities.POSITIOND_IDX, :] = cartpos_d
        elif policy=='cartonly':  # cart follows the trajectory, pole ignored
            per = 1./cost_function.cartonly_freq_hz  # seconds
            amp = cost_function.cartonly_amp  # meters
            horizon_endtime = time + mpc_horizon * dt
            times = np.linspace(time, horizon_endtime, num=mpc_horizon)
            from scipy.signal import sawtooth
            # sawtooth is out for now to avoid sharp turns at ends
            # cartpos = amp * sawtooth((2 * np.pi / per) * times, width=cost_function.cartonly_duty_cycle)  # width=.5 makes triangle https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html
            cartpos = amp * np.sin((2 * np.pi / per) * times)  # width=.5 makes triangle https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html
            cartvel = np.gradient(cartpos, dt)
            cartpos_vector = gui_target_position + cartpos
            traj[state_utilities.POSITION_IDX] = cartpos_vector
            # target_angle=np.pi * (1-gui_target_equilibrium)/2 # either 0 for up and pi for down
            # traj[state_utilities.ANGLE_COS_IDX, :] = -1 # we must include some pole cost or else the pole can start to spin
            # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
            # traj[state_utilities.ANGLE_IDX, :] = target_angle
            traj[state_utilities.ANGLED_IDX, :] = 0 # we must include some pole cost or else the pole can start to spin
            traj[state_utilities.POSITIOND_IDX, :] = cartvel
            # print(f'\rCARTPOS: time:{time:.1f}s gui_target_position: {gui_target_position*100:.1f}cm target: {cartpos_vector[0]*100:.1f}cm \033[K',end='') # magic string to go to start of line
        elif policy=='cartwheel':
            # cartwheel starts with balance, once balanced the cartwheels start, after the cartwheels we again balance
            cycles=cost_function.cartwheel_cycles
            # determine state transitions
            if self._policy_changed:
                self.set_cartwheel_state('before',time)
                self.cartwheel_cycles_to_do=np.abs(cost_function.cartwheel_cycles)
                self.cartwheel_direction=np.sign(cost_function.cartwheel_cycles)

            if self.cartwheel_state=='before':
                # if before cartwheel, we need to get balanced
                if self.is_pole_balanced(state):
                    self.set_cartwheel_state('during',time)
                    self.cartwheel_starttime=time
            elif self.cartwheel_state=='during':
                # during the cartwheel, we don't go to 'after' until at least time for a cartwheel plus we need to be balanced again
                if time-self.cartwheel_starttime>cost_function.cartwheel_target_duration_s:
                    if self.is_pole_balanced(state):
                        self.cartwheel_cycles_to_do-=1
                        self.cartwheel_starttime = time
                        if self.cartwheel_cycles_to_do==0:
                            self.set_cartwheel_state('after',time)
                    elif time-self.cartwheel_starttime>2*cost_function.cartwheel_target_duration_s:
                        log.debug('gave up trying to do this cartwheel, starting over')
                        self.set_cartwheel_state('before',time)
            elif self.cartwheel_state=='after':
                pass # end state, don't get out of it except by starting a new cartwheel

            if self.cartwheel_state=='before' or self.cartwheel_state=='after':
                traj[state_utilities.POSITION_IDX] = gui_target_position
                traj[state_utilities.ANGLE_COS_IDX, :] = 1
                traj[state_utilities.ANGLED_IDX, :] = 0
                traj[state_utilities.POSITIOND_IDX, :] = 0
            elif self.cartwheel_state=='during':
                traj[state_utilities.POSITION_IDX] = gui_target_position
                traj[state_utilities.POSITIOND_IDX, :] = 0
                # compute times from current time to end of horizon
                horizon_endtime = time - self.cartwheel_starttime + mpc_horizon * dt
                # time for step must be relative to start of step
                times = np.linspace(time - self.cartwheel_starttime, horizon_endtime, num=mpc_horizon)

                target_angle=self.cartwheel_direction*2*np.pi*times/cost_function.cartwheel_target_duration_s
                traj[state_utilities.ANGLE_IDX, :] = target_angle
                target_angled=np.gradient(target_angle,dt)
                traj[state_utilities.ANGLED_IDX, :] = target_angled

        else:
            log.error(f'cost policy "{policy}" is unknown')

        self.set_gui_status_text(time, state, cost_function) # update CartPoleSimulation GUI status line (if it exists)
        self.traj=traj
        return traj

    def set_cartwheel_state(self,new_state:str,time:float):
        """Sets the cartwheel state and the time the state changed if it did
        """
        if self.cartwheel_state is None or self.cartwheel_state!=new_state:
            self.cartwheel_time_state_changed=time
            log.info(f'changed state from {self.cartwheel_state}->{new_state} at t={time:.3f}')
        self.cartwheel_state=new_state

    def is_pole_balanced(self, state):
        """ Returns True if angle and angle_d are sufficiently small"""
        return np.abs(state[state_utilities.ANGLE_IDX])<self.cost_function.cartwheel_balance_angle_limit_deg*(np.pi/180)\
                and np.abs(state[state_utilities.ANGLED_IDX])<self.cost_function.cartwheel_balance_angled_limit_deg_per_sec*(np.pi/180)

    def set_gui_status_text(self, time: float, state: np.ndarray, cost_function: cost_function_base) -> None:
        s = self.get_status_string(time, state, cost_function)
        CartPoleMainWindow.set_status_text(s)

    def get_status_string(self, time, state, cost_function):
        """ Returns a string describing current objective trajectory or control target"""
        policy: str = cost_function.policy
        if policy == 'dance':
            return  # status of cartpole GUI set by dancer
        gui_target_position = float(cost_function.target_position)  # GUI slider position
        gui_target_equilibrium = float(
            cost_function.target_equilibrium)  # GUI switch +1 or -1 to make pole target up or down position

        if policy == 'balance':
            dir = self.decode_string(cost_function.balance_dir)
            s = f'Policy: balance/{dir} pos={gui_target_position:.1f}m'
        elif policy == 'spin':
            dir = self.decode_string(cost_function.spin_dir)
            s = f'Policy: spin/{dir}*up/down pos={gui_target_position:.1f}m'
        elif policy == 'shimmy':
            s = f'Policy: shimmy pos={gui_target_position:.1f}m freq={float(cost_function.shimmy_freq_hz):.1f}Hz amp={float(cost_function.shimmy_amp):.1f}Hz'
        elif policy == 'cartonly':
            s = f'Policy: cartonly pos={gui_target_position:.1f}m freq={float(cost_function.cartonly_freq_hz):.1f}Hz amp={float(cost_function.cartonly_amp):.1f}Hz'
        elif policy == 'cartwheel':
            s = f'Policy: cartwheel pos={gui_target_position:.1f}m state={self.cartwheel_state} cycles_to_do={self.cartwheel_cycles_to_do}'
        else:
            s = f'unknown/not implemented string'
        return s


    def decode_string(self,tfstring:tf.Variable):
        """ Decodes tensorflow string to unicode python str.
        :param tfstring: the tf.Variable with dtype 'string'.

        :returns: If tfstring is tf.Variable, returns tfstring.numpy().decode('utf-8'). If tfstring is already str, returns result of tfstring.encode().decode('utf-8')
        """
        if isinstance(tfstring,str): return tfstring.encode().decode('utf-8')
        return tfstring.numpy().decode('utf-8')

    def keyboard_input(self, c):
        if c == 'M':  # switch dance step (mode)
            newpol=self.next_policy()
            update_attributes({'policy':newpol},self.cost_function)

    def next_policy(self)->str:
        curpolicy = self.decode_string(self.cost_function.policy)
        ind=0
        try:
            ind=cartpole_trajectory_generator.POLICIES.index(curpolicy)+1
            if ind==len(cartpole_trajectory_generator.POLICIES): ind=0
        except ValueError as e:
            log.error(f'error cycling policy {e}')
        newpol=cartpole_trajectory_generator.POLICIES_NUMBERED[ind]
        print(f'changed policy from {curpolicy} -> {newpol}')
        return newpol

    def print_keyboard_help(self):
        print('M cycle dance policy')
