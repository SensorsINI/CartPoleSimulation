# import time
# from pydoc import text
from typing import List, Optional, Union

import numpy as np
# import scipy
# from matplotlib.pyplot import pause
from scipy.signal import sawtooth, square

from CartPole import state_utilities
from CartPole.state_utilities import NUM_STATES, ANGLED_IDX, ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, POSITION_IDX, POSITIOND_IDX
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Cost_Functions import cost_function_base
from Control_Toolkit_ASF.Cost_Functions.CartPole.cartpole_dancer import cartpole_dancer
from GUI import gui_default_params, CartPoleMainWindow
from SI_Toolkit.computation_library import TensorType
import tensorflow as tf

from Control_Toolkit.others.get_logger import get_logger
from others.globals_and_utils import update_attributes
from others.p_globals import CARTPOLE_PHYSICAL_CONSTANTS

log = get_logger(__name__)

def to_numpy(v):
    return v if not isinstance(v,(tf.Tensor,tf.Variable)) else v.numpy()

class cartpole_trajectory_generator:

    POLICIES=('balance','spin','shimmy','cartonly','cartwheel','toandfro')
    POLICIES_NUMBERED=('balance0','spin1','shimmy2','cartonly3','cartwheel4','toandfro5') # for tensorflow scalar BS
    CARTWHEEL_STATES={'before','starting','during','after'}

    def __init__(self):

        self.last_status_text:str=None
        self.step_start_time = None
        self.policy_changed=False
        self.prev_policy=None
        self.prev_dance_policy=None
        self.time_policy_changed=None # when the new dance step started
        self.shimmy_function=None # used for ramp of shimmy
        self.controller=None
        self.cost_function=None # set by each generate_trajectory, used by contained classes, e.g. cartpole_dancer, to access config values
        self.lib=None
        self.cartpole_dancer=cartpole_dancer()
        self.traj:np.ndarray=None # stores the latest trajectory, for later reference, e.g. for measuring model mismatch
        self.start_time=0

        self.cartwheel_cycles = None
        self.cartwheel_cycles_to_do:int = None
        self.cartwheel_state:str = None
        self.cartwheel_time_state_changed:float = None
        self.cartwheel_starttime:float = None
        self.cartwheel_direction:int = None
        self.last_cartwheel_cycles:int=None

        self.mpc_horizon=None
        """ the horizion, updated from cost function on every call of generate_cartpole_trajectory"""

        self.traj:np.ndarray=None
        """ The state vector trajectories to follow. It is a 2d array [state,horizon] where first dimension are the states and 2nd is the MPC horizon"""

        self.traj_states_to_consider:List[int]=[]
        """ a list of state indices to count difference over for cost computation. It is put to cost_function as a tf.Variable so that the compiled cost function can use it."""

        self.state_cost_weights=None
        """ Vector of state trajectory cost weights from config_cost_functions. Filled in on every generate_cartpole_trajectory()."""

        # we cannot determine if we are running physical cartpole with following method since control is not running at start time.time()
        # if is_physical_cartpole_running_and_control_enabled() else 0 # subtracted from all times for running physical cartpole to avoid numerical overflow
        # log.debug(f'self.start_time={self.start_time:.1f}s')

        self.reset_cartwheel_state()

    def reset_cartwheel_state(self):
        self.cartwheel_state = 'before'
        self.cartwheel_time_state_changed = 0
        self.set_cartwheel_state('before', self.start_time)
        self.cartwheel_cycles_to_do = 0
        if not self.cost_function is None and hasattr(self.cost_function,'cartwheel_cycles'):
            self.cartwheel_cycles_to_do=np.abs(to_numpy(self.cost_function.cartwheel_cycles))
        self.cartwheel_starttime = 0  # time that this particular cartwheel started
        self.cartwheel_direction = 1  # 1 for ccw (increasing angle), -1 for cw
        self.last_cartwheel_cycles=0 # to test if config file value changes so we can restart

    def reset(self):
        """ stops dance if it is going
        """
        self.cartpole_dancer.stop()
        self.reset_cartwheel_state()


    def generate_cartpole_trajectory(self, time: float, state: np.ndarray, controller:template_controller, cost_function: cost_function_base):
        """ Computes the desired future state trajectory at this time.
        Sets the cost function attributes target_trajectory and traj_states_to_consider.
        The target state trajectory of cartpole is valid for steps that use this trajectory (spin does not use it)
        It is np.ndarray[states,timesteps] where only considered states are in the 2d array. The first dimension is the state, 2nd is the mpc_horizon.

        :param time: the scalar time in seconds
        :param horizon: the number of horizon steps
        :param dt: the timestep in seconds
        :param state: the current state of the cartpole as 1d vector of current state components, e.g. the scalar angle state[state_utilities.ANGLE]

       """
        self.controller=controller
        self.cost_function=cost_function
        self.lib=self.cost_function.lib
        dt = gui_default_params.controller_update_interval # TODO fix for physical-cartpole which has CONTROL_PERIOD_MS in globals.py
        # dt=CartPoleMainWindow.CartPoleMainWindowInstance.dt_simulation # todo figure out way to get from single place
        self.mpc_horizon = to_numpy(controller.optimizer.mpc_horizon)
        gui_target_position = to_numpy(cost_function.target_position)  # GUI slider position
        gui_target_equilibrium = to_numpy(cost_function.target_equilibrium)  # GUI switch +1 or -1 to make pole target up or down position
        # log.debug(f'dt={dt:.3f} mpc_horizon={self.mpc_horizon}')
        time=time-self.start_time # account for 1970's time returned when running physical-cartpol



        policy:str = bytes.decode(to_numpy(cost_function.policy)) # we really want a python str out of the policy so we can compare it using python previous policy
        dancer_current_step=None
        if policy is None:
            raise RuntimeError(f'set policy in config_self.controller.cost_function_wrapper.cost_functions.yml')

        if self.prev_policy is None or policy!=self.prev_policy:
            self.time_policy_changed=time
            self.policy_changed=True
        else:
            self.policy_changed=False
        self.prev_policy=policy

        if self.policy_changed and self.prev_policy== 'dance' and policy!= 'dance':
            self.cartpole_dancer.stop() # stop music

        # ****************************
        # if we are following a complete dance from CSV file, then set the appropriate fields here from the CSV columns
        if policy=='dance':
            if self.policy_changed:
                self.cartpole_dancer.start(time,  self)
            self.cartpole_dancer.step(time=time)

            policy=self.cartpole_dancer.policy
            self.prev_dance_policy=policy

            gui_target_position=to_numpy(self.cartpole_dancer.cartpos)

            # must assign for TF/XLA to see the policy number in the compiled cost function. BS...
            self.lib.assign(getattr(cost_function, 'policy_number'),
                                 self.lib.to_variable(self.cartpole_dancer.policy_number, cost_function.lib.int32))
            if policy=='spin':
                spin_dir=self.cartpole_dancer.option
                # hack to get int value for TF cost function use, is referred to in cartpole_dancer_cost as self.spin_dir
                cost_function.lib.assign(getattr(cost_function, 'spin_dir'),
                                         cost_function.lib.to_variable(spin_dir,
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
                cost_function.cartwheel_cycles=self.cartpole_dancer.amp
            elif policy=='toandfro':
                cost_function.cartwheel_cycles=self.cartpole_dancer.amp
            else:
                log.error(f"policy '{policy}' is unknown")


        # **********************************
        # Now set the state trajectories to follow and for each type of dance step movement
        self.clear_traj()
        if policy== 'spin':  # spin pole CW or CCW depending on target_equilibrium up or down
            # spin is handled entirely by a special cost function in cartpole_dancer_cost.py
            pass
        elif policy=='balance':  # balance upright or down at desired cart position
            balance_dir = gui_target_equilibrium*to_numpy(cost_function.balance_dir)
            target_angle = np.pi * (1 - balance_dir) / 2  # either 0 for up and pi for down
            self.add_traj(POSITION_IDX,gui_target_position)
            # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
            if balance_dir>0:
                self.add_traj(ANGLE_IDX,0) # for up, target angle zero, which is increases linearly with angle, unlike cos which only goes away from 1 like 1-angle^2
            else:
                self.add_traj(ANGLE_COS_IDX,-1) # for down, avoid the 2pi cut ot +pi/-pi by targetting cos to as negative as possible

            self.add_traj(ANGLED_IDX,0)
            self.add_traj(POSITIOND_IDX,0)
        elif policy=='shimmy':  # cart follows a desired cart position shimmy while keeping pole up or down
            f0 = to_numpy(cost_function.shimmy_freq_hz)  # seconds
            a0 = to_numpy(cost_function.shimmy_amp)  # meters
            if self.policy_changed or self.step_start_time is None:
                log.debug(f'shimmy with freq={f0}Hz and amp={a0}m restarted at time={time}')
                self.step_start_time=time
            # shimmy_endtime=self._time_policy_changed+cost_function.shimmy_duration
            # compute times from current time to end of horizon
            time_since_step_started=time-self.step_start_time
            horizon_endtime =  time_since_step_started + self.mpc_horizon * dt
            # time for shimmy must be relative to start of shimmy step for freq ramp to make sense
            times = np.linspace(time_since_step_started, horizon_endtime, num=self.mpc_horizon, dtype=np.float32)
            # time_frac=times/cost_function.shimmy_duration
            # f1 = cost_function.shimmy_freq2_hz  # seconds
            # a1 = cost_function.shimmy_amp2  # meters

            # compute the shimmmy trajectory over the horizon
            # amps=a0+time_frac*(a1-a0)
            # freqs=f0+time_frac*(f1-f0)

            # print(f'abs_time={time:.2f}, rel_time={times[0]:.2f} time_frac={time_frac[0]:.3f} amp={amps[0]:.3f} freq={freqs[0]:.2f}')

            cartpos=a0*np.sin(2*np.pi*f0*times, dtype=np.float32)
            cartpos_d=np.gradient(cartpos,dt)
            angle=np.arcsin(cartpos/(2*CARTPOLE_PHYSICAL_CONSTANTS.L), dtype=np.float32)/2 # compute the angle towards the center of shimmy to keep head of pole fixed as well as possible. L is half of pole length.
            # we divide by 2 to reduce the required angle a bit more
            angle_d=np.gradient(angle,dt)
            self.add_traj(POSITION_IDX, gui_target_position + cartpos)
            self.add_traj(ANGLE_IDX,angle) # keep pole pointing towards center of shimmy to minimize pole tip movement
            self.add_traj(ANGLED_IDX,angle_d)
            self.add_traj(POSITIOND_IDX, cartpos_d)
        elif policy=='cartonly':  # cart follows the trajectory, pole ignored
            f0 = to_numpy(cost_function.cartonly_freq_hz)  # seconds
            a0 = to_numpy(cost_function.cartonly_amp)  # meters
            if self.policy_changed or self.step_start_time is None:
                log.debug(f'cartonly with freq={f0}Hz and amp={a0}m restarted at time={time}')
                self.step_start_time=time
            # shimmy_endtime=self._time_policy_changed+cost_function.shimmy_duration
            # compute times from current time to end of horizon
            time_since_step_started=time-self.step_start_time
            horizon_endtime =  time_since_step_started + self.mpc_horizon * dt
            # time for shimmy must be relative to start of shimmy step for freq ramp to make sense
            times = np.linspace(time_since_step_started, horizon_endtime, num=self.mpc_horizon, dtype=np.float32)
            # sawtooth is out for now to avoid sharp turns at ends
            # cartpos = a0 * square((2 * np.pi * f0) * times, duty=cost_function.cartonly_duty_cycle)  # duty=.5 makes square with equal halfs https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html
            cartpos = a0 * sawtooth((2 * np.pi * f0) * times, width=to_numpy(cost_function.cartonly_duty_cycle))  # width=.5 makes triangle https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html
            # cartpos = a0 * np.sin((2 * np.pi * f0) * times)  # width=.5 makes triangle https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html
            cartvel = np.gradient(cartpos, dt)
            cartpos_vector = gui_target_position + cartpos
            self.add_traj(ANGLE_COS_IDX,-1)
            self.add_traj(POSITION_IDX,cartpos_vector)
            self.add_traj(POSITIOND_IDX, cartvel)
        elif policy=='cartwheel' or policy=='toandfro':
            # cartwheel starts with balance, once balanced the cartwheels start, after the cartwheels we again balance
            # toandfro swings one way and then the other immediately, must have cycles>2

            # determine state transitions based on current state
            angle = state[state_utilities.ANGLE_IDX]
            angle_d = state[state_utilities.ANGLED_IDX]

            # need to convert cartwheel cycles to real float since equals test on tf.Variable does not work if object does not change like it does not with assignment in update_attributes.
            # also cartwheel cycles can be set to float during dance above
            self.cartwheel_cycles=to_numpy(cost_function.cartwheel_cycles)
            if self.policy_changed or self.cartwheel_cycles!=self.last_cartwheel_cycles:
                self.cartwheel_direction=np.sign(self.cartwheel_cycles)
                self.set_cartwheel_state('before',time)
                self.cartwheel_cycles_to_do=np.abs(self.cartwheel_cycles)
                self.last_cartwheel_cycles=self.cartwheel_cycles

            # first update state depending on cartwheel state, cartpole state, and time
            if self.cartwheel_state=='before':
                # if before cartwheel, we need to get balanced
                if self.cartwheel_cycles_to_do>0 and self.is_pole_balanced(state):
                    self.set_cartwheel_state('starting',time)
                    self.cartwheel_starttime=time
            elif self.cartwheel_state=='starting':
                # start the free fall in the correct direction using angle_d indicator function
               # if pole is falling in correct direction and fast enough and has fallen enough move to free fall
                if np.abs(angle_d)>to_numpy(self.cost_function.cartwheel_freefall_angle_limit_deg)*np.pi/180\
                        and np.sign(angle_d)==self.cartwheel_direction \
                        and np.abs(angle)>to_numpy(self.cost_function.cartwheel_freefall_angled_limit_deg_per_sec)*np.pi/180:
                    self.cartwheel_cycles_to_do-=1
                    self.set_cartwheel_state('during',time)
            elif self.cartwheel_state=='during':
                # during the cartwheel, we don't go to 'after' until the angle has exceeded some angle and spin speed
                if np.abs(angle)>to_numpy(self.cost_function.cartwheel_freefall_to_after_angle_limit_deg)*np.pi/180:
                    if self.cartwheel_cycles_to_do>0:
                        self.set_cartwheel_state('before',time)
                        if policy=='toandfro':
                            self.cartwheel_direction= -self.cartwheel_direction # go other direction this time
                    else:
                        self.set_cartwheel_state('after',time)
            elif self.cartwheel_state=='after':
                pass # end state, don't get out of it except by starting a new cartwheel

            # now set the target trajectory objectives according to state
            if self.cartwheel_state=='before' or self.cartwheel_state=='after':  # just get balanced again
                self.add_traj(POSITION_IDX, gui_target_position)
                self.add_traj(ANGLE_IDX, 0)
                self.add_traj(ANGLED_IDX, 0)
                self.add_traj(POSITIOND_IDX, 0)

            elif self.cartwheel_state=='starting': # we just try to get the pole spinning the correct direction
                self.add_traj(POSITION_IDX, gui_target_position)
                self.add_traj(POSITIOND_IDX, 0)
                self.add_traj(ANGLED_IDX,  1e6*self.cartwheel_direction)

            elif self.cartwheel_state=='during': # free fall, just get the cart back to target position
                self.add_traj(POSITION_IDX, gui_target_position)
                self.add_traj(POSITIOND_IDX, 0)

        else:
            log.error(f'cost policy "{policy}" is unknown')


        self.compile_traj_for_cost_function()

        self.set_status_text(time, state, cost_function) # update CartPoleSimulation GUI status line (if it exists)


    def add_traj(self, idx:int, traj:Union[float,np.ndarray]):
        """ Add a target trajectory vector to the target states trajectories.

        :param idx: the state index, e.g. POSITION
        :param traj: either a float value or a vector with length self.mpc_horizon
        """
        traj_vec=None

        self.traj_states_to_consider.append(idx)

        if isinstance(traj,(float,int, np.float32)):
            traj_vec=np.full(shape=(1,self.mpc_horizon),fill_value=traj,dtype=np.float32)
        elif isinstance(traj, np.ndarray):
            if len(traj.shape)>1 or traj.shape[0]!=self.mpc_horizon:
                raise AttributeError(f'trajectory {traj} must be a vector with length self.mpc_horizon')
            traj_vec=traj
        else:
            raise AttributeError(f'trajectory vector {traj} must be a scalar int,float value or np.ndarray or tf.Tensor')

        self.traj[idx,:]=traj_vec # put this vector to 2d array of trajectories

    def clear_traj(self):
        """ Clear the target state trajectories"""
        # initialize desired trajectory with 0 entries, will fill appropriate rows with target state trajectories

        self.traj = np.full(shape=(NUM_STATES, self.mpc_horizon),
                       fill_value=0., dtype=np.float32)  # use numpy not tf, too hard to work with immutable tensors
        self.traj_states_to_consider=[]

    def compile_traj_for_cost_function(self):
        """ Compile the target state trajectories to a fixed-size tensor is used in cartpole_dancer_cost.py to
         measure the rollout costs. Transfer the target states trajectory and effective cost weighting vector to cost function.
         """
        # states are "angle", "angleD", "angle_cos", "angle_sin", "position", "positionD"
        raw_state_cost_weights = np.array([to_numpy(self.cost_function.pole_angle_weight),
                                   to_numpy(self.cost_function.pole_swing_weight),
                                    to_numpy(self.cost_function.pole_angle_weight),
                                    to_numpy(self.cost_function.pole_angle_weight),
                                    to_numpy(self.cost_function.cart_pos_weight),
                                    to_numpy(self.cost_function.cart_vel_weight)])
        effective_cost_weights=np.zeros([1,NUM_STATES])
        np.put(effective_cost_weights, self.traj_states_to_consider,raw_state_cost_weights[self.traj_states_to_consider])


        updated_attributes = {}  # empty dict
        updated_attributes['target_trajectory'] = tf.Variable(self.traj, dtype=tf.float32)
        updated_attributes['effective_traj_cost_weights'] = tf.Variable(effective_cost_weights, dtype=tf.float32)
        update_attributes(updated_attributes, self.cost_function)  # update the cost_fuction tf.Variable's to give access to them in compiled cost_function

    def set_cartwheel_state(self,new_state:str,time:float):
        """Sets the cartwheel state and the time the state changed if it did
        """
        if self.cartwheel_state is None or self.cartwheel_state!=new_state:
            self.cartwheel_time_state_changed=time
            log.debug(f'changed state from {self.cartwheel_state}->{new_state} at t={time:.3f}')
        self.cartwheel_state=new_state

    def is_pole_balanced(self, state):
        """ Returns True if angle and angle_d are sufficiently small"""
        return np.abs(state[state_utilities.ANGLE_IDX])<to_numpy(self.cost_function.cartwheel_balance_angle_limit_deg)*(np.pi/180)\
                and np.abs(state[state_utilities.ANGLED_IDX])<to_numpy(self.cost_function.cartwheel_balance_angled_limit_deg_per_sec)*(np.pi/180)

    def set_status_text(self, time: float, state: np.ndarray, cost_function: cost_function_base) -> None:
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
            dir = to_numpy(cost_function.balance_dir)
            s = f'Policy: balance/{dir} pos={gui_target_position:.1f}m'
        elif policy == 'spin':
            dir_int = cost_function.spin_dir.numpy()
            dir='cw' if dir_int<0 else 'ccw'
            s = f'Policy: spin/{dir}*up/down pos={gui_target_position:.1f}m'
        elif policy == 'shimmy':
            s = f'Policy: shimmy pos={gui_target_position:.1f}m freq={float(cost_function.shimmy_freq_hz):.1f}Hz amp={float(cost_function.shimmy_amp):.3f}m'
        elif policy == 'cartonly':
            s = f'Policy: cartonly pos={gui_target_position:.1f}m freq={float(cost_function.cartonly_freq_hz):.1f}Hz amp={float(cost_function.cartonly_amp):.3f}m'
        elif policy == 'cartwheel':
            s = f'Policy: cartwheel pos={gui_target_position:.1f}m state={self.cartwheel_state} cycles_to_do={self.cartwheel_cycles_to_do} angle={state[state_utilities.ANGLE_IDX]*180/np.pi:.1f}deg angle_d={state[state_utilities.ANGLED_IDX]*180/np.pi:.1f}deg/s'
        elif policy == 'toandfro':
            s = f'Policy: toandfro pos={gui_target_position:.1f}m state={self.cartwheel_state} cycles_to_do={self.cartwheel_cycles_to_do} angle={state[state_utilities.ANGLE_IDX]*180/np.pi:.1f}deg angle_d={state[state_utilities.ANGLED_IDX]*180/np.pi:.1f}deg/s'
        else:
            s = f'unknown/not implemented string'
        self.last_status_text=s
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
