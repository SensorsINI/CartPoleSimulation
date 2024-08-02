# -*- coding: utf-8 -*-
"""
CartPole Class:
The file holds the main class of our simulator - it corresponds to a physical cartpole.
You can find here methods with performing experiment, saving data,
and many more. To run it needs some "environment": we provide you with GUI and data_generator
@author: Marcin
"""
# Import module to save history of the simulation as csv file

# Import module to interact with OS
import traceback
# Import module to get a current time and date used to name the files containing the history of simulations
import timeit
# To detect the latest csv file

import numpy as np
import pandas as pd

from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.environment import EnvironmentBatched
from Control_Toolkit.others.globals_and_utils import (
    get_available_controller_names, get_available_optimizer_names, get_controller_name, get_optimizer_name, import_controller_by_name)
from others.globals_and_utils import MockSpace, create_rng, load_config
from CartPole.cartpole_parameters import (J_fric, L, m_cart, M_fric, TrackHalfLength,
                                          controlBias, controlDisturbance, CP_PARAMETERS_DEFAULT,
                                          g, k, m_pole, u_max, v_max, controlDisturbance_mode)
# Interpolate function to create smooth random track
# Run range() automatically adding progress bar in terminal
from tqdm import trange

from CartPole._CartPole_mathematical_helpers import wrap_angle_rad

from CartPole.latency_adder import LatencyAdder
from CartPole.load import get_full_paths_to_csvs, load_csv_recording
from CartPole.noise_adder import NoiseAdder
from CartPole.noise_control_signal import add_control_noise
from CartPole.random_target_generator import Generate_Random_Trace_Function
from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                      ANGLED_IDX, POSITION_IDX, POSITIOND_IDX)
from CartPole.state_utilities import create_cartpole_state
from CartPole.summary_plots import summary_plots

s0 = create_cartpole_state()

# region Imported modules

# check memory usage of chosen methods. Commented by default
# from memory_profiler import profile



from random import random

# Angle convention to rotate the mast in right direction - depends on used Equation
from CartPole.cartpole_equations import CartPoleEquations

from CartPole.csv_logger import create_csv_file_name, create_csv_title, create_csv_header, create_csv_file, save_data_to_csv_file

from SI_Toolkit.Functions.FunctionalDict import FunctionalDict, HistoryClass


# endregion

# endregion

config = load_config("cartpole_physical_parameters.yml")
PATH_TO_EXPERIMENT_RECORDINGS_DEFAULT = config["cartpole"]["PATH_TO_EXPERIMENT_RECORDINGS_DEFAULT"]

rng = create_rng(__name__, config["cartpole"]["seed"])


class CartPole(EnvironmentBatched):
    num_states = 6
    num_acions = 1

    def __init__(self, initial_state=s0, path_to_experiment_recordings=None, target_slider=None):

        self.config = config["cartpole"]
        self.rng_CartPole = create_rng(self.__class__.__name__, self.config["seed"])

        self.slider = target_slider

        self.time_L_last_change = None
        self.start_changing_L = False

        if path_to_experiment_recordings is None:
            self.path_to_experiment_recordings = PATH_TO_EXPERIMENT_RECORDINGS_DEFAULT
        else:
            self.path_to_experiment_recordings = path_to_experiment_recordings

        # Global time of the simulation
        self.time = 0.0

        # CartPole is initialized with state, control input, target position all zero
        # This is however usually changed before running the simulation. Treat it just as placeholders.
        # Container for the augmented state (angle, position and their first and second derivatives)of the cart
        self.s = initial_state  # (s like state)
        self.angleDD = self.positionDD = 0.0
        
        # Variables for control input and target position.
        self.u = 0.0  # Physical force acting on the cart
        self.Q_applied = 0.0
        self.Q_calculated = 0.0
        self.Q = 0.0  # Dimensionless motor power in the range [-1,1] from which force is calculated with Q2u() method

        self.cpe = CartPoleEquations(numba_compiled=True)

        self.action_space = MockSpace(-1.0, 1.0, (1,), np.float32)
        state_low = [-np.pi, -np.inf, -1.0, -1.0, -TrackHalfLength, -np.inf]
        state_high = [-v for v in state_low]
        self.observation_space = MockSpace(state_low, state_high, (6,), np.float32)

        self.L_initial = float(L)

        self.change_L_every_x_second = np.inf
        self.time_last_L_change = None
        self.L_discount_factor = 1.0
        self.L_range = [0.03, 0.2]
        self.L_informed_controller = True
        if self.L_informed_controller:
            self.L_for_controller = L
        else:
            self.L_for_controller = float(self.L_initial)
        self.L_change_mode = 'step'
        self.L_step = 0.02

        self.latency = self.config["latency"]
        self.LatencyAdderInstance = LatencyAdder(latency=self.latency, dt_sampling=0.002)
        self.NoiseAdderInstance = NoiseAdder()
        self.s_with_noise_and_latency = np.copy(self.s)
        self.zero_angle_shift_init = np.deg2rad(self.config['zero_angle_shift']['init'])
        self.zero_angle_shift = self.zero_angle_shift_init
        self.zero_angle_shift_mode = self.config['zero_angle_shift']['mode']
        self.zero_angle_shift_increment = np.deg2rad(self.config['zero_angle_shift']['increment'])

        # region Time scales for simulation step, controller update and saving data
        # See last paragraph of "Time scales" section for explanations
        # ∆t in number of steps (related to simulation time step)
        # is set while setting corresponding dt through @property
        self.dt_controller_number_of_steps = 0
        self.dt_save_number_of_steps = 0

        # Counts time steps from last controller update or saving
        # is set while setting corresponding dt through @property
        self.dt_controller_steps_counter = 0
        self.dt_save_steps_counter = 0

        # Helper variables to set timescales
        self._dt_simulation = None
        self._dt_controller = None
        self._dt_save = None

        self.dt_simulation = None  # s, Update CartPole dynamical state every dt_simulation seconds
        self.dt_controller = None  # s, Recalculate control input every dt_controller_default seconds
        self.dt_save = None  # s,  Save CartPole state every dt_save_default seconds
        # endregion

        # region Variables controlling operation of the program - can be modified directly from CartPole environment
        self.rounding_decimals = np.inf  # Sets number of digits after coma to save in experiment history for each feature, make it np.inf to skip rounding entirely
        self.save_data_in_cart = True  # Decides whether to store whole data of the experiment in dict_history or not
        self.stop_at_90 = False  # If true pole is blocked after reaching the horizontal position
        # endregion

        # region Variables controlling operation of the program - should not be modified directly
        self.save_flag = False  # Signalizes that the current time step should be saved
        self.csv_filepath = None  # Where to save the experiment history.
        self.controller = None  # Placeholder for the currently used controller function
        self.controller_name = ''  # Placeholder for the currently used controller name
        self.optimizer_name = ''  # Placeholder for the currently used optimizer name
        self.controller_idx = None  # Placeholder for the currently used controller index
        self.optimizer_idx = None  # Placeholder for the currently used optimizer index
        self.controller_names = get_available_controller_names()  # list of controllers available in controllers folder
        self.optimizer_names = get_available_optimizer_names()  # list of controllers available in controllers folder
        # endregion

        # region Variables for generating experiments with random target trace
        # Parameters for random trace generation
        # These need to be set, before CartPole can generate random trace and random experiment
        self.track_relative_complexity = None  # randomly placed target points/s
        self.length_of_experiment = None  # seconds, length of the random length trace
        self.interpolation_type = None  # Sets how to interpolate between turning points of random trace
        # Possible choices: '0-derivative-smooth', 'linear', 'previous'
        # '0-derivative-smooth'
        #       -> turning points are connected with smooth interpolation curve having derivative = 0 at each turning p.
        # 'linear' -> turning points are connected with line segments
        # 'previous' -> between two turning points the value of the preceding point is kept constant.
        #                   In this last setting endpoint if set has no visible effect
        #                       (may however appear in the last line of the recording - TODO: not checked)
        self.turning_points_period = None  # How turning points should be distributed
        # Possible choices: 'regular', 'random'
        # Regular means that they are equidistant from each other
        # Random means we pick randomly points at time axis at which we place turning points
        # Where the target position of the random experiment starts and end:
        self.start_random_target_position_at = None
        self.end_random_target_position_at = None
        # Alternatively you can provide a list of target positions.
        # e.g. self.turning_points = [10.0, 0.0, 0.0]
        # If not None this variable has precedence -
        # track_relative_complexity, start/end_random_target_position_at_globals have no effect.
        self.turning_points = None

        # Set the max for smoothly interpolated random target position to avoid bumping into track ends.
        self.used_track_fraction = None

        self.random_track_f = None  # Function interpolataing the random target position between turning points
        self.new_track_generated = False  # Flag informing that a new target position track is generated
        self.number_of_timesteps_in_random_experiment = None
        self.use_pregenerated_target_position = False  # Informs method performing experiment
        #                                                    not to take target position from environment
        # endregion and

        self.variables_to_log = FunctionalDict(
            {
                'time': lambda: self.time,

                'angle': lambda: self.s[ANGLE_IDX],
                'angleD': lambda: self.s[ANGLED_IDX],
                'angleDD': lambda: self.angleDD,
                'angle_cos': lambda: self.s[ANGLE_COS_IDX],
                'angle_sin': lambda: self.s[ANGLE_SIN_IDX],
                'position': lambda: self.s[POSITION_IDX],
                'positionD': lambda: self.s[POSITIOND_IDX],
                'positionDD': lambda: self.positionDD,

                'Q_calculated': lambda: self.Q_calculated,
                'Q_applied': lambda: self.Q_applied,
                'u': lambda: self.u,

                # The target_position is not always meaningful
                # If it is not meaningful all values in this column are set to 0
                'target_position': lambda: self.target_position,
                'target_equilibrium': lambda: self.target_equilibrium,

                'L': lambda: float(L),

                'Q_update_time': lambda: self.Q_update_time,

            }
        )

        self.dict_history = HistoryClass()  # Class holding the history of the simulation
        self.dict_history.add_keys(keys=self.variables_to_log.keys())
        
        self.target_position = 0.0
        self.target_equilibrium = 1.0

        self.keep_target_equilibrium_x_seconds_up = np.inf
        self.keep_target_equilibrium_x_seconds_down = np.inf
        self.time_last_target_equilibrium_change = None

        self.Q_update_time = None

        # region Initialize CartPole in manual-stabilization mode
        self.set_controller(controller_name='manual-stabilization')
        # endregion
        
    # region 1. Methods related to dynamic evolution of CartPole system

    # This method changes the internal state of the CartPole
    # from a state at time t to a state at t+dt
    # We assume this function is called for the first time to calculate first time step
    # @profile(precision=4)
    def update_state(self):

        self.update_parameters()

        # Update the total time of the simulation
        self.step_time()

        # Update target position depending on the mode of operation
        self.update_target_position()

        self.update_target_equilibrium()

        # Calculate the next state
        self.cartpole_integration()

        # Calculate the correction to the state due to the bounce at the edge if applies
        self.edge_bounce()

        # stop pole at +/- 90 deg if enabled
        block_pole_at_90 = self.block_pole_at_90_deg()

        # Calculate cosine and sine
        self.update_cos_and_sin()

        # Wrap angle to +/-π
        self.wrap_angle()

        self.add_noise_and_latency()

        # Determine the dimensionless [-1,1] value of the motor power Q
        self.Update_Q()

        # Convert dimensionless motor power to a physical force acting on the Cart
        self.Q2u()

        # Update second derivatives
        self.cartpole_ode()

        if block_pole_at_90:
            self.angleDD = 0.0

        self.save_csv_routine()

    def step_time(self):
        self.time = self.time + self.dt_simulation

    def update_cos_and_sin(self):
        self.s[ANGLE_COS_IDX] = np.cos(self.s[ANGLE_IDX])
        self.s[ANGLE_SIN_IDX] = np.sin(self.s[ANGLE_IDX])

    def wrap_angle(self):
        self.s[ANGLE_IDX] = wrap_angle_rad(self.s[ANGLE_IDX])

    def add_noise_and_latency(self):
        self.LatencyAdderInstance.add_current_state_to_latency_buffer(self.s)
        s_delayed = self.LatencyAdderInstance.get_interpolated_delayed_state()
        self.s_with_noise_and_latency = self.NoiseAdderInstance.add_noise_to_measurement(s_delayed, copy=False)
        self.s_with_noise_and_latency = self.update_zero_angle_shift(self.s_with_noise_and_latency)

    def cartpole_ode(self):
        self.angleDD, self.positionDD = self.cpe.cartpole_ode_interface(self.s, self.u, L=float(L))

    def Q2u(self):
        self.u = self.cpe.Q2u(self.Q)

    def update_zero_angle_shift(self, s):
        if self.zero_angle_shift_mode == 'constant':
            da = 0.0
        elif self.zero_angle_shift_mode == 'random walk':
            da  = (1.0 if random() < 0.5 else -1.0)*self.zero_angle_shift_increment
        elif self.zero_angle_shift_mode == 'increase':
            self.zero_angle_shift_increment *= 1.000
            da = self.zero_angle_shift_increment
        else:
            raise ValueError('zero_angle_shift_mode with value {} not valid'.format(self.zero_angle_shift_mode))
        self.zero_angle_shift += da
        s[ANGLE_IDX] = wrap_angle_rad(s[ANGLE_IDX]+self.zero_angle_shift)
        s[ANGLE_COS_IDX] = np.cos(s[ANGLE_IDX])
        s[ANGLE_SIN_IDX] = np.sin(s[ANGLE_IDX])

        return s

    def update_target_position(self):
        if self.controller_name == 'manual-stabilization':
            self.target_position = 0.0  # In this case target position is not used.
            # This just fill the corresponding column in history with zeros
        else:
            if self.use_pregenerated_target_position:

                # If time exceeds the max time for which target position was defined
                if self.time >= self.length_of_experiment:
                    return

                self.target_position = self.random_track_f(self.time)
                self.slider.value = self.target_position/TrackHalfLength  # Assign target position to slider to display it
            else:
                self.target_position = self.slider.value * TrackHalfLength  # Get target position from slider

    def update_target_equilibrium(self):
        if self.time_last_target_equilibrium_change is None:
            self.time_last_target_equilibrium_change = self.time
        elif self.target_equilibrium == -1 and (self.time-self.time_last_target_equilibrium_change) > self.keep_target_equilibrium_x_seconds_down:
            self.time_last_target_equilibrium_change = self.time
            self.target_equilibrium = -1*self.target_equilibrium
        elif self.target_equilibrium == 1 and (self.time-self.time_last_target_equilibrium_change) > self.keep_target_equilibrium_x_seconds_up:
            self.time_last_target_equilibrium_change = self.time
            self.target_equilibrium = -1*self.target_equilibrium

    def block_pole_at_90_deg(self):
        if self.stop_at_90:
            if self.s[ANGLE_IDX] >= np.pi / 2:
                self.s[ANGLE_IDX] = np.pi / 2
                self.s[ANGLED_IDX] = 0.0
                return True  # Make also second derivatives 0 after they are calculated
            elif self.s[ANGLE_IDX] <= -np.pi / 2:
                self.s[ANGLE_IDX] = -np.pi / 2
                self.s[ANGLED_IDX] = 0.0
                return True  # Make also second derivatives 0 after they are calculated
            else:
                return False
        else:
            return False

    def save_csv_routine(self):
        # Calculate time steps from last saving
        # The counter should be initialized at max-1 to start with a control input update
        self.dt_save_steps_counter += 1

        # If update time interval elapsed save current state and zero the counter
        if self.dt_save_steps_counter == self.dt_save_number_of_steps:
            # If user chose to save history of the simulation it is saved now
            # It is saved first internally to a dictionary in the Cart instance
            if self.save_data_in_cart:

                self.dict_history.update_history(self.variables_to_log)

                try:
                    self.dict_history.update_history(self.controller.controller_data_for_csv)
                except AttributeError:
                    pass
                except Exception:
                    print(traceback.format_exc())

            else:
                # TODO: This is probably inefficient to recreate dict_history every time
                self.dict_history = HistoryClass()
                self.dict_history.add_keys(keys=self.variables_to_log.keys())

                try:
                    self.dict_history.add_keys(self.controller.controller_data_for_csv.keys())
                    self.dict_history.update_history(self.controller.controller_data_for_csv)
                except AttributeError:
                    pass
                except Exception:
                    print(traceback.format_exc())

                self.dict_history.update_history(self.variables_to_log)

                self.save_flag = True

            self.dt_save_steps_counter = 0

    # A method integrating the cartpole ode over time step dt
    # Currently we use a simple single step Euler stepping
    def cartpole_integration(self):
        """
        Simple single step integration of CartPole state by dt
        """

        self.s[ANGLE_IDX], self.s[ANGLED_IDX], self.s[POSITION_IDX], self.s[POSITIOND_IDX] = \
            self.cpe.cartpole_integration(
                self.s[ANGLE_IDX], self.s[ANGLED_IDX], self.angleDD,
                self.s[POSITION_IDX], self.s[POSITIOND_IDX], self.positionDD,
                self.dt_simulation, u=self.u,
                k=k, m_cart=m_cart, m_pole=m_pole, g=g, J_fric=J_fric, M_fric=M_fric, L=L
            )


    def edge_bounce(self):
        # Elastic collision at edges
        self.s[ANGLE_IDX], self.s[ANGLED_IDX], self.s[POSITION_IDX], self.s[POSITIOND_IDX] = self.cpe.edge_bounce(
            self.s[ANGLE_IDX],
            np.cos(self.s[ANGLE_IDX]),
            self.s[ANGLED_IDX],
            self.s[POSITION_IDX],
            self.s[POSITIOND_IDX],
            self.dt_simulation,
            L=float(L),
        )

    # Determine the dimensionless [-1,1] value of the motor power Q
    # This function should be called for the first time to calculate 0th time step
    # Otherwise it goes out of sync with saving
    def Update_Q(self):
        # Calculate time steps from last update
        # The counter should be initialized at max-1 to start with a control input update
        self.dt_controller_steps_counter += 1

        # If update time interval elapsed update control input and zero the counter
        if self.dt_controller_steps_counter == self.dt_controller_number_of_steps:

            if self.controller_name == 'manual-stabilization':
                # in this case slider corresponds already to the power of the motor
                self.Q_calculated = self.slider.value
                self.Q_update_time = 0.0
            else:  # in this case slider gives a target position, lqr regulator
                update_start = timeit.default_timer()
                self.Q_calculated = float(self.controller.step(
                    self.s_with_noise_and_latency,
                    self.time,
                    {"target_position": self.target_position, "target_equilibrium": self.target_equilibrium, 'L': float(self.L_for_controller)}
                ))
                self.Q_update_time = timeit.default_timer()-update_start

            self.Q_applied = add_control_noise(self.Q_calculated, rng)

            self.Q = self.Q_applied
            self.dt_controller_steps_counter = 0

    def update_parameters(self):
        global L
        if self.time_last_L_change is None:
            self.time_last_L_change = self.time
        else:
            if (self.time-self.time_last_L_change) > self.change_L_every_x_second:
                self.time_last_L_change = self.time
                if self.L_change_mode == 'uniform':
                    L[...] = np.random.uniform(*self.L_range)
                elif self.L_change_mode == 'step':
                    if L + self.L_step > self.L_range[1] or L + self.L_step < self.L_range[0]:
                        self.L_step *= -1.0
                    L[...] = L + self.L_step

            else:
                L[...] = L * self.L_discount_factor



    # endregion

    # region 2. Methods related to experiment history as a whole: saving, loading, plotting, resetting

    # This method saves the dictionary keeping the history of simulation to a .csv file
    def save_history_csv(
            self,
            csv_name=None,
            mode='init',
            length_of_experiment='unknown',
            path_to_experiment_recordings=None):

        if mode == 'init':
            if csv_name is None or csv_name == '':
                csv_name = create_csv_file_name(self.controller, self.controller_name, self.optimizer_name)
            csv_title = create_csv_title()
            header = create_csv_header(self, length_of_experiment)
            if path_to_experiment_recordings is None:
                path_to_experiment_recordings = self.path_to_experiment_recordings
            self.csv_filepath = create_csv_file(csv_name, self.dict_history.keys(),
                                                path_to_experiment_recordings=path_to_experiment_recordings,
                                                title=csv_title, header=header)
        else:
            save_data_to_csv_file(self.csv_filepath, self.dict_history, self.rounding_decimals, mode=mode)


    # load csv file with experiment recording (e.g. for replay)
    def load_history_csv(self, csv_name=None):
        file_paths = get_full_paths_to_csvs(default_locations=self.path_to_experiment_recordings, csv_names=csv_name)
        data = load_csv_recording(file_paths[0])
        return data, file_paths[0]

    # region 3. Methods for generating random target position for generation of random experiment

    # Prepare CartPole Instance to perform an experiment with random target position trace
    def setup_cartpole_random_experiment(self,

                                         # Initial state
                                         s0=None,

                                         controller=None,

                                         dt_simulation=None,
                                         dt_controller=None,
                                         dt_save=None,

                                         # Settings related to random trace generation
                                         track_relative_complexity=None,
                                         length_of_experiment=None,
                                         interpolation_type=None,
                                         turning_points_period=None,
                                         start_random_target_position_at=None,
                                         end_random_target_position_at=None,
                                         turning_points=None,
                                         used_track_fraction=0.8,
                                         target_equilibrium=None,
                                         keep_target_equilibrium_x_seconds_up=np.inf,
                                         keep_target_equilibrium_x_seconds_down=np.inf,

                                         L_initial=None,
                                         change_L_every_x_seconds=None,
                                         L_discount_factor=None,
                                         L_range=None,
                                         L_informed_controller=None,
                                         L_change_mode=None,
                                         L_step=None,

                                         ):

        # Set time scales:
        if dt_simulation is not None: self.dt_simulation = dt_simulation
        if dt_controller is not None: self.dt_controller = dt_controller
        if dt_save is not None: self.dt_save = dt_save

        # Set CartPole in the right (automatic control) mode
        # You may want to provide it before this function not to reload it every time
        if controller is not None: self.set_controller(controller)

        # Set initial state
        if s0 is not None: self.s = s0

        if track_relative_complexity is not None: self.track_relative_complexity = track_relative_complexity
        if length_of_experiment is not None: self.length_of_experiment = length_of_experiment
        if interpolation_type is not None: self.interpolation_type = interpolation_type
        if turning_points_period is not None: self.turning_points_period = turning_points_period
        if start_random_target_position_at is not None: self.start_random_target_position_at = start_random_target_position_at
        if end_random_target_position_at is not None: self.end_random_target_position_at = end_random_target_position_at
        if turning_points is not None: self.turning_points = turning_points
        if used_track_fraction is not None: self.used_track_fraction = used_track_fraction
        if target_equilibrium is not None: self.target_equilibrium = target_equilibrium
        if keep_target_equilibrium_x_seconds_up is not None: self.keep_target_equilibrium_x_seconds_up = keep_target_equilibrium_x_seconds_up
        if keep_target_equilibrium_x_seconds_down is not None: self.keep_target_equilibrium_x_seconds_down = keep_target_equilibrium_x_seconds_down
        if L_initial is not None: self.L_initial = L_initial
        if change_L_every_x_seconds is not None: self.change_L_every_x_second = change_L_every_x_seconds
        if L_discount_factor is not None: self.L_discount_factor = L_discount_factor
        if L_range is not None: self.L_range = L_range
        if L_informed_controller is not None: self.L_informed_controller = L_informed_controller
        if L_change_mode is not None: self.L_change_mode = L_change_mode
        if L_step is not None: self.L_step = self.L_step

        global L

        if self.L_informed_controller:
            self.L_for_controller = L
        else:
            self.L_for_controller = float(self.L_initial)

        self.random_track_f = Generate_Random_Trace_Function(

            length_of_experiment=self.length_of_experiment,
            rtf_rng=self.rng_CartPole,

            track_relative_complexity=self.track_relative_complexity,
            interpolation_type=self.interpolation_type,
            turning_points=self.turning_points,
            turning_points_period=self.turning_points_period,

            start_random_target_position_at=self.start_random_target_position_at,
            end_random_target_position_at=self.end_random_target_position_at,

            used_track_fraction=self.used_track_fraction,
        )
        self.new_track_generated = True

        self.use_pregenerated_target_position = 1

        self.number_of_timesteps_in_random_experiment = int(np.ceil(self.length_of_experiment / self.dt_simulation))

        # Target position at time 0
        self.target_position = self.random_track_f(self.time)

        # Reset variables
        self.set_cartpole_state_at_t0(reset_mode=2, s=self.s, target_position=self.target_position)

        L[...] = float(self.L_initial)

    # Runs a random experiment with parameters set with setup_cartpole_random_experiment
    # And saves the experiment recording to csv file
    # @profile(precision=4)
    def run_cartpole_random_experiment(self,
                                       csv=None,
                                       path_to_experiment_recordings=None,
                                       save_mode='offline',
                                       show_summary_plots=True
                                       ):
        """
        This function runs a random CartPole experiment
        and returns the history of CartPole states, control inputs and desired cart position
        """

        if save_mode == 'offline':
            self.save_data_in_cart = True
        elif save_mode == 'online':
            self.save_data_in_cart = False
        else:
            raise ValueError('Unknown save mode value')

        self.cartpole_ode()

        # Create csv file for saving
        self.save_history_csv(
            csv_name=csv,
            mode='init',
            length_of_experiment=self.length_of_experiment,
            path_to_experiment_recordings=path_to_experiment_recordings,
        )

        # Save 0th timestep
        if save_mode == 'online':
            self.save_history_csv(csv_name=csv, mode='save online')

        # Run the CartPole experiment for number of time
        for _ in trange(self.number_of_timesteps_in_random_experiment):

            # Print an error message if it runs already to long (should stop before)
            if self.time > self.length_of_experiment:
                raise Exception('ERROR: It seems the experiment is running too long...')

            self.update_state()

            # Additional option to stop the experiment
            if abs(self.s[POSITION_IDX]) > 45.0:  # FIXME: THIS LIMIT CURRENTLY MAKES NO SENSE... (MP)
                print('Cart went out of safety boundaries')
                break

            # if abs(self.s[ANGLE_IDX]) > 0.8*np.pi:
            #     # raise ValueError('Cart went unstable')
            #     # print('Cart went unstable')
            #     break

            # It seems that if pole is to short angleD overflows quite quickly.
            # We limit pole to 1 mm
            if L < 0.005:
                print('Pole is too short! Terminating experiment before numeric errors will occur')
                break

            if save_mode == 'online' and self.save_flag:
                self.save_history_csv(csv_name=csv, mode='save online')
                self.save_flag = False

        data = pd.DataFrame(self.dict_history)

        if save_mode == 'offline':
            self.save_history_csv(csv_name=csv, mode='save offline')
        
        if show_summary_plots: summary_plots(self.dict_history)

        mean_abs_dist = np.mean([np.abs(self.dict_history["position"][i] - self.dict_history["target_position"][i]) for i in range(len(self.dict_history["target_position"]))])
        mean_abs_angle = np.mean(np.abs(self.dict_history["angle"])) * 180.0 / np.pi
        print(f"Mean absolute distance to target: {mean_abs_dist}m\nMean absolute angle: {mean_abs_angle}deg")

        # Set CartPole state - the only use is to make sure that experiment history is discared
        # Maybe you can delete this line
        self.set_cartpole_state_at_t0(reset_mode=0)

        return data

    # endregion

    # region 4. Methods "Get, set, reset"
    
    def set_optimizer(self, optimizer_name=None, optimizer_idx=None):
        self.optimizer_name, self.optimizer_idx = get_optimizer_name(
            optimizer_name=optimizer_name, optimizer_idx=optimizer_idx
        )
        if self.controller is not None and getattr(self.controller, "has_optimizer", False):
            self.controller.configure(self.optimizer_name)

    # Set the controller of CartPole
    def set_controller(self, controller_name=None, controller_idx=None):
        self.controller_name, self.controller_idx = get_controller_name(
            controller_name=controller_name, controller_idx=controller_idx
        )
        
        if self.controller_name != 'manual-stabilization':
            Controller: "type[template_controller]" = import_controller_by_name(self.controller_name)
            self.controller = Controller(
                environment_name="CartPole",
                initial_environment_attributes={
                    "target_position": self.target_position,
                    "target_equilibrium": self.target_equilibrium,
                    "L": float(self.L_for_controller)
                },
                control_limits=(self.action_space.low, self.action_space.high),
            )
            # Final configuration of controller
            if self.controller.has_optimizer:
                self.controller.configure(self.optimizer_name)
                self.optimizer_name, self.optimizer_idx = get_optimizer_name(
                    optimizer_name=self.controller.optimizer.optimizer_name
                )

            else:
                self.controller.configure()
            
                
        # Set the maximal allowed value of the slider - relevant only for GUI
        if self.slider is not None:
            if self.controller_name == 'manual-stabilization':
                self.slider.Slider_Arrow.set_positions((0, 0), (0, 0))
            else:
                self.slider.Slider_Bar.set_width(0.0)

        # TODO: optimally reset_dict_history would be False and the controller could be switched during experiment
        #   The False option is not implemented yet. So it is possible to switch controller only when the experiment is not running.
        #   Check also how it covers the case when controller is switched (possibly multiple times) when experiment is not running
        self.set_cartpole_state_at_t0(reset_mode=2, s=self.s, target_position=self.target_position, reset_dict_history=True)

        return True

    # This method resets the internal state of the CartPole instance
    # The starting state (for t = 0) may be
    # all zeros (reset_mode = 0)
    # set in this function (reset_mode = 1)
    # provide by user (reset_mode = 1), by giving s, Q and target_position
    def set_cartpole_state_at_t0(self, reset_mode=1, s=None, target_position=None, reset_dict_history=True):

        # Some controllers may need reset before being reused in the next experiment without reloading
        try:
            self.controller.controller_reset()
        except AttributeError:
            pass
        except NotImplementedError:
            pass

        # reset global variables
        global k, m_cart, m_pole, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength, controlDisturbance_mode
        (k[...], m_cart[...], m_pole[...], g[...], J_fric[...], M_fric[...], L[...], v_max[...], u_max[...],
         controlDisturbance[...], controlBias[...], TrackHalfLength[...], controlDisturbance_mode) = CP_PARAMETERS_DEFAULT.export_parameters()

        self.time = 0.0
        self.time_last_target_equilibrium_change = None
        self.time_last_L_change = None
        if reset_mode == 0:  # Don't change it
            self.s[POSITION_IDX] = self.s[POSITIOND_IDX] = 0.0
            self.s[ANGLE_IDX] = self.s[ANGLED_IDX] = 0.0
            self.s[ANGLE_COS_IDX] = np.cos(self.s[ANGLE_IDX])
            self.s[ANGLE_SIN_IDX] = np.sin(self.s[ANGLE_IDX])

            self.target_position = 0.0
            self.slider.value = 0.0

        elif reset_mode == 1:  # You may change this but be careful with other user. Better use 3
            # You can change here with which initial parameters you wish to start the simulation
            self.s[POSITION_IDX] = 0.0
            self.s[POSITIOND_IDX] = 0.0
            self.s[ANGLE_IDX] = (1.0 * self.rng_CartPole.normal() - 1.0) * np.pi / 180.0  # np.pi/2.0 #
            self.s[ANGLED_IDX] = 0.0  # 1.0

            self.s[ANGLE_COS_IDX] = np.cos(self.s[ANGLE_IDX])
            self.s[ANGLE_SIN_IDX] = np.sin(self.s[ANGLE_IDX])

            self.target_position = 0.0
            self.slider.value = 0.0

        elif reset_mode == 2:  # Don't change it
            if (s is not None) and (target_position is not None):

                self.s = s
                self.s[ANGLE_COS_IDX] = np.cos(self.s[ANGLE_IDX])
                self.s[ANGLE_SIN_IDX] = np.sin(self.s[ANGLE_IDX])

                if self.slider is not None:
                    self.slider.value = self.target_position = target_position

            else:
                raise ValueError('s, Q or target position not provided for initial state')

            if self.controller_name == 'manual-stabilization':
                # in this case slider corresponds already to the power of the motor
                if self.slider is not None:
                    self.Q_calculated = self.slider.value
                else:
                    self.Q_calculated = 0.0
            else:  # in this case slider gives a target position, lqr regulator
                self.Q_calculated = float(self.controller.step(
                    self.s,
                    self.time,
                    {"target_position": self.target_position, "target_equilibrium": self.target_equilibrium, "L": float(self.L_for_controller)}
                ))

            self.Q_applied = add_control_noise(self.Q_calculated, rng)

            self.Q = self.Q_applied
            self.u = self.cpe.Q2u(self.Q)  # Calculate CURRENT control input
            self.angleDD, self.positionDD = self.cpe.cartpole_ode_interface(self.s, self.u, L=float(L))  # Calculate CURRENT second derivatives

        # Reset the dict keeping the experiment history and save the state for t = 0
        self.dt_save_steps_counter = 0
        self.dt_controller_steps_counter = 0

        if reset_dict_history:
            self.dict_history = HistoryClass()
            self.dict_history.add_keys(keys=self.variables_to_log.keys())
            try:
                self.dict_history.add_keys(self.controller.controller_data_for_csv.keys())
                self.dict_history.update_history(self.controller.controller_data_for_csv)
            except AttributeError:
                pass
            except Exception:
                print(traceback.format_exc())
            self.dict_history.update_history(self.variables_to_log)

        else:  # If you don't want to reset dict_history you still need to add to the dictionary additional keys from controller.controller_data_for_csv
            ...
            # TODO: Implement this part when you want to switch controllers during experiment

    # region Get and set timescales

    # Makes sure that when dt is updated also related variables are updated

    @property
    def dt_simulation(self):
        return self._dt_simulation

    @dt_simulation.setter
    def dt_simulation(self, value):
        self._dt_simulation = value
        if self._dt_simulation is not None:
            # Set latency
            self.LatencyAdderInstance.dt_sampling = self._dt_simulation
            self.LatencyAdderInstance.set_latency(self.latency)
        if self._dt_controller is not None:
            self.dt_controller_number_of_steps = np.rint(self._dt_controller / value).astype(np.int32)
            if self.dt_controller_number_of_steps == 0:
                self.dt_controller_number_of_steps = 1
            # Initialize counter at max value to start with update
            self.dt_controller_steps_counter = 0
        if self._dt_save is not None:
            self.dt_save_number_of_steps = np.rint(self._dt_save / value).astype(np.int32)
            if self.dt_save_number_of_steps == 0:
                self.dt_save_number_of_steps = 1
            self.dt_save_steps_counter = 0

    @property
    def dt_controller(self):
        return self._dt_controller

    @dt_controller.setter
    def dt_controller(self, value):
        self._dt_controller = value
        if self._dt_simulation is not None:
            self.dt_controller_number_of_steps = np.rint(value / self._dt_simulation).astype(np.int32)
            if self.dt_controller_number_of_steps == 0:
                self.dt_controller_number_of_steps = 1
            # Initialize counter at max value to start with update
            self.dt_controller_steps_counter = 0

    @property
    def dt_save(self):
        return self._dt_save

    @dt_save.setter
    def dt_save(self, value):
        self._dt_save = value
        if self._dt_simulation is not None:
            self.dt_save_number_of_steps = np.rint(value / self._dt_simulation).astype(np.int32)
            if self.dt_save_number_of_steps == 0:
                self.dt_save_number_of_steps = 1
            # This counter is initialized at 0 - 0th step is saved manually
            self.dt_save_steps_counter = 0

    # endregion

    # endregion

    @staticmethod
    def current_L():
        return L



