# -*- coding: utf-8 -*-
"""
CartPole Class:
The file holds the main class of our simulator - it corresponds to a physical cartpole.
You can find here methods with performing experiment, saving data, displaying CartPole graphically
and many more. To run it needs some "environment": we provide you with GUI and data_generator
@author: Marcin
"""

# region Imported modules

from CartPole._CartPole_mathematical_helpers import wrap_angle_rad
from CartPole.state_utilities import ANGLED_IDX, ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX, POSITIOND_IDX, POSITION_IDX, cartpole_state_varname_to_index, cartpole_state_index_to_varname, cartpole_state_varnames_to_indices
from CartPole.cartpole_model import Q2u, cartpole_ode, s0, edge_bounce, euler_step
from others.p_globals import P_GLOBALS

import numpy as np
import pandas as pd

# Import module to save history of the simulation as csv file
import csv
# To detect the latest csv file
import glob
# Import module to interact with OS
import os
# Import module to get a current time and date used to name the files containing the history of simulations
from datetime import datetime
# Interpolate function to create smooth random track
from scipy.interpolate import BPoly, interp1d
# Run range() automatically adding progress bar in terminal
from tqdm import trange
try:
    # Use gitpython to get a current revision number and use it in description of experimental data
    from git import Repo
except:
    pass

# check memory usage of chosen methods. Commented by default
# from memory_profiler import profile

# region Graphics imports
import matplotlib.pyplot as plt
from matplotlib import animation
# rc sets global parameters for matplotlib; transforms is used to rotate the Mast
from matplotlib import rc, transforms
# Shapes used to draw a Cart and the slider
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch
# Angle convention to rotate the mast in right direction - depends on used Equation
from CartPole.cartpole_model import ANGLE_CONVENTION

# Set the font parameters for matplotlib figures
font = {'size': 22}
rc('font', **font)
# endregion

# endregion

import yaml
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
PATH_TO_CONTROLLERS = config["cartpole"]["PATH_TO_CONTROLLERS"]
PATH_TO_EXPERIMENT_RECORDINGS_DEFAULT = config["cartpole"]["PATH_TO_EXPERIMENT_RECORDINGS_DEFAULT"]

class CartPole:

    def __init__(self, initial_state=s0, path_to_experiment_recordings=None):

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
        self.Q = 0.0  # Dimensionless motor power in the range [-1,1] from which force is calculated with Q2u() method
        self.target_position = 0.0

        # Physical parameters of the CartPole
        self.p = P_GLOBALS

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
        self.controller_idx = None  # Placeholder for the currently used controller index
        self.controller_names = self.get_available_controller_names()  # list of controllers available in controllers folder
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
        self.t_max_pre = None  # Placeholder for the end time of the generated random experiment
        self.number_of_timesteps_in_random_experiment = None
        self.use_pregenerated_target_position = False  # Informs method performing experiment
        #                                                    not to take target position from environment
        # endregion and

        self.dict_history = {}  # Dictionary holding the experiment history

        # region Variables initialization for drawing/animating a CartPole
        # DIMENSIONS OF THE DRAWING ONLY!!!
        # NOTHING TO DO WITH THE SIMULATION AND NOT INTENDED TO BE MANIPULATED BY USER !!!

        # Variable relevant for interactive use of slider
        self.slider_max = 1.0
        self.slider_value = 0.0

        self.physical_to_graphics = None
        self.graphics_to_physical = None

        # Parameters needed to display CartPole in GUI
        # They are assigned with values in self.init_elements()
        self.CartLength = None
        self.WheelRadius = None
        self.WheelToMiddle = None
        self.y_plane = None
        self.y_wheel = None
        self.MastHight = None  # For drawing only. For calculation see L
        self.MastThickness = None
        self.TrackHalfLengthGraphics = None  # Length of the track

        # Elements of the drawing
        self.Mast = None
        self.Chassis = None
        self.WheelLeft = None
        self.WheelRight = None

        # Arrow indicating acceleration (=motor power)
        self.Acceleration_Arrow = None

        self.y_acceleration_arrow = None
        self.scaling_dx_acceleration_arrow = None
        self.x_acceleration_arrow = None

        # Depending on mode, slider may be displayed either as bar or as an arrow
        self.Slider_Bar = None
        self.Slider_Arrow = None
        self.t2 = None  # An abstract container for the transform rotating the mast

        self.init_graphical_elements()  # Assign proper object to the above variables
        # endregion

        # region Initialize CartPole in manual-stabilization mode
        self.set_controller('manual-stabilization')
        # endregion

    # region 1. Methods related to dynamic evolution of CartPole system

    # This method changes the internal state of the CartPole
    # from a state at time t to a state at t+dt
    # We assume this function is called for the first time to calculate first time step
    # @profile(precision=4)
    def update_state(self):
        # Update the total time of the simulation
        self.time = self.time + self.dt_simulation

        # Update target position depending on the mode of operation
        if self.use_pregenerated_target_position:

            # If time exceeds the max time for which target position was defined
            if self.time >= self.t_max_pre:
                return

            self.target_position = self.random_track_f(self.time)
            self.slider_value = self.target_position/self.p.TrackHalfLength  # Assign target position to slider to display it
        else:
            if self.controller_name == 'manual-stabilization':
                self.target_position = 0.0  # In this case target position is not used.
                # This just fill the corresponding column in history with zeros
            else:
                self.target_position = self.slider_value * self.p.TrackHalfLength  # Get target position from slider

        # Calculate the next state
        self.cartpole_integration()

        # Snippet to stop pole at +/- 90 deg if enabled
        zero_DD = None
        if self.stop_at_90:
            if self.s[cartpole_state_varname_to_index('angle')] >= np.pi / 2:
                self.s[cartpole_state_varname_to_index('angle')] = np.pi / 2
                self.s[cartpole_state_varname_to_index('angleD')] = 0.0
                zero_DD = True  # Make also second derivatives 0 after they are calculated
            elif self.s[cartpole_state_varname_to_index('angle')] <= -np.pi / 2:
                self.s[cartpole_state_varname_to_index('angle')] = -np.pi / 2
                self.s[cartpole_state_varname_to_index('angleD')] = 0.0
                zero_DD = True  # Make also second derivatives 0 after they are calculated
            else:
                zero_DD = False

        # Wrap angle to +/-π
        self.s[cartpole_state_varname_to_index('angle')] = wrap_angle_rad(self.s[cartpole_state_varname_to_index('angle')])

        # Calculate cosine and sine
        self.s[cartpole_state_varname_to_index('angle_cos')] = np.cos(self.s[cartpole_state_varname_to_index('angle')])
        self.s[cartpole_state_varname_to_index('angle_sin')] = np.sin(self.s[cartpole_state_varname_to_index('angle')])

        # Determine the dimensionless [-1,1] value of the motor power Q
        self.Update_Q()

        # Convert dimensionless motor power to a physical force acting on the Cart
        self.u = Q2u(self.Q)

        # Update second derivatives
        self.angleDD, self.positionDD = cartpole_ode(self.s, self.u)

        if zero_DD:
            self.angleDD = 0.0

        # Calculate time steps from last saving
        # The counter should be initialized at max-1 to start with a control input update
        self.dt_save_steps_counter += 1

        # If update time interval elapsed save current state and zero the counter
        if self.dt_save_steps_counter == self.dt_save_number_of_steps:
            # If user chose to save history of the simulation it is saved now
            # It is saved first internally to a dictionary in the Cart instance
            if self.save_data_in_cart:

                # Saving simulation data
                self.dict_history['time'].append(self.time)

                self.dict_history['angle'].append(self.s[cartpole_state_varname_to_index('angle')])
                self.dict_history['angleD'].append(self.s[cartpole_state_varname_to_index('angleD')])
                self.dict_history['angleDD'].append(self.angleDD)
                self.dict_history['angle_cos'].append(self.s[cartpole_state_varname_to_index('angle_cos')])
                self.dict_history['angle_sin'].append(self.s[cartpole_state_varname_to_index('angle_sin')])
                self.dict_history['position'].append(self.s[cartpole_state_varname_to_index('position')])
                self.dict_history['positionD'].append(self.s[cartpole_state_varname_to_index('positionD')])
                self.dict_history['positionDD'].append(self.positionDD)

                self.dict_history['Q'].append(self.Q)
                self.dict_history['u'].append(self.u)

                # The target_position is not always meaningful
                # If it is not meaningful all values in this column are set to 0
                self.dict_history['target_position'].append(self.target_position)

            else:

                self.dict_history = {
                                     'time': [self.time],

                                     'angle': [self.s[cartpole_state_varname_to_index('angle')]],
                                     'angleD': [self.s[cartpole_state_varname_to_index('angleD')]],
                                     'angleDD': [self.angleDD],
                                     'angle_cos': [self.s[cartpole_state_varname_to_index('angle_cos')]],
                                     'angle_sin': [self.s[cartpole_state_varname_to_index('angle_sin')]],
                                     'position': [self.s[cartpole_state_varname_to_index('position')]],
                                     'positionD': [self.s[cartpole_state_varname_to_index('positionD')]],
                                     'positionDD': [self.positionDD],


                                     'Q': [self.Q],
                                     'u': [self.u],

                                     'target_position': [self.target_position],

                                     }

                self.save_flag = True

            self.dt_save_steps_counter = 0

    # A method integrating the cartpole ode over time step dt
    # Currently we use a simple single step Euler stepping
    def cartpole_integration(self):
        """
        Simple single step integration of CartPole state by dt

        Takes state as numpy array.

        :param s: state of the CartPole (position, positionD, angle, angleD must be set). Array order follows global definition.
        :param dt: time step by which the CartPole state should be integrated
        """
        self.s[POSITION_IDX] = euler_step(self.s[POSITION_IDX], self.s[POSITIOND_IDX], self.dt_simulation)
        self.s[POSITIOND_IDX] = euler_step(self.s[POSITIOND_IDX], self.positionDD, self.dt_simulation)
        self.s[ANGLE_IDX] = euler_step(self.s[ANGLE_IDX], self.s[ANGLED_IDX], self.dt_simulation)
        self.s[ANGLED_IDX] = euler_step(self.s[ANGLED_IDX], self.angleDD, self.dt_simulation)

        # Elastic collision at edges
        self.s[ANGLE_IDX], self.s[ANGLED_IDX], self.s[POSITION_IDX], self.s[POSITIOND_IDX] = edge_bounce(
            self.s[ANGLE_IDX],
            self.s[ANGLED_IDX],
            self.s[POSITION_IDX],
            self.s[POSITIOND_IDX],
            self.dt_simulation
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
                self.Q = self.slider_value
            else:  # in this case slider gives a target position, lqr regulator
                self.Q = self.controller.step(self.s, self.target_position, self.time)

            self.dt_controller_steps_counter = 0

    # endregion

    # region 2. Methods related to experiment history as a whole: saving, loading, plotting, resetting

    # This method saves the dictionary keeping the history of simulation to a .csv file
    def save_history_csv(self, csv_name=None, mode='init', length_of_experiment='unknown'):

        if mode == 'init':

            # Make folder to save data (if not yet existing)
            try:
                os.makedirs(self.path_to_experiment_recordings[:-1])
            except FileExistsError:
                pass

            # Set path where to save the data
            if csv_name is None or csv_name == '':
                self.csv_filepath = self.path_to_experiment_recordings + 'CP_' + self.controller_name + str(
                    datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')) + '.csv'
            else:
                self.csv_filepath = csv_name
                if csv_name[-4:] != '.csv':
                    self.csv_filepath += '.csv'

                # If such file exists, append index to the end (do not overwrite)
                net_index = 1
                logpath_new = self.csv_filepath
                while True:
                    if os.path.isfile(logpath_new):
                        logpath_new = self.csv_filepath[:-4]
                    else:
                        self.csv_filepath = logpath_new
                        break
                    logpath_new = logpath_new + '-' + str(net_index) + '.csv'
                    net_index += 1

            # Write the .csv file
            with open(self.csv_filepath, "a") as outfile:
                writer = csv.writer(outfile)

                writer.writerow(['# ' + 'This is CartPole simulation from {} at time {}'
                                .format(datetime.now().strftime('%d.%m.%Y'), datetime.now().strftime('%H:%M:%S'))])
                try:
                    repo = Repo()
                    git_revision = repo.head.object.hexsha
                except:
                    git_revision = 'unknown'
                writer.writerow(['# ' + 'Done with git-revision: {}'
                                .format(git_revision)])

                writer.writerow(['#'])
                writer.writerow(['# Length of experiment: {} s'.format(str(length_of_experiment))])

                writer.writerow(['#'])
                writer.writerow(['# Time intervals dt:'])
                writer.writerow(['# Simulation: {} s'.format(str(self.dt_simulation))])
                writer.writerow(['# Controller update: {} s'.format(str(self.dt_controller))])
                writer.writerow(['# Saving: {} s'.format(str(self.dt_save))])

                writer.writerow(['#'])

                writer.writerow(['# Controller: {}'.format(self.controller_name)])

                writer.writerow(['#'])
                writer.writerow(['# Parameters:'])
                for k in self.p.__dict__:
                    writer.writerow(['# ' + k + ': ' + str(getattr(self.p, k))])
                writer.writerow(['#'])

                writer.writerow(['# Data:'])
                writer.writerow(self.dict_history.keys())

        elif mode == 'save online':

            # Save this dict
            with open(self.csv_filepath, "a") as outfile:
                writer = csv.writer(outfile)
                if self.rounding_decimals == np.inf:
                    pass
                else:
                    self.dict_history = {key: np.around(value, self.rounding_decimals)
                                         for key, value in self.dict_history.items()}
                writer.writerows(zip(*self.dict_history.values()))
            self.save_now = False

        elif mode == 'save offline':
            # Round data to a set precision
            with open(self.csv_filepath, "a") as outfile:
                writer = csv.writer(outfile)
                if self.rounding_decimals == np.inf:
                    pass
                else:
                    self.dict_history = {key: np.around(value, self.rounding_decimals)
                                         for key, value in self.dict_history.items()}
                writer.writerows(zip(*self.dict_history.values()))
            self.save_now = False
            # Another possibility to save data.
            # DF_history = pd.DataFrame.from_dict(self.dict_history).round(self.rounding_decimals)
            # DF_history.to_csv(self.csv_filepath, index=False, header=False, mode='a') # Mode (a)ppend

    # load csv file with experiment recording (e.g. for replay)
    def load_history_csv(self, csv_name=None):

        # Set path where to save the data
        if csv_name is None or csv_name == '':
            # get the latest file from the default location
            try:
                list_of_files = glob.glob(self.path_to_experiment_recordings + '/*.csv')
                file_path = max(list_of_files, key=os.path.getctime)
            except FileNotFoundError:
                print('Cannot load: No experiment recording found in data folder ' + './data/')
                return False
        else:
            filename = csv_name
            if csv_name[-4:] != '.csv':
                filename += '.csv'

            # check if file found in DATA_FOLDER_NAME or at local starting point
            if not os.path.isfile(filename):
                file_path = os.path.join(self.path_to_experiment_recordings, filename)
                if not os.path.isfile(file_path):
                    print(
                        'Cannot load: There is no experiment recording file with name {} at local folder or in {}'.format(
                            filename, self.path_to_experiment_recordings))
                    return False
            else:
                file_path = filename

        # Get race recording
        print('Loading file {}'.format(file_path))
        try:
            data: pd.DataFrame = pd.read_csv(file_path, comment='#')  # skip comment lines starting with #
        except Exception as e:
            print('Cannot load: Caught {} trying to read CSV file {}'.format(e, file_path))
            return False

        return data, file_path

    # Method plotting the dynamic evolution over time of the CartPole
    # It should be called after an experiment and only if experiment data was saved
    def summary_plots(self):
        fontsize_labels = 14
        fontsize_ticks = 12
        fig, axs = plt.subplots(4, 1, figsize=(16, 9), sharex=True)  # share x axis so zoom zooms all plots

        # Plot angle error
        axs[0].set_ylabel("Angle (deg)", fontsize=fontsize_labels)
        axs[0].plot(np.array(self.dict_history['time']), np.array(self.dict_history['angle']) * 180.0 / np.pi,
                    'b', markersize=12, label='Ground Truth')
        axs[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

        # Plot position
        axs[1].set_ylabel("position (m)", fontsize=fontsize_labels)
        axs[1].plot(self.dict_history['time'], self.dict_history['position'], 'g', markersize=12,
                    label='Ground Truth')
        axs[1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

        # Plot motor input command
        try:
            axs[2].set_ylabel("motor (N)", fontsize=fontsize_labels)
            axs[2].plot(self.dict_history['time'], self.dict_history['u'], 'r', markersize=12,
                        label='motor')
            axs[2].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
        except KeyError:
            axs[2].set_ylabel("motor normalized (-)", fontsize=fontsize_labels)
            axs[2].plot(self.dict_history['time'], self.dict_history['Q'], 'r', markersize=12,
                        label='motor')
            axs[2].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

        # Plot target position
        axs[3].set_ylabel("position target (m)", fontsize=fontsize_labels)
        axs[3].plot(self.dict_history['time'], self.dict_history['target_position'], 'k')
        axs[3].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

        axs[3].set_xlabel('Time (s)', fontsize=fontsize_labels)

        fig.align_ylabels()

        plt.show()

        return fig, axs

    # endregion

    # region 3. Methods for generating random target position for generation of random experiment

    # Generates a random target position
    # in a form of a function interpolating between turning points
    def Generate_Random_Trace_Function(self):
        if (self.turning_points is None) or (self.turning_points == []):

            number_of_turning_points = int(np.floor(self.length_of_experiment * self.track_relative_complexity))

            y = np.random.uniform(-1.0, 1.0, number_of_turning_points)
            y = y * self.used_track_fraction * self.p.TrackHalfLength

            if number_of_turning_points == 0:
                y = np.append(y, 0.0)
                y = np.append(y, 0.0)
            elif number_of_turning_points == 1:
                if self.start_random_target_position_at is not None:
                    y[0] = self.start_random_target_position_at
                elif self.end_random_target_position_at is not None:
                    y[0] = self.end_random_target_position_at
                else:
                    pass
                y = np.append(y, y[0])
            else:
                if self.start_random_target_position_at is not None:
                    y[0] = self.start_random_target_position_at
                if self.end_random_target_position_at is not None:
                    y[-1] = self.end_random_target_position_at

        else:
            number_of_turning_points = len(self.turning_points)
            if number_of_turning_points == 0:
                raise ValueError('You should not be here!')
            elif number_of_turning_points == 1:
                y = np.array([self.turning_points[0], self.turning_points[0]])
            else:
                y = np.array(self.turning_points)

        number_of_timesteps = np.ceil(self.length_of_experiment / self.dt_simulation)
        self.t_max_pre = number_of_timesteps * self.dt_simulation

        random_samples = number_of_turning_points - 2 if number_of_turning_points - 2 >= 0 else 0

        # t_init = linspace(0, self.t_max_pre, num=self.track_relative_complexity, endpoint=True)
        if self.turning_points_period == 'random':
            t_init = np.sort(np.random.uniform(self.dt_simulation, self.t_max_pre - self.dt_simulation, random_samples))
            t_init = np.insert(t_init, 0, 0.0)
            t_init = np.append(t_init, self.t_max_pre)
        elif self.turning_points_period == 'regular':
            t_init = np.linspace(0, self.t_max_pre, num=random_samples + 2, endpoint=True)
        else:
            raise NotImplementedError('There is no mode corresponding to this value of turning_points_period variable')

        # Try algorithm setting derivative to 0 a each point
        if self.interpolation_type == '0-derivative-smooth':
            yder = [[y[i], 0] for i in range(len(y))]
            random_track_f = BPoly.from_derivatives(t_init, yder)
        elif self.interpolation_type == 'linear':
            random_track_f = interp1d(t_init, y, kind='linear')
        elif self.interpolation_type == 'previous':
            random_track_f = interp1d(t_init, y, kind='previous')
        else:
            raise ValueError('Unknown interpolation type.')

        # Truncate the target position to be not grater than 80% of track length
        def random_track_f_truncated(time):

            target_position = random_track_f(time)
            if target_position > 0.8 * self.p.TrackHalfLength:
                target_position = 0.8 * self.p.TrackHalfLength
            elif target_position < -0.8 * self.p.TrackHalfLength:
                target_position = -0.8 * self.p.TrackHalfLength

            return target_position

        self.random_track_f = random_track_f_truncated

        self.new_track_generated = True

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

        self.Generate_Random_Trace_Function()
        self.use_pregenerated_target_position = 1

        self.number_of_timesteps_in_random_experiment = int(np.ceil(self.length_of_experiment / self.dt_simulation))

        # Target position at time 0
        self.target_position = self.random_track_f(self.time)

        # Make already in the first timestep Q appropriate to the initial state, target position and controller

        if self.controller_name == 'manual-stabilization':
            # in this case slider corresponds already to the power of the motor
            self.Q = self.slider_value
        else:  # in this case slider gives a target position, lqr regulator
            self.Q = self.controller.step(self.s, self.target_position, self.time)

        self.set_cartpole_state_at_t0(reset_mode=2, s=self.s, Q=self.Q, target_position=self.target_position)

    # Runs a random experiment with parameters set with setup_cartpole_random_experiment
    # And saves the experiment recording to csv file
    # @profile(precision=4)
    def run_cartpole_random_experiment(self,
                                       csv=None,
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

        # Create csv file for saving
        self.save_history_csv(csv_name=csv, mode='init', length_of_experiment=self.length_of_experiment)

        # Save 0th timestep
        if save_mode == 'online':
            self.save_history_csv(csv_name=csv, mode='save online')

        # Run the CartPole experiment for number of time
        for _ in trange(self.number_of_timesteps_in_random_experiment):

            # Print an error message if it runs already to long (should stop before)
            if self.time > self.t_max_pre:
                raise Exception('ERROR: It seems the experiment is running too long...')

            self.update_state()

            # Additional option to stop the experiment
            if abs(self.s[cartpole_state_varname_to_index('position')]) > 45.0:
                break
                print('Cart went out of safety boundaries')

            # if abs(self.s[cartpole_state_varname_to_index('angle')]) > 0.8*np.pi:
            #     # raise ValueError('Cart went unstable')
            #     print('Cart went unstable')
            #     break

            if save_mode == 'online' and self.save_flag:
                self.save_history_csv(csv_name=csv, mode='save online')
                self.save_flag = False

        data = pd.DataFrame(self.dict_history)

        if save_mode == 'offline':
            self.save_history_csv(csv_name=csv, mode='save offline')
        
        if show_summary_plots: self.summary_plots()

        mean_abs_dist = np.mean([np.abs(self.dict_history["position"][i] - self.dict_history["target_position"][i]) for i in range(len(self.dict_history["target_position"]))])
        mean_abs_angle = np.mean(np.abs(self.dict_history["angle"])) * 180.0 / np.pi
        print(f"Mean absolute distance to target: {mean_abs_dist}m\nMean absolute angle: {mean_abs_angle}deg")

        # Set CartPole state - the only use is to make sure that experiment history is discared
        # Maybe you can delete this line
        self.set_cartpole_state_at_t0(reset_mode=0)

        return data

    # endregion

    # region 4. Methods "Get, set, reset"

    # Method returns the list of controllers available in the PATH_TO_CONTROLLERS folder
    def get_available_controller_names(self):
        """
        Method returns the list of controllers available in the PATH_TO_CONTROLLERS folder
        """
        controller_files = glob.glob(PATH_TO_CONTROLLERS + 'controller_' + '*.py')
        controller_names = ['manual-stabilization']
        controller_names.extend(np.sort(
            [os.path.basename(item)[len('controller_'):-len('.py')].replace('_', '-') for item in controller_files]
        ))

        return controller_names

    # Set the controller of CartPole
    def set_controller(self, controller_name=None, controller_idx=None):
        """
        The method sets a new controller as the current controller of the CartPole instance.
        The controller may be indicated either by its name
        or by the index on the controller list (see get_available_controller_names method).
        """

        # Check if the proper information was provided: either controller_name or controller_idx
        if (controller_name is None) and (controller_idx is None):
            raise ValueError('You have to specify either controller_name or controller_idx to set a new controller.'
                             'You have specified none of the two.')
        elif (controller_name is not None) and (controller_idx is not None):
            raise ValueError('You have to specify either controller_name or controller_idx to set a new controller.'
                             'You have specified both.')
        else:
            pass

        # If controller name provided get controller index and vice versa
        if (controller_name is not None):
            try:
                controller_idx = self.controller_names.index(controller_name)
            except ValueError:
                print('{} is not in list. \n In list are: {}'.format(controller_name, self.controller_names))
                return False
        else:
            controller_name = self.controller_names[controller_idx]

        # save controller name and index to variables in the CartPole namespace
        self.controller_name = controller_name
        self.controller_idx = controller_idx

        # Load controller
        if self.controller_name == 'manual-stabilization':
            self.controller = None
        else:
            controller_full_name = 'controller_' + self.controller_name.replace('-', '_')
            path_import = PATH_TO_CONTROLLERS[2:].replace('/', '.').replace(r'\\', '.')
            import_str = 'from ' + path_import + controller_full_name + ' import ' + controller_full_name
            exec(import_str)
            self.controller = eval(controller_full_name + '()')

        # Set the maximal allowed value of the slider - relevant only for GUI
        if self.controller_name == 'manual-stabilization':
            self.Slider_Arrow.set_positions((0, 0), (0, 0))
        else:
            self.Slider_Bar.set_width(0.0)

        return True

    # This method resets the internal state of the CartPole instance
    # The starting state (for t = 0) may be
    # all zeros (reset_mode = 0)
    # set in this function (reset_mode = 1)
    # provide by user (reset_mode = 1), by giving s, Q and target_position
    def set_cartpole_state_at_t0(self, reset_mode=1, s=None, Q=None, target_position=None):
        self.time = 0.0
        if reset_mode == 0:  # Don't change it
            self.s[cartpole_state_varname_to_index('position')] = self.s[cartpole_state_varname_to_index('positionD')] = self.positionDD = 0.0
            self.s[cartpole_state_varname_to_index('angle')] = self.s[cartpole_state_varname_to_index('angleD')] = self.angleDD = 0.0
            self.Q = self.u = 0.0
            self.slider = self.target_position = 0.0

        elif reset_mode == 1:  # You may change this but be carefull with other user. Better use 3
            # You can change here with which initial parameters you wish to start the simulation
            self.s[cartpole_state_varname_to_index('position')] = 0.0
            self.s[cartpole_state_varname_to_index('positionD')] = 0.0
            self.s[cartpole_state_varname_to_index('angle')] = (1.0 * np.random.normal() - 1.0) * np.pi / 180.0  # np.pi/2.0 #
            self.s[cartpole_state_varname_to_index('angleD')] = 0.0  # 1.0

            if self.controller_name == 'manual-stabilization':
                self.target_position = 0.0
            else:
                self.target_position = self.slider_value * self.p.TrackHalfLength

            if self.controller_name == 'manual-stabilization':
                # in this case slider corresponds already to the power of the motor
                self.Q = self.slider_value
            else:  # in this case slider gives a target position, lqr regulator
                self.Q = self.controller.step(self.s, self.target_position, self.time)

            self.u = Q2u(self.Q)
            self.angleDD, self.positionDD = cartpole_ode(self.s, self.u)

        elif reset_mode == 2:  # Don't change it
            if (s is not None) and (Q is not None) and (target_position is not None):
                self.s = s
                self.Q = Q
                self.slider = self.target_position = target_position

                self.u = Q2u(self.Q)  # Calculate CURRENT control input
                self.angleDD, self.positionDD = cartpole_ode(self.s, self.u)  # Calculate CURRENT second derivatives
            else:
                raise ValueError('s, Q or target position not provided for initial state')

        # Reset the dict keeping the experiment history and save the state for t = 0
        self.dt_save_steps_counter = 0
        self.dt_controller_steps_counter = 0
        self.dict_history = {

                             'time': [self.time],

                             'angle': [self.s[cartpole_state_varname_to_index('angle')]],
                             'angleD': [self.s[cartpole_state_varname_to_index('angleD')]],
                             'angleDD': [self.angleDD],
                             'angle_cos': [np.cos(self.s[cartpole_state_varname_to_index('angle')])],
                             'angle_sin': [np.sin(self.s[cartpole_state_varname_to_index('angle')])],
                             'position': [self.s[cartpole_state_varname_to_index('position')]],
                             'positionD': [self.s[cartpole_state_varname_to_index('positionD')]],
                             'positionDD': [self.positionDD],

                             'Q': [self.Q],
                             'u': [self.u],

                             'target_position': [self.target_position],

                             }

    # region Get and set timescales

    # Makes sure that when dt is updated also related variables are updated

    @property
    def dt_simulation(self):
        return self._dt_simulation

    @dt_simulation.setter
    def dt_simulation(self, value):
        self._dt_simulation = value
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

    # region 5. Methods needed to display CartPole in GUI
    """
    This section contains methods related to displaying CartPole in GUI of the simulator.
    One could think of moving these function outside of CartPole class and connecting them rather more tightly
    with GUI of the simulator.
    We leave them however as a part of CartPole class as they rely on variables of the CartPole.
    """

    # This method initializes CartPole elements to be plotted in CartPole GUI
    def init_graphical_elements(self):

        self.CartLength = 10.0
        self.WheelRadius = 0.5
        self.WheelToMiddle = 4.0
        self.y_plane = 0.0
        self.y_wheel = self.y_plane + self.WheelRadius
        self.MastHight = 10.0  # For drawing only. For calculation see L
        self.MastThickness = 0.05
        self.TrackHalfLengthGraphics = 50.0  # Full Length of the track

        self.physical_to_graphics = (self.TrackHalfLengthGraphics-self.WheelToMiddle)/self.p.TrackHalfLength  # TrackHalfLength is the effective length of track
        self.graphics_to_physical = 1.0/self.physical_to_graphics

        self.y_acceleration_arrow = 1.5 * self.WheelRadius
        self.scaling_dx_acceleration_arrow = 20.0
        self.x_acceleration_arrow = (
                                   self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics +
                                   # np.sign(self.Q) * (self.CartLength / 2.0) +
                                   self.scaling_dx_acceleration_arrow * self.Q
        )

        # Initialize elements of the drawing
        self.Mast = FancyBboxPatch(xy=(self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics - (self.MastThickness / 2.0), 1.25 * self.WheelRadius),
                                   width=self.MastThickness,
                                   height=self.MastHight,
                                   fc='g')

        self.Chassis = FancyBboxPatch((self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics - (self.CartLength / 2.0), self.WheelRadius),
                                      self.CartLength,
                                      1 * self.WheelRadius,
                                      fc='r')

        self.WheelLeft = Circle((self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics - self.WheelToMiddle, self.y_wheel),
                                radius=self.WheelRadius,
                                fc='y',
                                ec='k',
                                lw=5)

        self.WheelRight = Circle((self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics + self.WheelToMiddle, self.y_wheel),
                                 radius=self.WheelRadius,
                                 fc='y',
                                 ec='k',
                                 lw=5)

        self.Acceleration_Arrow = FancyArrowPatch((self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics, self.y_acceleration_arrow),
                                                  (self.x_acceleration_arrow, self.y_acceleration_arrow),
                                                  arrowstyle='simple', mutation_scale=10,
                                                  facecolor='gold', edgecolor='orange')

        self.Slider_Arrow = FancyArrowPatch((self.slider_value, 0), (self.slider_value, 0),
                                            arrowstyle='fancy', mutation_scale=50)
        self.Slider_Bar = Rectangle((0.0, 0.0), self.slider_value, 1.0)
        self.t2 = transforms.Affine2D().rotate(0.0)  # An abstract container for the transform rotating the mast

    # This method accepts the mouse position and updated the slider value accordingly
    # The mouse position has to be captured by a function not included in this class
    def update_slider(self, mouse_position):
        # The if statement formulates a saturation condition

        if mouse_position > self.slider_max:
            self.slider_value = self.slider_max
        elif mouse_position < -self.slider_max:
            self.slider_value = -self.slider_max
        else:
            self.slider_value = mouse_position

    # This method draws elements and set properties of the CartPole figure
    # which do not change at every frame of the animation
    def draw_constant_elements(self, fig, AxCart, AxSlider):
        # Delete all elements of the Figure
        AxCart.clear()
        AxSlider.clear()

        ## Upper chart with Cart Picture
        # Set x and y limits
        AxCart.set_xlim((-self.TrackHalfLengthGraphics * 1.1, self.TrackHalfLengthGraphics * 1.1))
        AxCart.set_ylim((-1.0, 15.0))
        # Remove ticks on the y-axes
        AxCart.yaxis.set_major_locator(plt.NullLocator())  # NullLocator is used to disable ticks on the Figures

        locs = [-50.0, -25.0, - 0.0, 25.0, 50.0]
        labels = [str(np.around(np.array(x*self.graphics_to_physical), 3)) for x in locs]
        AxCart.xaxis.set_major_locator(plt.FixedLocator(locs))
        AxCart.xaxis.set_major_formatter(plt.FixedFormatter(labels))

        # Draw track
        Floor = Rectangle((-self.TrackHalfLengthGraphics, -1.0),
                          2 * self.TrackHalfLengthGraphics,
                          1.0,
                          fc='brown')
        AxCart.add_patch(Floor)

        # Draw an invisible point at constant position
        # Thanks to it the axes is drawn high enough for the mast
        InvisiblePointUp = Rectangle((0, self.MastHight + 2.0),
                                     self.MastThickness,
                                     0.0001,
                                     fc='w',
                                     ec='w')

        AxCart.add_patch(InvisiblePointUp)
        # Apply scaling
        AxCart.axis('scaled')

        ## Lower Chart with Slider
        # Set y limits
        AxSlider.set(xlim=(-1.1 * self.slider_max, self.slider_max * 1.1))
        # Remove ticks on the y-axes
        AxSlider.yaxis.set_major_locator(plt.NullLocator())  # NullLocator is used to disable ticks on the Figures

        if self.controller_name == 'manual-stabilization':
            pass
        else:
            locs = np.array([-50.0, -37.5, -25.0, -12.5, - 0.0, 12.5, 25.0, 37.5, 50.0])/50.0
            labels = [str(np.around(np.array(x * self.p.TrackHalfLength), 3)) for x in locs]
            AxSlider.xaxis.set_major_locator(plt.FixedLocator(locs))
            AxSlider.xaxis.set_major_formatter(plt.FixedFormatter(labels))

        # Apply scaling
        AxSlider.set_aspect("auto")

        return fig, AxCart, AxSlider

    # This method updates the elements of the Cart Figure which change at every frame.
    # Not that these elements are not ploted directly by this method
    # but rather returned as objects which can be used by another function
    # e.g. animation function from matplotlib package
    def update_drawing(self):

        self.x_acceleration_arrow = (
                                   self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics +
                                   # np.sign(self.Q) * (self.CartLength / 2.0) +
                                   self.scaling_dx_acceleration_arrow * self.Q
        )

        self.Acceleration_Arrow.set_positions((self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics, self.y_acceleration_arrow),
                                             (self.x_acceleration_arrow, self.y_acceleration_arrow))

        # Draw mast
        mast_position = (self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics - (self.MastThickness / 2.0))
        self.Mast.set_x(mast_position)
        # Draw rotated mast
        t21 = transforms.Affine2D().translate(-mast_position, -1.25 * self.WheelRadius)
        if ANGLE_CONVENTION == 'CLOCK-NEG':
            t22 = transforms.Affine2D().rotate(self.s[cartpole_state_varname_to_index('angle')])
        elif ANGLE_CONVENTION == 'CLOCK-POS':
            t22 = transforms.Affine2D().rotate(-self.s[cartpole_state_varname_to_index('angle')])
        else:
            raise ValueError('Unknown angle convention')
        t23 = transforms.Affine2D().translate(mast_position, 1.25 * self.WheelRadius)
        self.t2 = t21 + t22 + t23
        # Draw Chassis
        self.Chassis.set_x(self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics - (self.CartLength / 2.0))
        # Draw Wheels
        self.WheelLeft.center = (self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics - self.WheelToMiddle, self.y_wheel)
        self.WheelRight.center = (self.s[cartpole_state_varname_to_index('position')]*self.physical_to_graphics + self.WheelToMiddle, self.y_wheel)
        # Draw SLider
        if self.controller_name == 'manual-stabilization':
            self.Slider_Bar.set_width(self.slider_value)
        else:
            self.Slider_Arrow.set_positions((self.slider_value, 0), (self.slider_value, 1.0))

        return self.Mast, self.t2, self.Chassis, self.WheelRight, self.WheelLeft,\
               self.Slider_Bar, self.Slider_Arrow, self.Acceleration_Arrow

    # A function redrawing the changing elements of the Figure
    def run_animation(self, fig):
        def init():
            # Adding variable elements to the Figure
            fig.AxCart.add_patch(self.Mast)
            fig.AxCart.add_patch(self.Chassis)
            fig.AxCart.add_patch(self.WheelLeft)
            fig.AxCart.add_patch(self.WheelRight)
            fig.AxCart.add_patch(self.Acceleration_Arrow)
            fig.AxSlider.add_patch(self.Slider_Bar)
            fig.AxSlider.add_patch(self.Slider_Arrow)
            return self.Mast, self.Chassis, self.WheelLeft, self.WheelRight,\
                   self.Slider_Bar, self.Slider_Arrow, self.Acceleration_Arrow

        def animationManage(i):
            # Updating variable elements
            self.update_drawing()
            # Special care has to be taken of the mast rotation
            self.t2 = self.t2 + fig.AxCart.transData
            self.Mast.set_transform(self.t2)
            return self.Mast, self.Chassis, self.WheelLeft, self.WheelRight,\
                   self.Slider_Bar, self.Slider_Arrow, self.Acceleration_Arrow

        # Initialize animation object
        anim = animation.FuncAnimation(fig, animationManage,
                                       init_func=init,
                                       frames=300,
                                       # fargs=(CartPoleInstance,), # It was used when this function was a part of GUI class. Now left as an example how to add arguments to FuncAnimation
                                       interval=10,
                                       blit=True,
                                       repeat=True)
        return anim

    # endregion
