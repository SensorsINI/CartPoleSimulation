# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:27:59 2020

@author: Marcin
"""

# Cart Class
# The file contain the class holding all the parameters and methods
# related to CartPole

from CartPole.cartpole_model import *

# Default timescales
dt_simulation_default = 0.002  # s, Update CartPole dynamical state every dt_simulation_default seconds
dt_controller_default = 0.02  # s, Recalculate control input every dt_controller_default seconds
dt_save_default = 0.02  # s,  Save CartPole state every dt_save_default seconds


class CartPole:

    # Methods related to dynamic evolution of CartPole system
    from ._CartPole_dynamical_evolution import update_state, Update_Q
    # Various mostly small functions for getting, setting and resetting modes of operation, variables, current state...
    from ._CartPole_get_set_reset import get_available_controller_names, set_controller, set_cartpole_state_at_t0
    from ._CartPole_get_set_reset import dt_save, dt_controller, dt_simulation
    # Methods related to experiment history as a whole: saving, loading, plotting, resetting
    from ._CartPole_experiment_history import save_history_csv, load_history_csv, summary_plots
    # Method for generating random target position for generation of random experiment
    from ._CartPole_generate_random_experiment import Generate_Random_Trace_Function
    # Methods needed to display CartPole in GUI
    from ._CartPole_gui_graphics import init_graphical_elements, draw_constant_elements, update_drawing, update_slider, run_animation

    def __init__(self):

        # Global time of the simulation
        self.time = 0.0

        self.s = s0  # Container for the state of the cart (s like state)
        # Variables for control input and target position.
        self.u = 0.0
        self.Q = 0.0
        self.target_position = 0.0
        # CartPole is initialized with state, control input, target position all zero
        # This is however usually changed before running the simulation. Treat it just as placeholders for variables
        # The same

        # Physical parameters of the cart
        self.p = p_globals

        # Time scales
        # ∆t in number of steps (related to simulation time step) is set while setting corresponding dt through @property
        self.dt_controller_number_of_steps = 0
        self.dt_save_number_of_steps = 0

        # Counts time steps from last controller update or saving
        # is set while setting corresponding dt through @property
        self.dt_controller_steps_counter = 0
        self.dt_save_steps_counter = 0

        # Helper variables to set timescales
        self._dt_simulation = 0.0
        self._dt_controller = 0.0
        self._dt_save = 0.0

        self.dt_simulation = dt_simulation_default
        self.dt_controller = dt_controller_default
        self.dt_save = dt_save_default

        # Other variables controlling operation of the program
        self.Q_max = 1.0 # We want control input to be normed between -1 and 1. The true force acting on the cart is calculated with Q2u function
        self.rounding_decimals = 5  # Sets number of digits after coma to save in experiment history for each feature

        self.save_data_in_cart = True  # This is usually modified before using the CartPole either from GUI or from Data Generator
        self.save_flag = False
        self.csv_filepath = None  # Where to save the experiment history. Must be set before attemting to save anything
        self.stop_at_90 = False  # If true pole is blocked after reaching the horizontal position

        self.controller = None  # Placeholder for the currently used controller function
        self.controller_name = ''  # Placeholder for the currently used controller name
        self.controller_idx = None  # Placeholder for the currently used controller index
        self.controller_names = self.get_available_controller_names() # list of controllers available in controllers folder
        self.set_controller('manual-stabilization')  # Initialize CartPole in manual-stabilization mode

        # Variables for pre-generated random trace
        # Parameters for random trace generation
        # These need to be set, before CartPole can generate random trace and random experiment
        self.track_relative_complexity = None  # randomly placed target points/s, 0.5 is normal default
        self.random_length = None  # seconds, length of the random length trace
        self.interpolation_type = None # Sets how to interpolate between turning points of random trace
        # Possible choices: '0-derivative-smooth', 'linear', 'previous'
        self.turning_points_period = None # How turning points should be distributed
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

        # Containers for variables related to random trace generation
        self.random_track_f = None # Function interpolataing the random target position between turning points, placeholder
        self.new_track_generated = False  # Flag informing if new target position track is generated
        self.t_max_pre = None # Placeholder for the end time of the generated random experimet
        self.use_pregenerated_target_position = False  # TODO Documentation missing

        self.dict_history = {}  # Create the dictionary to hold the experiment history

        # THE REMAINING PART OF __init__ METHOD CONCERNS DRAWING SETTINGS ONLY

        # DIMENSIONS OF THE DRAWING ONLY!!!
        # NOTHING TO DO WITH THE SIMULATION AND NOT INTENDED TO BE MANIPULATED BY USER !!!

        # Variable relevant for interactive use of slider
        self.slider_max = 0.0
        self.slider_value = 0.0

        # Variables needed to display CartPole in GUI
        # They are assign values in self.init_elements()
        self.CartLength = None
        self.WheelRadius = None
        self.WheelToMiddle = None
        self.y_plane = None
        self.y_wheel = None
        self.MastHight = None # For drawing only. For calculation see L
        self.MastThickness = None
        self.HalfLength = None  # Length of the track

        # Elements of the drawing
        self.Mast = None
        self.Chassis = None
        self.WheelLeft = None
        self.WheelRight = None

        self.Slider = None
        self.t2 = None  # An abstract container for the transform rotating the mast

        self.init_graphical_elements()
