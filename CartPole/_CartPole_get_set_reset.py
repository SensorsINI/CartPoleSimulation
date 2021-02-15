import numpy as np
from CartPole.cartpole_model import cartpole_ode, Q2u
import glob
import os

PATH_TO_CONTROLLERS = './controllers/'

def get_available_controller_names(self):
    """
    Method returns the list of controllers availavle in the PATH_TO_CONTROLLERS folder
    """
    controller_files = glob.glob(PATH_TO_CONTROLLERS + 'controller_' + '*.py')
    controller_names = ['manual-stabilization']
    controller_names.extend(np.sort(
        [os.path.basename(item)[len('controller_'):-len('.py')].replace('_', '-') for item in controller_files]
    ))

    return controller_names

# Set the controller of CartPole
def set_controller(self, controller_name = None, controller_idx = None):
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
            raise ValueError('{} is not in list. \n In list are: {}'.format(controller_name, self.controller_names))
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
        self.slider_max = self.Q_max
    else:
        self.slider_max = self.p.TrackHalfLength


# This method resets the internal state of the CartPole instance
# The starting state (for t = 0) may be
# all zeros (reset_mode = 0)
# set in this function (reset_mode = 1)
# provide by user (reset_mode = 1), by giving s, Q and target_position
def set_cartpole_state_at_t0(self, reset_mode=1, s=None, Q=None, target_position=None):
    self.time = 0.0
    if reset_mode == 0: # Don't change it
        self.s.position = self.s.positionD = self.s.positionDD = 0.0
        self.s.angle = self.s.angleD = self.s.angleDD = 0.0
        self.Q = self.u = 0.0
        self.slider = self.target_position = 0.0

    elif reset_mode == 1: # You may change this but be carefull with other user. Better use 3
        # You can change here with which initial parameters you wish to start the simulation
        self.s.position = 0.0
        self.s.positionD = 0.0
        self.s.angle = (2.0 * np.random.normal() - 1.0) * np.pi / 180.0  # np.pi/2.0 #
        self.s.angleD = 0.0  # 1.0
        self.target_position = self.slider_value

        self.Q = 0.0

        self.u = Q2u(self.Q, self.p)
        self.s.angleDD, self.s.positionDD = cartpole_ode(self.p, self.s, self.u)

    elif reset_mode == 2:  # Don't change it
        if (s is not None) and (Q is not None) and (target_position is not None):
            self.s = s
            self.Q = Q
            self.slider = self.target_position = target_position

            self.u = Q2u(self.Q, self.p)  # Calculate CURRENT control input
            self.s.angleDD, self.s.positionDD = cartpole_ode(self.p, self.s, self.u)  # Calculate CURRENT second derivatives
        else:
            raise ValueError('s, Q or target position not provided for initial state')

    # Reset the dict keeping the experiment history and save the state for t = 0
    self.dict_history = {'time': [self.time],
                         's.position': [self.s.position],
                         's.positionD': [self.s.positionD],
                         's.positionDD': [self.s.positionDD],
                         's.angle': [self.s.angle],
                         's.angleD': [self.s.angleD],
                         's.angleDD': [self.s.angleDD],
                         'u': [self.u],
                         'Q': [self.Q],
                         'target_position': [self.target_position],
                         's.angle.sin': [np.sin(self.s.angle)],
                         's.angle.cos': [np.cos(self.s.angle)]}

# Make sure that when dt is updated also related number of time steps is updated
@property
def dt_simulation(self):
    return self._dt_simulation

@dt_simulation.setter
def dt_simulation(self, value):
    self._dt_simulation = value
    self.dt_controller_number_of_steps = np.rint(self._dt_controller / value)
    self.dt_save_number_of_steps = np.rint(self._dt_save / value)
    if self.dt_controller_number_of_steps == 0:
        self.dt_controller_number_of_steps = 1
    if self.dt_save_number_of_steps == 0:
        self.dt_save_number_of_steps = 1
    # Initialize counter at max value to start with update
    self.dt_controller_steps_counter = self.dt_controller_number_of_steps - 1
    self.dt_save_steps_counter = 0

@property
def dt_controller(self):
    return self._dt_controller

@dt_controller.setter
def dt_controller(self, value):
    self._dt_controller = value
    self.dt_controller_number_of_steps = np.rint(value / self._dt_simulation)
    if self.dt_controller_number_of_steps == 0:
        self.dt_controller_number_of_steps = 1
    # Initialize counter at max value to start with update
    self.dt_controller_steps_counter = self.dt_controller_number_of_steps - 1

@property
def dt_save(self):
    return self._dt_save

@dt_save.setter
def dt_save(self, value):
    self._dt_save = value
    self.dt_save_number_of_steps = np.rint(value / self._dt_simulation)
    if self.dt_save_number_of_steps == 0:
        self.dt_save_number_of_steps = 1

    # This counter is initialized at 0 - 0th step is saved manually
    self.dt_save_steps_counter = 0