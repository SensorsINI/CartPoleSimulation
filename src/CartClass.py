# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:27:59 2020

@author: Marcin
"""

# Cart Class
# The file contain the class holding all the parameters and methods
# related to CartPole which are not related to PyQt5 GUI


from numpy import around, random, pi, array
# Shapes used to draw a Cart and the slider
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
# NullLocator is used to disable ticks on the Figures
import matplotlib.pyplot as plt
# rc sets global parameters for matlibplot; transforms is used to rotate the Mast
from matplotlib import transforms, rc
# Import module to interact with OS
import os
# Import module to save history of the simulation as csv file
import csv
# Import module to get a current time and date used to name the files containing the history of simulations
from datetime import datetime

# To detect the latest csv file
import glob
# to keep the loaded data
import pandas as pd


# Interpolate function to create smooth random track
from scipy.interpolate import interp1d, BPoly

from src.globals import *

from types import SimpleNamespace

from src.utilis import wrap_angle_rad


from memory_profiler import profile

import timeit

# Set the font parameters for matplotlib figures
font = {'size': 22}
rc('font', **font)


class Cart:
    def __init__(self,

                 # Variables controlling flow of the program
                 save_history=save_history_globals,

                 # Variables used for physical simulation
                 dt=dt_main_simulation_globals,
                 m=m_globals,  # mass of pend, kg
                 M=M_globals,  # mass of cart, kg
                 L=L_globals,  # half length of pend, m
                 u_max=u_max_globals,  # max cart force, N
                 M_fric=M_fric_globals,  # 1.0, # cart friction, N/m/s
                 J_fric=J_fric_globals,  # 10.0, # friction coefficient on angular velocity, Nm/rad/s
                 v_max=v_max_globals,  # max DC motor speed, m/s, in absense of friction, used for motor back EMF model
                 controlDisturbance=controlDisturbance_globals,  # 0.01, # disturbance, as factor of u_max
                 sensorNoise=sensorNoise_globals,  # 0.01, # noise, as factor of max values

                 # Variables for random trace generation
                 track_relative_complexity=track_relative_complexity_globals,  # Complexity of the random trace, randomly placed target points/s
                 random_length=random_length_globals,  # Number of points in the random length trece
                 interpolation_type=interpolation_type_globals,  # Sets how to interpolate between turning points of random trace
                 turning_points_period=turning_points_period_globals,  # How timeaxis of random trace should be devided
                 start_random_target_position_at=start_random_target_position_at_globals,
                 end_random_target_position_at=end_random_target_position_at_globals,
                 turning_points=turning_points_globals,

                 mode_init = mode_globals  # In which mode the Cart should be initialized
                 ):

        # State of the cart
        self.s = SimpleNamespace()  # s like state
        self.s.position = 0.0
        self.s.positionD = 0.0
        self.s.positionDD = 0.0
        self.s.angle = 0.0
        self.s.angleD = 0.0
        self.s.angleDD = 0.0

        self.u = 0.0
        self.Q = 0.0
        self.target_position = 0.0

        # Other variables controlling flow or initial state of the program
        self.Q_thread_enabled = False  # If True, control input is computed asynchronously to the simulation in a separate thread
        self.Q_max = 1.0

        self.slider_value = 0.0
        self.dt = dt
        self.time = 0.0
        self.rounding_decimals = 5
        self.dict_history = {}
        self.reset_dict_history()
        self.save_history = save_history
        self.csv_filepath = None
        self.random_trace_generated = False
        self.use_pregenerated_target_position = False
        self.stop_at_90 = False

        # Physical parameters of the cart
        self.p = SimpleNamespace()  # p like parameters
        self.p.m = m  # mass of pend, kg
        self.p.M = M  # mass of cart, kg
        self.p.M_fric = M_fric  # cart friction, N/m/s
        self.p.J_fric = J_fric  # friction coefficient on angular velocity, Nm/rad/s
        g = g_globals
        self.p.g = g_globals
        self.p.L = L  # half length of pend, m
        k = k_globals
        self.p.k = k_globals
        self.p.u_max = u_max  # max cart force, N
        self.p.v_max = v_max  # max DC motor speed, m/s, in absence of friction, used for motor back EMF model
        self.p.controlDisturbance = controlDisturbance  # disturbance, as factor of u_max
        self.p.sensorNoise = sensorNoise  # noise, as factor of max values
        self.p.force_damping = 1.0

        # Jacobian of the system linearized around upper equilibrium position
        # x' = f(x)
        # x = [x, v, theta, omega]
        # TODO if parameters change in runtime this Jacobian wont be updated
        self.Jacobian_UP = array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, (-(1 + k) * M_fric) / (-m + (1 + k) * (m + M)), (g * m) / (-m + (1 + k) * (m + M)),
             (-J_fric) / (L * (-m + (1 + k) * (m + M)))],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, (-M_fric) / (L * (-m + (1 + k) * (m + M))), (g * (M + m)) / (L * (-m + (1 + k) * (m + M))),
             -m * (M + m) * J_fric / (L * L * (-m + (1 + k) * (m + M)))],
        ])

        # Array gathering control around equilibrium
        self.B = u_max * array([
            [0.0],
            [(1 + k) / (-m + (1 + k) * (m + M))],
            [0.0],
            [1.0 / (L * (-m + (1 + k) * (m + M)))],
        ])


        self.controller = None
        self.controller_name = ''

        controller_files = glob.glob(PATH_TO_CONTROLLERS + 'controller_' + '*.py')
        self.controller_names = ['manual-stabilization']
        self.controller_names.extend(np.sort(
            [os.path.basename(item)[len('controller_'):-len('.py')].replace('_', '-') for item in controller_files]
        ))

        self.controller_interval = np.inf
        self.controller_interval_threshold = controller_interval_threshold_globals

        # Variables for pre-generated random trace

        self.track_relative_complexity = track_relative_complexity
        self.random_length = random_length
        self.interpolation_type = interpolation_type
        self.turning_points_period = turning_points_period
        self.start_random_target_position_at = start_random_target_position_at
        self.end_random_target_position_at = end_random_target_position_at
        self.turning_points = turning_points
        self.random_track_f = None
        self.new_track_generated = False
        self.t_max_pre = None

        # THE REMAINING PART OF __init__ METHOD CONCERNS DRAWING SETTINGS ONLY

        # DIMENSIONS OF THE DRAWING ONLY!!!
        # NOTHING TO DO WITH THE SIMULATION AND NOT INTENDED TO BE MANIPULATED BY USER !!!
        self.CartLength = 10.0
        self.WheelRadius = 0.5
        self.WheelToMiddle = 4.0
        self.y_plane = 0.0
        self.y_wheel = self.y_plane + self.WheelRadius
        self.MastHight = 10.0  # For drowing only. For calculation see L
        self.MastThickness = 0.05
        self.HalfLength = 50.0  # Length of the track

        # Elements of the drawing
        self.Mast = FancyBboxPatch(xy=(self.s.position - (self.MastThickness / 2.0), 1.25 * self.WheelRadius),
                                   width=self.MastThickness,
                                   height=self.MastHight,
                                   fc='g')

        self.Chassis = FancyBboxPatch((self.s.position - (self.CartLength / 2.0), self.WheelRadius),
                                      self.CartLength,
                                      1 * self.WheelRadius,
                                      fc='r')

        self.WheelLeft = Circle((self.s.position - self.WheelToMiddle, self.y_wheel),
                                radius=self.WheelRadius,
                                fc='y',
                                ec='k',
                                lw=5)

        self.WheelRight = Circle((self.s.position + self.WheelToMiddle, self.y_wheel),
                                 radius=self.WheelRadius,
                                 fc='y',
                                 ec='k',
                                 lw=5)

        self.Slider = Rectangle((0.0, 0.0), self.slider_value, 1.0)
        self.t2 = transforms.Affine2D().rotate(0.0)  # An abstract container for the transform rotating the mast

        # Set starting mode of operation
        self.slider_max = 0.0
        self.mode = mode_init
        self.set_mode(new_mode=mode_init)

        self.save_now = False
        self.controller_index = 1

    # Generates a random target position in a form of a function
    def Generate_Random_Trace_Function(self):

        if (self.turning_points is None) or (self.turning_points == []):

            number_of_turning_points = int(np.floor(self.random_length * self.track_relative_complexity))

            y = 2.0 * (random.random(number_of_turning_points) - 0.5)
            y = y * 0.5 * self.HalfLength

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

        number_of_timesteps = np.ceil(self.random_length / self.dt)
        self.t_max_pre = number_of_timesteps * self.dt

        random_samples = number_of_turning_points - 2 if number_of_turning_points - 2 >= 0 else 0

        # t_init = linspace(0, self.t_max_pre, num=self.track_relative_complexity, endpoint=True)
        if self.turning_points_period == 'random':
            t_init = np.sort(random.uniform(self.dt, self.t_max_pre-self.dt, random_samples))
            t_init = np.insert(t_init, 0, 0.0)
            t_init = np.append(t_init, self.t_max_pre)
        elif self.turning_points_period == 'regular':
            t_init = np.linspace(0, self.t_max_pre, num=random_samples+2, endpoint=True)


        # Try algorithm setting derivative to 0 a each point
        if self.interpolation_type == '0-derivative-smooth':
            yder = [[y[i], 0] for i in range(len(y))]
            self.random_track_f = BPoly.from_derivatives(t_init, yder)
        elif self.interpolation_type == 'linear':
            self.random_track_f = interp1d(t_init, y, kind='linear')
        elif self.interpolation_type == 'previous':
            self.random_track_f = interp1d(t_init, y, kind='previous')
        else:
            raise ValueError('Unknown interpolation type.')


        self.new_track_generated = True

    # Function gathering equations to update the CartPole state:
    # x' = f(x)
    # x = [x, v, theta, omega]
    def Equations_of_motion(self):

        s_next = cartpole_integration(self.s, self.dt)
        self.s.position = s_next.position
        self.s.positionD = s_next.positionD
        self.s.angle = s_next.angle
        self.s.angleD = s_next.angleD

        zero_DD = None
        if self.stop_at_90:
            if self.s.angle >= np.pi/2:
                self.s.angle = np.pi/2
                self.s.angleD = 0.0
                zero_DD = True
            elif self.s.angle <= -np.pi/2:
                self.s.angle = -np.pi/2
                self.s.angleD = 0.0
                zero_DD = True
            else:
                pass

        self.s.angleDD, self.s.positionDD = cartpole_ode(self.p, self.s, self.u)

        if zero_DD:
            self.s.angleDD = 0.0

    # Determine the dimensionales [-1,1] value of the motor power Q
    def Update_Q(self):

        # if self.controller_index == 0:
        #     if abs(self.s.angle) > 2.5 or abs(self.s.angleD) > 1.6 or abs(self.s.positionD) > 20.0:
        #         self.controller_index = 1  # switch controller on
        # elif self.controller_index == 1:
        #     if abs(self.s.angle) < 0.1 and abs(self.s.angleD) < 0.01 and abs(self.s.positionD) < 1.0:
        #         self.controller_index = 0  # swich controller off
        #
        # if self.mode == 0:  # in this case slider corresponds already to the power of the motor
        #     self.Q = self.slider_value
        #
        # else:  # in this case slider gives a target position, lqr regulator
        #
        #     self.controller_interval += self.dt
        #     if self.controller_interval >= self.controller_interval_threshold:
        #
        #         if self.controller_index == 1:
        #             self.Q = self.controller.step(self.s, self.target_position)*(1+0.1*np.random.uniform(-1.0, 1.0))
        #         else:
        #             self.Q = 0
        #
        #         self.controller_interval = 0.0
        #         self.save_now=True

        if self.mode == 0:  # in this case slider corresponds already to the power of the motor
            self.Q = self.slider_value

        else:  # in this case slider gives a target position, lqr regulator

            if self.controller_interval >= self.controller_interval_threshold:
                self.Q = self.controller.step(self.s, self.target_position)
                self.controller_interval = 0.0
                self.save_now = True
        self.controller_interval += self.dt



    # This method changes the internal state of the CartPole
    # from a state at time t to a state at t+dt
    # @profile(precision=4)
    def update_state(self, slider=None, dt=None, save_history=None):

        # Optionally update slider, mode and dt values
        if slider:
            self.slider_value = slider
        if dt:
            self.dt = dt
        if save_history:
            self.save_history = save_history

        if self.use_pregenerated_target_position == True:
            self.target_position = self.random_track_f(self.time)
            if self.target_position > 0.8 * self.HalfLength:
                self.target_position = 0.8 * self.HalfLength
            elif self.target_position < -0.8 * self.HalfLength:
                self.target_position = -0.8 * self.HalfLength
            self.slider_value = self.target_position
        else:
            if self.mode == 0:
                self.target_position = 0.0
            else:
                self.target_position = self.slider_value

        # Calculate the next state
        # self.Equations_of_motion()
        s_next = cartpole_integration(self.s, self.dt)
        self.s.position = s_next.position
        self.s.positionD = s_next.positionD
        self.s.angle = s_next.angle
        self.s.angleD = s_next.angleD

        # Snippet to stop pole at +/- 90 deg if enabled
        zero_DD = None
        if self.stop_at_90:
            if self.s.angle >= np.pi/2:
                self.s.angle = np.pi/2
                self.s.angleD = 0.0
                zero_DD = True
            elif self.s.angle <= -np.pi/2:
                self.s.angle = -np.pi/2
                self.s.angleD = 0.0
                zero_DD = True
            else:
                pass

        # Wrap angle to +/-π
        self.s.angle = wrap_angle_rad(self.s.angle)

        # In case in the next step the wheel of the cart
        # went beyond the track
        # Bump elastically into an (invisible) boarder
        if (abs(self.s.position) + self.WheelToMiddle) > self.HalfLength:
            self.s.positionD = -self.s.positionD

        # Determine the dimensionales [-1,1] value of the motor power Q
        if not self.Q_thread_enabled:
            self.Update_Q()

        # Convert dimensionless motor power to a physical force acting on the Cart
        self.u = Q2u(self.Q, self.p)

        # Update second derivatives
        self.s.angleDD, self.s.positionDD = cartpole_ode(self.p, self.s, self.u)

        if zero_DD:
            self.s.angleDD = 0.0

        # Update the total time of the simulation
        self.time = self.time + self.dt

        # If user chose to save history of the simulation it is saved now
        # It is saved first internally to a dictionary in the Cart instance
        if self.save_history:
            # Saving simulation data
            # TODO: Move dict to pandas
            self.dict_history['time'].append(around(self.time, self.rounding_decimals))
            self.dict_history['dt'].append(around(self.dt, self.rounding_decimals))
            self.dict_history['s.position'].append(around(self.s.position, self.rounding_decimals))
            self.dict_history['s.positionD'].append(around(self.s.positionD, self.rounding_decimals))
            self.dict_history['s.positionDD'].append(around(self.s.positionDD, self.rounding_decimals))
            self.dict_history['s.angle'].append(around(self.s.angle, self.rounding_decimals))
            self.dict_history['s.angleD'].append(around(self.s.angleD, self.rounding_decimals))
            self.dict_history['s.angleDD'].append(around(self.s.angleDD, self.rounding_decimals))
            self.dict_history['u'].append(around(self.u, self.rounding_decimals))
            self.dict_history['Q'].append(around(self.Q, self.rounding_decimals))
            # The target_position is not always meaningful
            # If it is not meaningful all values in this column are set to 0
            self.dict_history['target_position'].append(around(self.target_position, self.rounding_decimals))

            self.dict_history['s.angle.sin'].append(around(np.sin(self.s.angle), self.rounding_decimals))
            self.dict_history['s.angle.cos'].append(around(np.cos(self.s.angle), self.rounding_decimals))
        else:
            self.reset_dict_history()

        # Return the state of the CartPole
        return self.s.position, self.s.positionD, self.s.positionDD, \
               self.s.angle, self.s.angleD, self.s.angleDD, \
               self.u

    # This method only returns the state of the CartPole instance 
    def get_state(self):
        return self.s.position, self.s.positionD, self.s.positionDD, \
               self.s.angle, self.s.angleD, self.s.angleDD, \
               self.u

    # This method resets the internal state of the CartPole instance
    def reset_state(self, reset_mode=1):
        if reset_mode == 0:
            self.s.position = self.s.positionD = self.s.positionDD = 0.0
            self.s.angle = self.s.angleD = self.s.angleDD = 0.0
            self.Q = self.u = 0.0
            self.slider_value = 0.0
            self.time = 0.0
        else:
            # You can change here with which initial parameters you wish to start the simulation
            self.s.position = 0.0
            self.s.positionD = 0.0
            self.s.angle = (2.0 * random.normal() - 1.0) * pi / 180.0
            self.s.angleD = 0.0
            self.s.angleDD = 0.0

            self.Q = 0.0
            self.u = 0.0

            self.s.angleDD, self.s.positionDD = cartpole_ode(self.p, self.s, self.u)

            self.slider_value = 0.0

            self.time = 0.0

    # This method draws elements and set properties of the CartPole figure
    # which do not change at every frame of the animation
    def draw_constant_elements(self, fig, AxCart, AxSlider):

        # Delete all elements of the Figure
        AxCart.clear()
        AxSlider.clear()

        ## Upper chart with Cart Picture
        # Set x and y limits
        AxCart.set_xlim((-self.HalfLength * 1.1, self.HalfLength * 1.1))
        AxCart.set_ylim((-1.0, 15.0))
        # Remove ticks on the y-axes
        AxCart.yaxis.set_major_locator(plt.NullLocator())  # NullLocator is used to disable ticks on the Figures

        # Draw track
        Floor = Rectangle((-self.HalfLength, -1.0),
                          2 * self.HalfLength,
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
        # Apply scaling
        AxSlider.set_aspect("auto")

        return fig, AxCart, AxSlider

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

    # This method updates the elements of the Cart Figure which change at every frame.
    # Not that these elements are not ploted directly by this method
    # but rather returned as objects which can be used by another function
    # e.g. animation function from matplotlib package
    def update_drawing(self):

        # Draw mast
        mast_position = (self.s.position - (self.MastThickness / 2.0))
        self.Mast.set_x(mast_position)
        # Draw rotated mast
        t21 = transforms.Affine2D().translate(-mast_position, -1.25 * self.WheelRadius)
        if ANGLE_CONVENTION == 'CLOCK-NEG':
            t22 = transforms.Affine2D().rotate(self.s.angle)
        elif ANGLE_CONVENTION == 'CLOCK-POS':
            t22 = transforms.Affine2D().rotate(-self.s.angle)
        else:
            raise ValueError('Unknown angle convention')
        t23 = transforms.Affine2D().translate(mast_position, 1.25 * self.WheelRadius)
        self.t2 = t21 + t22 + t23
        # Draw Chassis
        self.Chassis.set_x(self.s.position - (self.CartLength / 2.0))
        # Draw Wheels
        self.WheelLeft.center = (self.s.position - self.WheelToMiddle, self.y_wheel)
        self.WheelRight.center = (self.s.position + self.WheelToMiddle, self.y_wheel)
        # Draw SLider
        self.Slider.set_width(self.slider_value)

        return self.Mast, self.t2, self.Chassis, self.WheelRight, self.WheelLeft, self.Slider

    # This method resets the dictionary keeping the history of simulation
    def reset_dict_history(self):
        self.dict_history = {'time': [around(self.time, self.rounding_decimals)],
                             'dt': [around(self.dt, self.rounding_decimals)],
                             's.position': [around(self.s.position, self.rounding_decimals)],
                             's.positionD': [around(self.s.positionD, self.rounding_decimals)],
                             's.positionDD': [around(self.s.positionDD, self.rounding_decimals)],
                             's.angle': [around(self.s.angle, self.rounding_decimals)],
                             's.angleD': [around(self.s.angleD, self.rounding_decimals)],
                             's.angleDD': [around(self.s.angleDD, self.rounding_decimals)],
                             'u': [around(self.u, self.rounding_decimals)],
                             'Q': [around(self.Q, self.rounding_decimals)],
                             'target_position': [around(self.target_position, self.rounding_decimals)],
                             's.angle.sin': [around(np.sin(self.s.angle), self.rounding_decimals)],
                             's.angle.cos': [around(np.cos(self.s.angle), self.rounding_decimals)]}



    # This method saves the dictionary keeping the history of simulation to a .csv file
    def save_history_csv(self, csv_name=None, init=True, iter=True):

        if init:
            # Make folder to save data (if not yet existing)
            try:
                os.makedirs('./data')
            except FileExistsError:
                pass

            # Set path where to save the data
            if csv_name is None or csv_name == '':
                self.csv_filepath = './data/' + 'CP_' + self.controller_name + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')) + '.csv'
            else:
                self.csv_filepath = './data/' + csv_name
                if csv_name[-4:] != '.csv':
                    self.csv_filepath += '.csv'

                # If such file exists, add index to the end (do not overwrite)
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

                writer.writerow(['# ' + 'This is CartPole experiment from {} at time {}'
                                .format(datetime.now().strftime('%d.%m.%Y'), datetime.now().strftime('%H:%M:%S'))])
                if iter:
                    writer.writerow(['# Number of data points: {}'.format(len(self.dict_history['time']))])
                else:
                    writer.writerow(['# Number of data points: data saved online'])
                writer.writerow(['# Controller: {}'.format(self.controller_name)])

                writer.writerow(['#'])
                writer.writerow(['# Parameters:'])
                for k in self.p.__dict__:
                    writer.writerow(['# ' + k + ': ' + str(getattr(self.p, k))])
                writer.writerow(['#'])
                writer.writerow(['# Data:'])
                writer.writerow(self.dict_history.keys())

        if iter:
            # Write the .csv file
            if self.save_now:
                with open(self.csv_filepath, "a") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerows(zip(*self.dict_history.values()))
                self.save_now=False
            else:
                pass

    # load csv file with experiment recording (e.g. for replay)
    def load_history_csv(self, csv_name=None, visualisation_only=True):
        # Set path where to save the data
        if csv_name is None or csv_name == '':
            # get the latest file
            try:
                list_of_files = glob.glob('./data/' + '/*.csv')
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
                file_path = os.path.join('data', filename)
                if not os.path.isfile(file_path):
                    print(
                        'Cannot load: There is no experiment recording file with name {} at local folder or in {}'.format(
                            filename, './data/'))
                    return False

        # Get race recording
        print('Loading file {}'.format(file_path))
        try:
            data: pd.DataFrame = pd.read_csv(file_path, comment='#')  # skip comment lines starting with #
        except Exception as e:
            print('Cannot load: Caught {} trying to read CSV file {}'.format(e, file_path))
            return False

        if visualisation_only:
            data = data[['time', 'dt', 's.position', 's.positionD', 's.angle', 'u', 'target_position', 'Q']]

        return data

    # Set mode i.e. select controller
    def set_mode(self, new_mode=0):

        self.mode = new_mode
        self.controller_name = self.controller_names[self.mode]

        # Load controller
        if self.controller_name == 'manual-stabilization':
            self.controller = None
        else:
            controller_full_name = 'controller_' + self.controller_name.replace('-', '_')
            path_import = PATH_TO_CONTROLLERS[2:].replace('/','.').replace(r'\\', '.')
            import_str = 'from ' + path_import + controller_full_name + ' import ' + controller_full_name
            exec(import_str)
            if self.controller_name == 'lqr':
                self.controller = eval(controller_full_name + '(self.Jacobian_UP, self.B)')
            else:
                self.controller = eval(controller_full_name + '()')

        # Set the maximal allowed value of the slider
        if self.controller_name == 'manual-stabilization':
            self.slider_max = self.Q_max
        else:
            self.slider_max = self.HalfLength

    # Method printing the parameters of the CartPole over time after an experiment
    def summary_plots(self):

        fig, axs = plt.subplots(4, 1, figsize=(18, 14), sharex=True)  # share x axis so zoom zooms all plots

        # Plot angle error
        axs[0].set_ylabel("Angle (deg)", fontsize=18)
        axs[0].plot(array(self.dict_history['time']), array(self.dict_history['s.angle']) * 180.0 / pi,
                    'b', markersize=12, label='Ground Truth')
        axs[0].tick_params(axis='both', which='major', labelsize=16)

        # Plot position
        axs[1].set_ylabel("position (m)", fontsize=18)
        axs[1].plot(self.dict_history['time'], self.dict_history['s.position'], 'g', markersize=12,
                    label='Ground Truth')
        axs[1].tick_params(axis='both', which='major', labelsize=16)

        # Plot motor input command
        axs[2].set_ylabel("motor (N)", fontsize=18)
        axs[2].plot(self.dict_history['time'], self.dict_history['u'], 'r', markersize=12,
                    label='motor')
        axs[2].tick_params(axis='both', which='major', labelsize=16)

        # Plot target position
        axs[3].set_ylabel("position target (m)", fontsize=18)
        axs[3].plot(self.dict_history['time'], self.dict_history['target_position'], 'k')
        axs[3].tick_params(axis='both', which='major', labelsize=16)

        axs[3].set_xlabel('Time (s)', fontsize=18)

        fig.align_ylabels()

        plt.show()

        return fig, axs
