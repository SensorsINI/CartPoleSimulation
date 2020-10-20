# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:27:59 2020

@author: Marcin
"""

# Cart Class
# The file contain the class holding all the parameters and methods
# related to CartPole which are not related to PyQt5 GUI


from numpy import around, random, pi, sin, cos, array, diag
import numpy as np
# Shapes used to draw a Cart and the slider
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
# NullLocator is used to disable ticks on the Figures
from matplotlib.pyplot import NullLocator
# rc sets global parameters for matlibplot; transforms is used to rotate the Mast
from matplotlib import transforms, rc
# Import module to interact with OS
import os
# Import module to save history of the simulation as csv file
import csv
# Import module to get a current time and date used to name the files containing the history of simulations
from datetime import datetime
import timeit

from controllers.controller_lqr import controller_lqr
from controllers.controller_do_mpc import controller_do_mpc

# Interpolate function to create smooth random track
from scipy.interpolate import interp1d

from globals import *

from types import SimpleNamespace

from math import fmod

def normalize_angle_rad(angle):
    Modulo = fmod(angle, 2*np.pi)  # positive modulo
    if Modulo < -np.pi:
        angle = Modulo+2*np.pi
    elif Modulo > np.pi:
        angle = Modulo-2*np.pi
    else:
        angle = Modulo
    return angle



# Set the font parameters for matplotlib figures
font = {'size': 22}
rc('font', **font)


class Cart:
    def __init__(self,

                 # Variables controlling flow of the program
                 save_history=save_history_globals,

                 # Variables used for physical simulation
                 dt=dt_globals,
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
                 N=N_globals,  # Complexity of the random trace, number of random points used for interpolation
                 random_length=random_length_globals  # Number of points in the random length trece
                 ):

        # State of the cart
        self.s = SimpleNamespace()  # s like state
        self.s.CartPosition = 0.0
        self.s.CartPositionD = 0.0
        self.s.CartPositionDD = 0.0
        self.s.angle = 0.0
        self.s.angleD = 0.0
        self.s.angleDD = 0.0

        self.u = 0.0
        self.Q = 0.0
        self.PositionTarget = 0.0

        # Other variables controlling flow or initial state of the program
        self.Q_thread_enabled = False  # If True, control input is computed asynchronously to the simulation in a separate thread
        self.mode = 0
        self.Q_max = 1.0
        # Set the maximal allowed value of the slider dependant on the mode of simulation
        if self.mode == 0:
            self.slider_max = self.Q_max
        elif self.mode == 1 or self.mode == 2:
            self.slider_max = self.HalfLength
        self.slider_value = 0.0
        self.dt = dt
        self.time_total = 0.0
        self.dict_history = {}
        self.reset_dict_history()
        self.save_history = save_history
        self.random_trace_generated = False
        self.play_pregenerated = False

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

        # Cost matrices for LQR controller
        self.Q_matrix = diag([10.0, 1.0, 1.0, 1.0])  # How much to punish x, v, theta, omega
        self.R_matrix = 1.0e9  # How much to punish Q

        self.controller_lqr = controller_lqr(self.Jacobian_UP, self.B, self.Q_matrix, self.R_matrix)
        self.controller_do_mpc = controller_do_mpc()
        self.controller = None

        # Variables for pre-generated random trace

        self.N = int(N)
        self.random_length = random_length
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
        self.Mast = FancyBboxPatch(xy=(self.s.CartPosition - (self.MastThickness / 2.0), 1.25 * self.WheelRadius),
                                   width=self.MastThickness,
                                   height=self.MastHight,
                                   fc='g')

        self.Chassis = FancyBboxPatch((self.s.CartPosition - (self.CartLength / 2.0), self.WheelRadius),
                                      self.CartLength,
                                      1 * self.WheelRadius,
                                      fc='r')

        self.WheelLeft = Circle((self.s.CartPosition - self.WheelToMiddle, self.y_wheel),
                                radius=self.WheelRadius,
                                fc='y',
                                ec='k',
                                lw=5)

        self.WheelRight = Circle((self.s.CartPosition + self.WheelToMiddle, self.y_wheel),
                                 radius=self.WheelRadius,
                                 fc='y',
                                 ec='k',
                                 lw=5)

        self.Slider = Rectangle((0.0, 0.0), self.slider_value, 1.0)
        self.t2 = transforms.Affine2D().rotate(0.0)  # An abstract container for the transform rotating the mast

    def Generate_Random_Trace_Function(self):
        # t_pre = arange(0, self.random_length)*self.dt
        self.t_max_pre = (self.random_length - 1) * self.dt

        # t_init = linspace(0, self.t_max_pre, num=self.N, endpoint=True)
        t_init = random.uniform(self.dt, self.t_max_pre, self.N)
        t_init = np.insert(t_init, 0, 0.0)
        t_init = np.append(t_init, self.t_max_pre)

        y = 2.0 * (random.random(self.N) - 0.5)

        y = y * 0.8 * self.HalfLength / max(abs(y))
        y = np.insert(y, 0, 0.0)
        y = np.append(y, 0.0)

        self.random_track_f = interp1d(t_init, y, kind='cubic')
        self.new_track_generated = True

    # Function gathering equations to update the CartPole state:
    # x' = f(x)
    # x = [x, v, theta, omega]
    def Equations_of_motion(self):

        self.s.CartPosition, self.s.CartPositionD, self.s.angle, self.s.angleD = \
            cartpole_integration(self.s, self.dt)

        self.s.angleDD, self.s.CartPositionDD = cartpole_ode(self.p, self.s, self.u)

    # Determine the dimensionales [-1,1] value of the motor power Q
    def Update_Q(self):

        if self.mode == 1:  # in this case slider gives a target position, lqr regulator
            tic = timeit.default_timer()
            state = array(
                [[self.s.CartPosition - self.PositionTarget], [self.s.CartPositionD], [self.s.angle], [self.s.angleD]])
            self.Q = self.controller.step(state)
            toc = timeit.default_timer()

            print("Time to find control input = {} ms".format((toc - tic) * 1000.0))

        elif self.mode == 2:  # in this case slider gives a target position, do-mpc regulator

            tic = timeit.default_timer()
            self.Q = self.controller.step(self.s)
            toc = timeit.default_timer()

            print("Time to find control input = {} ms".format((toc - tic) * 1000.0))

            if self.Q > 1.0:
                print('Q to big! ' + str(self.Q))
            elif self.Q < -1.0:
                print('Q to small! ' + str(self.Q))

        elif self.mode == 0:  # in this case slider corresponds already to the power of the motor
            self.Q = self.slider_value


    # This method changes the internal state of the CartPole
    # from a state at time t to a state at t+dt   
    def update_state(self, slider=None, mode=None, dt=None, save_history=True):

        # Optionally update slider, mode and dt values
        if slider:
            self.slider_value = slider
        if mode:
            self.mode = mode
        if dt:
            self.dt = dt
        self.save_history = save_history

        if self.play_pregenerated == True:
            self.PositionTarget = self.random_track_f(self.time_total)
            if self.PositionTarget > 0.8 * self.HalfLength:
                self.PositionTarget = 0.8 * self.HalfLength
            elif self.PositionTarget < -0.8 * self.HalfLength:
                self.PositionTarget = -0.8 * self.HalfLength
            self.slider_value = self.PositionTarget
        else:
            if self.mode == 1 or self.mode == 2:
                self.PositionTarget = self.slider_value
            elif self.mode == 0:
                self.PositionTarget = 0.0

        # Calculate the next state
        self.Equations_of_motion()

        # Normalize angle

        self.s.angle = normalize_angle_rad(self.s.angle)

        # In case in the next step the wheel of the cart
        # went beyond the track
        # Bump elastically into an (invisible) boarder
        if (abs(self.s.CartPosition) + self.WheelToMiddle) > self.HalfLength:
            self.s.CartPositionD = -self.s.CartPositionD

        # Determine the dimensionales [-1,1] value of the motor power Q
        if not self.Q_thread_enabled:
            self.Update_Q()

        self.u = Q2u(self.Q, self.p)

        # Update the total time of the simulation
        self.time_total = self.time_total + self.dt

        # If user chose to save history of the simulation it is saved now
        # It is saved first internally to a dictionary in the Cart instance
        if self.save_history:
            # Saving simulation data
            self.dict_history['time'].append(around(self.time_total, 4))
            self.dict_history['deltaTimeMs'].append(around(self.dt * 1000.0, 3))
            self.dict_history['position'].append(around(self.s.CartPosition, 3))
            self.dict_history['positionD'].append(around(self.s.CartPositionD, 4))
            self.dict_history['positionDD'].append(around(self.s.CartPositionDD, 4))
            self.dict_history['angleErr'].append(around(self.s.angle, 4))
            self.dict_history['angleD'].append(around(self.s.angleD, 4))
            self.dict_history['angleDD'].append(around(self.s.angleDD, 4))
            self.dict_history['motor'].append(around(self.u, 4))
            # The PositionTarget is not always meaningful
            # If it is not meaningful all values in this column are set to 0
            self.dict_history['PositionTarget'].append(around(self.PositionTarget, 4))

        # Return the state of the CartPole
        return self.s.CartPosition, self.s.CartPositionD, self.s.CartPositionDD, \
               self.s.angle, self.s.angleD, self.s.angleDD, \
               self.u

    # This method only returns the state of the CartPole instance 
    def get_state(self):
        return self.s.CartPosition, self.s.CartPositionD, self.s.CartPositionDD, \
               self.s.angle, self.s.angleD, self.s.angleDD, \
               self.u

    # This method resets the internal state of the CartPole instance
    def reset_state(self):
        self.s.CartPosition = 0.0
        self.s.CartPositionD = 0.0
        self.s.CartPositionDD = 0.0
        self.s.angle = (2.0 * random.normal() - 1.0) * pi / 180.0
        self.s.angleD = 0.0
        self.s.angleDD = 0.0

        self.u = 0.0

        self.dt = 0.002

        self.slider_value = 0.0

        self.time_total = 0.0

    # This method draws elements and set properties of the CartPole figure
    # which do not change at every frame of the animation
    def draw_constant_elements(self, fig, AxCart, AxSlider):

        # Get the appropriate max of slider depending on the mode of operation
        if self.mode == 0:
            self.slider_max = self.Q_max
        elif self.mode == 1:
            self.slider_max = self.HalfLength

        # Delete all elements of the Figure
        AxCart.clear()
        AxSlider.clear()

        ## Upper chart with Cart Picture
        # Set x and y limits
        AxCart.set_xlim((-self.HalfLength * 1.1, self.HalfLength * 1.1))
        AxCart.set_ylim((-1.0, 15.0))
        # Remove ticks on the y-axes
        AxCart.yaxis.set_major_locator(NullLocator())

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
        AxSlider.yaxis.set_major_locator(NullLocator())
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
        mast_position = (self.s.CartPosition - (self.MastThickness / 2.0))
        self.Mast.set_x(mast_position)
        # Draw rotated mast
        t21 = transforms.Affine2D().translate(-mast_position, -1.25 * self.WheelRadius)
        t22 = transforms.Affine2D().rotate(self.s.angle)
        t23 = transforms.Affine2D().translate(mast_position, 1.25 * self.WheelRadius)
        self.t2 = t21 + t22 + t23
        # Draw Chassis
        self.Chassis.set_x(self.s.CartPosition - (self.CartLength / 2.0))
        # Draw Wheels
        self.WheelLeft.center = (self.s.CartPosition - self.WheelToMiddle, self.y_wheel)
        self.WheelRight.center = (self.s.CartPosition + self.WheelToMiddle, self.y_wheel)
        # Draw SLider
        self.Slider.set_width(self.slider_value)

        return self.Mast, self.t2, self.Chassis, self.WheelRight, self.WheelLeft, self.Slider

    # This method resets the dictionary keeping the history of simulation
    def reset_dict_history(self):
        self.dict_history = {'time': [0.0],
                             'deltaTimeMs': [0.0],
                             'position': [self.s.CartPosition],
                             'positionD': [self.s.CartPositionD],
                             'positionDD': [self.s.CartPositionDD],
                             'angleErr': [self.s.angle],
                             'angleD': [self.s.angleD],
                             'angleDD': [self.s.angleDD],
                             'motor': [self.u],
                             'PositionTarget': [self.PositionTarget]}

    # This method saves the dictionary keeping the history of simulation to a .csv file
    def save_history_csv(self):

        # Make folder to save data (if not yet existing)
        try:
            os.makedirs('save')
        except:
            pass

        # Set path where to save the data
        logpath = './save/' + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + '.csv'
        # Write the .csv file
        with open(logpath, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.dict_history.keys())
            writer.writerows(zip(*self.dict_history.values()))
