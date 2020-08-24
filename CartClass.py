# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:27:59 2020

@author: Marcin
"""

# Cart Class
# The file contain the class holding all the parameters and methods
# related to CartPole which are not related to PyQt5 GUI


from numpy import around, random, pi, sin, cos, sign, array, identity, dot, asscalar, diag, arange, insert, linspace
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


# Interpolate function to create smooth random track
from scipy.interpolate import interp1d
# Import functions to create lqr controller
import scipy.linalg

#Set the font parameters for matplotlib figures
font = {'size'   : 22}
rc('font', **font)

def controller_lqr(A, B, Q, R):
    #From https://github.com/markwmuller/controlpy/blob/master/controlpy/synthesis.py#L8
    """Solve the continuous time LQR controller for a continuous time system.
    
    A and B are system matrices, describing the systems dynamics:
     dx/dt = A x + B u
    
    The controller minimizes the infinite horizon quadratic cost function:
     cost = integral (x.T*Q*x + u.T*R*u) dt
    
    where Q is a positive semidefinite matrix, and R is positive definite matrix.
    
    Returns K, X, eigVals:
    Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
    The optimal input is then computed as:
     input: u = -K*x
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = scipy.linalg.solve_continuous_are(A, B, Q, R)
    
    #compute the LQR gain
    if np.array(R).ndim == 0:
        Ri = 1.0/R
    else:
        Ri = np.linalg.inv(R)
    
    K = np.dot(Ri,(np.dot(B.T,X)))
    
    eigVals = np.linalg.eigvals(A-np.dot(B,K))
    
    return K, X, eigVals


class Cart:                  
    def __init__ (self,             
                  
                  # (Initial) State of the cart
                  # It is only a state after initialization, not after reset
                  # For the later see reset_state function
                  CartPosition = 0.0,
                  CartPositionD = 0.0,
                  CartPositionDD = 0.0,
                  angle = (2.0*random.normal()-1.0)  * pi/180.0,
                  angleD = 0.0,
                  angleDD = 0.0,
                  ueff = 0.0,
                  Q = 0.0,
                  
                  # Other variables controlling flow of the program
                  Q_max = 1,
                  slider_value = 0.0,
                  mode = 0,
                  dt = 0.002,
                  save_history = True,
                  
                  # Variables used for physical simulation
                  m = 0.1, # mass of pend, kg
                  M = 1.0, # mass of cart, kg
                  Mfric = 0.0,#1.0, # cart friction, N/m/s
                  Jfric = 0.0,#10.0, # friction coefficient on angular velocity, Nm/rad/s
                  g = 9.8, # gravity, m/s^2
                  L = 2.0, # half length of pend, m
                  k = 4.0/3.0, # Dimensionless factor, for moment of inertia of the pend (with L beeing half if the lengh)
                  umax = 600.0, # max cart force, N
                  maxv = 10.0, # max DC motor speed, m/s, in absense of friction, used for motor back EMF model
                  controlDisturbance = 0.0,#0.01, # disturbance, as factor of umax
                  sensorNoise = 0.0,#0.01, # noise, as factor of max values
                  
                  # Variables for the controller
                  angle_safety_limit = 6.0,
                  kx = 0.5,
                  kxd = 5.0,
                  ka = 0.5,
                  kad = 5.0,
                  
                  # Dimensions of the drawing
                  CartLength = 10.0,
                  WheelRadius = 0.5,
                  WheelToMiddle = 4.0,
                  MastHight = 10.0, # Only drawing, not the one used for physical simulation!
                  MastThickness = 0.05,
                  y_plane = 0.0,
                  HalfLength = 50.0,
                  
                  # Variables for random trace generation
                  N = 5, # Complexity of the random trace, number of random points used for interpolation
                  random_length = 5e4 # Number of points in the random length trece
                  ):
    

        
        
        # State of the cart
        self.CartPosition = CartPosition
        self.CartPositionD = CartPositionD
        self.CartPositionDD = CartPositionDD
        self.angle = angle
        self.angleD = angleD
        self.angleDD = angleDD
        self.ueff = ueff
        self.Q = Q
        self.PositionTarget = 0.0
        
        #Other variables controlling flow of the program
        self.mode = mode
        self.Q_max = Q_max
        # Set the maximal allowed value of the slider dependant on the mode of simulation
        if self.mode == 0:                 
            self.slider_max = self.Q_max
        elif self.mode == 1:
            self.slider_max = self.HalfLength
        self.slider_value = slider_value
        self.dt = dt
        self.time_total = 0.0
        self.dict_history = {}
        self.reset_dict_history()
        self.save_history = save_history
        self.random_trace_generated = False
        self.play_pregenerated = False
        
        #Variables for the controller
        self.angle_safety_limit = angle_safety_limit
        self.kx = kx
        self.kxd = kxd
        self.ka = ka
        self.kad = kad
        
        #Physical parameters of the cart
        self.m = m # mass of pend, kg
        self.M = M # mass of cart, kg
        self.Mfric = Mfric # cart friction, N/m/s
        self.Jfric = Jfric # friction coefficient on angular velocity, Nm/rad/s
        self.g = g # gravity, m/s^2
        self.L = L # half length of pend, m
        self.k = k # Dimensionless factor, for moment of inertia of the pend (with L beeing half if the lengh)
        self.umax = umax # max cart force, N
        self.maxv = maxv # max DC motor speed, m/s, in absense of friction, used for motor back EMF model 
        self.controlDisturbance = controlDisturbance # disturbance, as factor of umax
        self.sensorNoise = sensorNoise # noise, as factor of max values
        self.force_damping = 1.0
        
        # Jacobian of the system linearized around upper equilibrium position
        # x' = f(x)
        # x = [x, v, theta, omega]
        self.Jacobian_UP = array([ 
                                    [0.0,              1.0,                     0.0,                                                0.0], 
                                    [0.0, (-(1+k)*Mfric)/(-m+(1+k)*(m+M)),    (g*m)/(-m+(1+k)*(m+M)),         (-Jfric)/(L*(-m+(1+k)*(m+M)))],
                                    [0.0,              0.0,                     0.0,                                                1.0], 
                                    [0.0, (-Mfric)/(L*(-m+(1+k)*(m+M))),    (g*(M+m))/(L*(-m+(1+k)*(m+M))),         -m*(M+m)*Jfric/(L*L*(-m+(1+k)*(m+M)))],\
                                    ])
        
        #Array gathering control around equilibrium
        self.B = self.umax * array([
                                    [0.0],
                                    [(1+k)/(-m+(1+k)*(m+M))],
                                    [0.0],
                                    [1.0/(L*(-m+(1+k)*(m+M)))],
                                    ])
        
        # Cost matrices for LQR controller
        self.Q_matrix = diag([10.0, 1.0, 1.0, 1.0]) # How much to punish x, v, theta, omega
        self.R_matrix = 1.0e9  #How much to punish Q
        

        self.K, _, _ = controller_lqr(self.Jacobian_UP, self.B, self.Q_matrix, self.R_matrix)
        
        
        # Variables for pregenerated random traca

        self.N = int(N)
        self.random_length = random_length
        self.random_track_f =  None
        self.new_track_generated = False
        self.t_max_pre = None

        
        #Dimensions of the drawing
        self.CartLength = CartLength
        self.WheelRadius = WheelRadius
        self.WheelToMiddle = WheelToMiddle
        self.y_plane = y_plane
        self.y_wheel = self.y_plane+self.WheelRadius
        self.MastHight = MastHight # For drowing only. For calculation see L
        self.MastThickness = MastThickness
        self.HalfLength = HalfLength # Length of the track
        
        #Elements of the drawing
        self.Mast = FancyBboxPatch((self.CartPosition-(self.MastThickness/2.0), 1.25*self.WheelRadius),
                              self.MastThickness,
                              self.MastHight,
                              fc='g')
        
        self.Chassis = FancyBboxPatch((self.CartPosition-(self.CartLength/2.0), self.WheelRadius),
                              self.CartLength,
                              1*self.WheelRadius,
                              fc='r')
        
        self.WheelLeft = Circle((self.CartPosition-self.WheelToMiddle,self.y_wheel),
                     radius=self.WheelRadius,
                     fc='y',
                     ec = 'k',
                     lw = 5)
        
        self.WheelRight = Circle((self.CartPosition+self.WheelToMiddle, self.y_wheel),
                             radius=self.WheelRadius,
                             fc='y',
                             ec = 'k',
                             lw = 5)
        
        self.Slider = Rectangle((0.0,0.0), slider_value, 1.0)
        self.t2 = transforms.Affine2D().rotate(0.0) # An abstract container for the transform rotating the mast
        
        
    def Generate_Random_Trace_Function(self):
        #t_pre = arange(0, self.random_length)*self.dt
        self.t_max_pre = (self.random_length-1)*self.dt
        
        #t_init = linspace(0, self.t_max_pre, num=self.N, endpoint=True)
        t_init = random.uniform(self.dt, self.t_max_pre, self.N)
        t_init = np.insert(t_init, 0, 0.0)
        t_init = np.append(t_init, self.t_max_pre)
        
        y = 2.0*(random.random(self.N)-0.5)
        
        y = y*0.8*self.HalfLength/max(abs(y))
        y = np.insert(y, 0,0.0)
        y = np.append(y, 0.0)
        
        self.random_track_f = interp1d(t_init, y, kind='cubic')
        self.new_track_generated = True
        
        
    # Function gathering equations to update the CartPole state:
    # x' = f(x)
    # x = [x, v, theta, omega]
    def Equations_of_motion(self):
        
        self.angleD = self.angleD+self.angleDD*self.dt
        self.CartPositionD = self.CartPositionD+self.CartPositionDD*self.dt
        self.angle = self.angle+self.angleD*self.dt
        self.CartPosition = self.CartPosition+self.CartPositionD*self.dt
        
        ca = cos(self.angle)
        sa = sin(self.angle)
        A = (self.k+1)*(self.M+self.m)-self.m*(ca**2)
        
        self.angleDD = (self.g*(self.m+self.M)*sa-((self.Jfric*(self.m+self.M)*self.angleD)/(self.L*self.m))-ca*(self.m*self.L*(self.angleD**2)*sa+self.Mfric*self.CartPositionD)+ca*self.ueff)/(A*self.L)
        self.CartPositionDD = (self.m*self.g*sa*ca-((self.Jfric*self.angleD*ca)/(self.L))-(self.k+1)*(self.m*self.L*(self.angleD**2)*sa+self.Mfric*self.CartPositionD)+(self.k+1)*self.ueff)/A

    # Determine the dimensionales [-1,1] value of the motor power Q
    def Update_Q(self):

        if self.mode == 1:  # in this case slider gives a target position
            
            # Control with LQR
            self.force_damping = (1-abs(self.CartPositionD)/self.maxv)
            if self.force_damping < 0.1:
                self.CartPositionD = 0.9*self.maxv*sign(self.CartPositionD)
                self.force_damping = 0.1
                
                
            state = array([[self.CartPosition-self.PositionTarget], [self.CartPositionD], [self.angle], [self.angleD]])
            self.Q = asscalar(dot(-self.K,state))/self.force_damping
            
            if self.Q > 1.0:
                print('Q to big! ' + str(self.Q))
                #self.Q = 1.0
            elif self.Q < -1.0:
                print('Q to small! ' + str(self.Q))
                #self.Q = -1.0
            
        elif self.mode == 0: # in this case slider corresponds already to the power of the motor
            self.Q = self.slider_value
    
    # Calculate the force created by the motor
    def Update_ueff(self):
        self.ueff = (self.umax*self.Q)*self.force_damping # dumb model of EMF of motor, Q is drive -1:1 range
        self.ueff = self.ueff+self.controlDisturbance*random.normal()*self.umax # noise on control    
        
    # This method changes the internal state of the CartPole
    # from a state at time t to a state at t+dt   
    def update_state(self, slider = None, mode = None, dt = None, save_history = True):
        
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
            if self.PositionTarget > 0.8*self.HalfLength:
                self.PositionTarget = 0.8*self.HalfLength
            elif self.PositionTarget < -0.8*self.HalfLength:
                self.PositionTarget = -0.8*self.HalfLength
            self.slider_value = self.PositionTarget
        else:
            if self.mode == 1:
                self.PositionTarget = self.slider_value
            elif self.mode == 0:
                self.PositionTarget = 0.0
        
        
        # Calculate the next state
        self.Equations_of_motion()
        
        # In case in the next step the wheel of the cart
        # went beyond the track
        # Bump elastically into an (invisible) boarder
        if (abs(self.CartPosition)+self.WheelToMiddle)>self.HalfLength:
            self.CartPositionD = -self.CartPositionD
        
        # Determine the dimensionales [-1,1] value of the motor power Q
        self.Update_Q()
        
        self.Update_ueff()
        
        #Update the total time of the simulation
        self.time_total = self.time_total + self.dt
        
        
        # If user chose to save history of the simulation it is saved now
        # It is saved first internally to a dictionary in the Cart instance
        if self.save_history:
            # Saving simulation data
            self.dict_history['time'].append(around(self.time_total, 4))
            self.dict_history['deltaTimeMs'].append(around(self.dt*1000.0,3))
            self.dict_history['position'].append(around(self.CartPosition,3))
            self.dict_history['positionD'].append(around(self.CartPositionD,4))
            self.dict_history['positionDD'].append(around(self.CartPositionDD,4))
            self.dict_history['angleErr'].append(around(self.angle,4))
            self.dict_history['angleD'].append(around(self.angleD,4))
            self.dict_history['angleDD'].append(around(self.angleDD,4))
            self.dict_history['motor'].append(around(self.ueff,4))
            # The PositionTarget is not always meaningful
            # If it is not meaningful all values in this column are set to 0
            self.dict_history['PositionTarget'].append(around(self.PositionTarget,4))
        
        # Return the state of the CartPole
        return self.CartPosition, self.CartPositionD, self.CartPositionDD, \
                self.angle, self.angleD, self.angleDD, \
                    self.ueff
                    
                    
    # This method only returns the state of the CartPole instance 
    def get_state(self):
        return self.CartPosition, self.CartPositionD, self.CartPositionDD, \
                self.angle, self.angleD, self.angleDD, \
                    self.ueff
    
    
    # This method resets the internal state of the CartPole instance
    def reset_state(self):
        self.CartPosition = 0.0
        self.CartPositionD = 0.0
        self.CartPositionDD = 0.0
        self.angle = (2.0*random.normal()-1.0)  * pi/180.0
        self.angleD = 0.0
        self.angleDD = 0.0
        
        self.ueff = 0.0
        
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
        AxCart.set_xlim((-self.HalfLength*1.1,self.HalfLength*1.1))
        AxCart.set_ylim((-1.0,15.0))
        # Remove ticks on the y-axes
        AxCart.yaxis.set_major_locator(NullLocator())
        
        # Draw track
        Floor = Rectangle((-self.HalfLength, -1.0),
                                  2*self.HalfLength,
                                  1.0,
                                  fc='brown')
        AxCart.add_patch(Floor)
        
        # Draw an invisible point at constant position
        # Thanks to it the axes is drawn high enough for the mast
        InvisiblePointUp = Rectangle((0,self.MastHight+2.0),
                              self.MastThickness,
                              0.0001,
                              fc='w',
                              ec = 'w')
        
        AxCart.add_patch(InvisiblePointUp)
        # Apply scaling
        AxCart.axis('scaled')
        
        ## Lower Chart with Slider
        # Set y limits
        AxSlider.set(xlim = (-1.1*self.slider_max,self.slider_max*1.1))
        # Remove ticks on the y-axes
        AxSlider.yaxis.set_major_locator(NullLocator())
        # Apply scaling
        AxSlider.set_aspect("auto")
        
        return fig, AxCart, AxSlider
    
    
    # This method accepts the mouse position and updated the slider value accordingly
    # The mouse position has to be captured by a function not included in this class
    def update_slider(self, mouse_position):
        # The if statement formulates a saturation condition
        if mouse_position>self.slider_max:
            self.slider_value = self.slider_max
        elif mouse_position<-self.slider_max:
            self.slider_value = -self.slider_max
        else:
            self.slider_value = mouse_position
    
    
    # This method updates the elements of the Cart Figure which change at every frame.
    # Not that these elements are not ploted directly by this method
    # but rather returned as objects which can be used by another function
    # e.g. animation function from matplotlib package
    def update_drawing(self):
        
        #Draw mast
        mast_position = (self.CartPosition-(self.MastThickness/2.0))
        self.Mast.set_x(mast_position)
        #Draw rotated mast
        t21 = transforms.Affine2D().translate(-mast_position,-1.25*self.WheelRadius)
        t22 = transforms.Affine2D().rotate(self.angle) 
        t23 = transforms.Affine2D().translate(mast_position,1.25*self.WheelRadius)
        self.t2 = t21+t22+t23
        #Draw Chassis
        self.Chassis.set_x(self.CartPosition-(self.CartLength/2.0)) 
        #Draw Wheels
        self.WheelLeft.center = (self.CartPosition-self.WheelToMiddle,self.y_wheel)
        self.WheelRight.center = (self.CartPosition+self.WheelToMiddle,self.y_wheel)
        #Draw SLider
        self.Slider.set_width(self.slider_value)
        
        return self.Mast, self.t2, self. Chassis, self.WheelRight, self.WheelLeft, self.Slider
    
    
    # This method resets the dictionary keeping the history of simulation
    def reset_dict_history(self):
        self.dict_history = {'time':              [0.0],
                                'deltaTimeMs':    [0.0],
                                'position':       [self.CartPosition],
                                'positionD':      [self.CartPositionD],
                                'positionDD':     [self.CartPositionDD],
                                'angleErr':          [self.angle],
                                'angleD':         [self.angleD],
                                'angleDD':        [self.angleDD],
                                'motor':          [self.ueff],
                                'PositionTarget': [self.PositionTarget]}
        
        
    # This method saves the dictionary keeping the history of simulation to a .csv file
    def save_history_csv(self):
        
        # Make folder to save data (if not yet existing)
        try:
            os.makedirs('save')
        except:
            pass
        
        # Set path where to save the data
        logpath = './save/'+str(datetime.now().strftime('%Y-%m-%d_%H%M%S'))+'.csv'
        # Write the .csv file
        with open(logpath, "w") as outfile:
           writer = csv.writer(outfile)
           writer.writerow(self.dict_history.keys())
           writer.writerows(zip(*self.dict_history.values()))