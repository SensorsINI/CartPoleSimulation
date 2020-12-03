"""Parameters of CartPole system and simulation, Ground Truth equations of CartPole"""

import numpy as np
from types import SimpleNamespace

# You can choose CartPole dynamical equations you want to use in simulation by setting CARTPOLE_EQUATIONS variable
# The possible choices and their explanation are listed below
# Notice that any set of equation require setting the convention for the angle
# to draw a CartPole correctly in the CartPole GUI
CARTPOLE_EQUATIONS = 'Marcin-Sharpneat'
""" 
Possible choices: 'Marcin-Sharpneat', 'Krishna'

'Marcin-Sharpneat' is done on the basis of:
https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
Required angle convention: CLOCK-NEG

Krishna is done on the basis of:
Required angle convention:
"""

ANGLE_CONVENTION = 'CLOCK-NEG'
"""Defines if a clockwise angle change is negative ('CLOCK-NEG') or positive ('CLOCK-POS')

The 0-angle state is always defined as pole in upright position. This currently cannot be changed
"""

# Variables settings parameters CartPole GUI starts with
# This is useful if you need to many times restart the GUI to some particular settings
# e.g. while testing new controller
mode_globals = 0  # Defines which controller is loaded
"""Sets the controller of the CartPole

Possible choices
0: manual stabilization
1: linear-quadratic regulator (LQR)
2: MPC based on Ground Truth CartPole ODE (continuous model) implemented with do-mpc library
3: MPC based on Ground Truth CartPole ODE wrapped (integrated) to predict the next state of the system (discrete model)
    implemented in do-mpc library
4: RNN trained to mimic the MPC controller, RNN implemented in Pytorch
5: like as 4, just different RNN implemented in TensorFlow
6: like 3, just CartPole dynamics model provided not with true equation,
        but rather with RNN (Pytorch) trained to predict future state of the CartPole
7: like  6, just different RNN implemented in TensorFlow

"""
save_history_globals = True  # Save experiment history as CSV
stop_at_90_globals = False  # Block the pole if it reaches +/-90 deg (horizontal position)
load_recording_globals = False  # Start/Stop button: True: load and replay a recording; False: start new experiment
slider_on_click_globals = True  # True: update slider only on click, False: update slider while hoovering over it
speedup_globals = 1.0  # Multiplicative factor by which the simulation seen by the user differs from real time
# E.g. 2.0 means that you watch simulation double speed
# WARNING: This is the target value, max speedup is limited by speed of performing CartPole simulation
# True instantaneous speedup is displayed in CartPole GUI as "Speed-up(measured)"
# BUG: It seems that if speedup is set the way it cannot be reached,
#   it may lead to unstable MPC instead of just making simulation run on the fastest achievable speed

# Variables used for physical simulation
dt_main_simulation_globals = 0.020  # Time step of CartPole simulation

# MPC
dt_mpc_simulation_globals = 0.2  # Time step used by MPC controller
# WARNING: if using RNN to provide CartPole model to MPC
# make sure that it is trained to predict future states with this timestep
# TODO: Add dt information to .txt file associated with and describing each RNN
mpc_horizon_globals = 20 # Number of steps into future MPC controller simulates at each evaluation

# Parameters of the CartPole
m_globals = 2.0  # mass of pend, kg
M_globals = 1.0  # mass of cart, kg
L_globals = 1.0  # HALF (!!!) length of pend, m
u_max_globals = 200.0  # max cart force, N
M_fric_globals = 0.0  # cart friction, N/m/s
J_fric_globals = 0.0  # friction coefficient on angular velocity, Nm/rad/s
v_max_globals = 10.0  # max DC motor speed, m/s, in absense of friction, used for motor back EMF model
controlDisturbance_globals = 0.0  # disturbance, as factor of u_max
sensorNoise_globals = 0.0  # noise, as factor of max values

g_globals = 9.81  # absolute value of gravity acceleration, m/s^2
k_globals = 4.0 / 3.0  # Dimensionless factor of moment of inertia of the pole
# (I = k*m*L^2) (with L being half if the length)

# Variables for random trace generation
# Complexity of the random trace, number of random points used for interpolation
track_relative_complexity_globals = 0.3
random_length_globals = 1e1  # Number of seconds in the random length trace
interpolation_type_globals = 'previous'  # Sets how to interpolate between turning points of random trace
# Possible choices: '0-derivative-smooth', 'linear', 'previous'
turning_points_period_globals = 'regular'  # How turning points should be distributed
# Possible choices: 'regular', 'random'


def cartpole_integration(s, dt):
    """Simple single step integration of CartPole state by dt

    Takes state as SimpleNamespace, but returns as separate variables
    # TODO: Consider changing it to return a SimpleNamepece for consistency

    :param s: state of the CartPole (contains: s.position, s.positionD, s.angle and s.angleD)
    :param dt: time step by which the CartPole state should be integrated
    """
    s_next = SimpleNamespace()

    s_next.position = s.position + s.positionD * dt
    s_next.positionD = s.positionD + s.positionDD * dt

    s_next.angle = s.angle + s.angleD * dt
    s_next.angleD = s.angleD + s.angleDD * dt

    return s_next


def cartpole_ode(p, s, u):
    """Calculates current values of second derivative of angle and position
    from current value of angle and position, and their first derivatives


    """
    ca = np.cos(s.angle)
    sa = np.sin(s.angle)

    if CARTPOLE_EQUATIONS == 'Krishna':

        A = (p.M + p.m) - p.m * (ca ** 2)
        angleDD = ((p.m + p.M) * p.g * sa + (u * ca) - (p.m * p.L * (s.angleD ** 2) * sa * ca)) / (A * p.L)
        positionDD = (u - (p.m * p.g * sa * ca) + (p.m * p.L * (s.angleD ** 2) * sa)) / A

    elif CARTPOLE_EQUATIONS == 'Marcin-Sharpneat':

        A = (p.k + 1) * (p.M + p.m) - p.m * (ca ** 2)

        angleDD = (p.g * (p.m + p.M) * sa -
                   ((p.J_fric * (p.m + p.M) * s.angleD) / (p.L * p.m)) -  # Friction of the pole in its joint
                   p.m * p.L * (s.angleD ** 2) * sa * ca +
                   ca * p.M_fric * s.positionD +  # Friction of the cart on the track
                   ca * u) / (A * p.L)

        positionDD = (
                             p.m * p.g * sa * ca +
                             ((p.J_fric * s.angleD * ca) / (p.L)) -
                             (p.k + 1) * (p.m * p.L * (s.angleD ** 2) * sa + p.M_fric * s.positionD) +
                             (p.k + 1) * u
                     ) / A
    else:
        raise ValueError('An undefined name for Cartpole equations')

    return angleDD, positionDD


def Q2u(Q, p):
    """Converts dimensionless motor power [-1,1] to a physical force acting on a cart.

    In future there might be implemented here a more sophisticated model of a motor driving CartPole
    """
    u = p.u_max * Q + p.controlDisturbance * np.random.normal() * p.u_max  # Q is drive -1:1 range, add noise on control
    return u


def mpc_next_state(s, p, u, dt):
    """Wrapper for CartPole ODE. Given a current state (without second derivatives), returns a state after time dt

    TODO: This might be combined with cartpole_integration,
        although the order of cartpole_ode and cartpole_integration is different than in CartClass
    """

    angleDD, positionDD = cartpole_ode(p, s, u)  # Calculates CURRENT second derivatives

    # Calculate NEXT state:
    position_next = s.position + s.positionD * dt
    positionD_next = s.positionD + positionDD * dt

    angle_next = s.angle + s.angleD * dt
    angleD_next = s.angleD + angleDD * dt

    return position_next, positionD_next, angle_next, angleD_next
