# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:28:34 2020

@author: Marcin
"""

from math import fmod

from src.globals import *



def normalize_angle_rad(angle):
    Modulo = fmod(angle, 2*np.pi)  # positive modulo
    if Modulo < -np.pi:
        angle = Modulo+2*np.pi
    elif Modulo > np.pi:
        angle = Modulo-2*np.pi
    else:
        angle = Modulo
    return angle


def Generate_Experiment(MyCart, exp_len=64 + 640 + 1, dt=0.002):
    """
    This function runs a random CartPole experiment
    and returns the history of CartPole states, control inputs and desired cart position
    :param MyCart: instance of CartPole containing CartPole dynamics
    :param exp_len: How many time steps should the experiment have
                (default: 64+640+1 this format is used as it can )
    """

    # Set CartPole in the right (automatic control) mode
    MyCart.set_mode(1)  # 1 - you are controlling with LQR, 2- with do-mpc

    # Initialize Variables
    states = []
    u_effs = []
    target_positions = []

    # Generate new random function returning desired target position of the cart
    MyCart.dt = dt
    MyCart.random_length = exp_len
    MyCart.N = 10  # Complexity of generated target position track
    MyCart.Generate_Random_Trace_Function()

    # Randomly set the initial state

    MyCart.time_total = 0.0

    MyCart.s.position = np.random.uniform(low=-MyCart.HalfLength / 2.0,
                                          high=MyCart.HalfLength / 2.0)
    MyCart.s.positionD = np.random.uniform(low=-10.0,
                                           high=10.0)
    MyCart.s.angle = np.random.uniform(low=-17.5 * (np.pi / 180.0),
                                     high=17.5 * (np.pi / 180.0))
    MyCart.s.angleD = np.random.uniform(low=-15.5 * (np.pi / 180.0),
                                      high=15.5 * (np.pi / 180.0))

    MyCart.u = np.random.uniform(low=-0.9 * MyCart.p.u_max,
                                 high=0.9 * MyCart.p.u_max)

    # Target position at time 0 (should be always 0)
    MyCart.target_position = MyCart.random_track_f(MyCart.time_total)  # = 0
    # Constrain target position
    if MyCart.target_position > 0.8 * MyCart.HalfLength:
        MyCart.target_position = 0.8 * MyCart.HalfLength
    elif MyCart.target_position < -0.8 * MyCart.HalfLength:
        MyCart.target_position = -0.8 * MyCart.HalfLength

    # Run the CartPole experiment for number of time
    for i in range(int(exp_len)):

        # Start by incrementing total time
        MyCart.time_total = i * MyCart.dt
        # Print an error message if it runs already to long (should stop before)
        if MyCart.time_total > MyCart.t_max_pre:
            MyCart.time_total = MyCart.t_max_pre
            print('ERROR: It seems the experiment is running too long...')

        # Calculate acceleration, angular acceleration, velocity, angular velocity, position and angle
        # for this new time_step
        MyCart.Equations_of_motion()

        # Normalize angle

        MyCart.s.angle = normalize_angle_rad(MyCart.s.angle)

        if (abs(MyCart.s.position) + MyCart.WheelToMiddle) > MyCart.HalfLength:
            MyCart.s.positionD = -MyCart.s.positionD

        # Determine the dimensionales [-1,1] value of the motor power Q
        MyCart.Update_Q()

        # Calculate the force created by the motor
        MyCart.u = Q2u(MyCart.Q, MyCart.p)

        # Get the new value of desired cart position and constrain it
        MyCart.target_position = MyCart.random_track_f(MyCart.time_total)
        if MyCart.target_position > 0.8 * MyCart.HalfLength:
            MyCart.target_position = 0.8 * MyCart.HalfLength
        elif MyCart.target_position < -0.8 * MyCart.HalfLength:
            MyCart.target_position = -0.8 * MyCart.HalfLength

        # Save all the new cart state variables you just updated
        state = (MyCart.s.position,
                 MyCart.s.positionD,
                 MyCart.s.angle,
                 MyCart.s.angleD)

        # Save current cart state, control input and target position to a corresponding list
        states.append(state)
        u_effs.append(MyCart.u)
        target_positions.append(MyCart.target_position)

    # After generating experiment finished, return states, control input and target_positions history
    return states, u_effs, target_positions