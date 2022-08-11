from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from others.globals_and_utils import create_rng, load_config
from others.p_globals import (J_fric, L, M, M_fric, TrackHalfLength,
                              controlBias, controlDisturbance, g, k, m, u_max,
                              v_max)
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                      ANGLED_IDX, POSITION_IDX, POSITIOND_IDX,
                                      create_cartpole_state)

config = load_config("config.yml")

k = tf.convert_to_tensor(k)
M = tf.convert_to_tensor(M)
m = tf.convert_to_tensor(m)
g = tf.convert_to_tensor(g)
J_fric = tf.convert_to_tensor(J_fric)
M_fric = tf.convert_to_tensor(M_fric)
L = tf.convert_to_tensor(L)
v_max = tf.convert_to_tensor(v_max)
u_max = tf.convert_to_tensor(u_max)
controlDisturbance = tf.convert_to_tensor(controlDisturbance)
controlBias = tf.convert_to_tensor(controlBias)
TrackHalfLength = tf.convert_to_tensor(TrackHalfLength)


rng = create_rng(__name__, config["cartpole"]["seed"])


# -> PLEASE UPDATE THE cartpole_model.nb (Mathematica file) IF YOU DO ANY CHANAGES HERE (EXCEPT \
# FOR PARAMETERS VALUES), SO THAT THESE TWO FILES COINCIDE. AND LET EVERYBODY \
# INVOLVED IN THE PROJECT KNOW WHAT CHANGES YOU DID.

"""This script contains equations and parameters used currently in CartPole simulator."""

# Notice that any set of equation require setting the convention for the angle
# to draw a CartPole correctly in the CartPole GUI

""" 
derived by Marcin, checked by Krishna, coincide with:
https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html

Should be the same up to the angle-direction-convention and notation changes.

The convention:
Pole upright position defines 0 angle
Cart movement to the right is positive
Clockwise angle rotation is defined as negative

Required angle convention for CartPole GUI: CLOCK-NEG
"""

ANGLE_CONVENTION = 'CLOCK-NEG'
"""Defines if a clockwise angle change is negative ('CLOCK-NEG') or positive ('CLOCK-POS')

The 0-angle state is always defined as pole in upright position. This currently cannot be changed
"""

# Create initial state vector
s0 = create_cartpole_state()


def _cartpole_ode(ca, sa, angleD, positionD, u,
                      k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric, L=L):

    """
    Calculates current values of second derivative of angle and position
    from current value of angle and position, and their first derivatives

    :param angle, angleD, position, positionD: Essential state information of cart
    :param u: Force applied on cart in unnormalized range

    :returns: angular acceleration, horizontal acceleration
    """

    # Clockwise rotation is defined as negative
    # force and cart movement to the right are defined as positive
    # g (gravitational acceleration) is positive (absolute value)
    # Checked independently by Marcin and Krishna

    A = (k + 1.0) * (M + m) - m * (ca ** 2)
    F_fric = - M_fric * positionD  # Force resulting from cart friction, notice that the mass of the cart is not explicitly there
    T_fric = - J_fric * angleD  # Torque resulting from pole friction

    positionDD = (
            (
                    m * g * sa * ca  # Movement of the cart due to gravity
                    + ((T_fric * ca) / L)  # Movement of the cart due to pend' s friction in the joint
                    + (k + 1) * (
                            - (m * L * (
                                        angleD ** 2) * sa)  # Keeps the Cart-Pole center of mass fixed when pole rotates
                            + F_fric  # Braking of the cart due its friction
                            + u  # Effect of force applied to cart
                    )
            ) / A
    )

    # Making m go to 0 and setting J_fric=0 (fine for pole without mass)
    # positionDD = (u_max/M)*Q-(M_fric/M)*positionD
    # Compare this with positionDD = a*Q-b*positionD
    # u_max = M*a = 0.230*19.6 = 4.5, 0.317*19.6 = 6.21, (Second option is if I account for pole mass)
    # M_fric = M*b = 0.230*20 = 4.6, 0.317*20 = 6.34
    # From experiment b = 20, a = 28
    angleDD = (
            (
                    g * sa + positionDD * ca + T_fric / (m * L)
            ) / ((k + 1) * L)
    )

    # making M go to infinity makes angleDD = (g/k*L)sin(angle) - angleD*J_fric/(k*m*L^2)
    # This is the same as equation derived directly for a pendulum.
    # k is 4/3! It is the factor for pendulum with length 2L: I = k*m*L^2

    return angleDD, positionDD


def cartpole_ode_namespace(s: SimpleNamespace, u: float,
                           k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric, L=L):
    angleDD, positionDD = _cartpole_ode(
        np.cos(s.angle), np.sin(s.angle), s.angleD, s.positionD, u,
        k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric, L=L
    )
    return angleDD, positionDD


def cartpole_ode(s: np.ndarray, u: float,
                 k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric, L=L):
    angleDD, positionDD = _cartpole_ode(
        s[..., ANGLE_COS_IDX], s[..., ANGLE_SIN_IDX], s[..., ANGLED_IDX], s[..., POSITIOND_IDX], u,
        k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric, L=L
    )
    return angleDD, positionDD

def edge_bounce(angle, angle_cos, angleD, position, positionD, t_step, L=L):
    if position >= TrackHalfLength or -position >= TrackHalfLength:  # Without abs to compile with tensorflow
        angleD -= 2 * (positionD * angle_cos) / L
        angle += angleD * t_step
        positionD = -positionD
        position += positionD * t_step
    return angle, angleD, position, positionD


def edge_bounce_wrapper(angle, angle_cos, angleD, position, positionD, t_step, L=L):
    for i in range(position.size):
        angle[i], angleD[i], position[i], positionD[i] = edge_bounce(angle[i], angle_cos[i], angleD[i], position[i], positionD[i],
                                                                     t_step, L)
    return angle, angleD, position, positionD


def Q2u(Q):
    """
    Converts dimensionless motor power [-1,1] to a physical force acting on a cart.

    In future there might be implemented here a more sophisticated model of a motor driving CartPole
    """
    u = u_max * (
        Q + controlDisturbance * rng.standard_normal(size=np.shape(Q), dtype=np.float32) + controlBias
    )  # Q is drive -1:1 range, add noise on control

    return u


def euler_step(state, stateD, t_step):
    return state + stateD * t_step


def cartpole_integration(angle, angleD, angleDD, position, positionD, positionDD, t_step):
    angle_next = euler_step(angle, angleD, t_step)
    angleD_next = euler_step(angleD, angleDD, t_step)
    position_next = euler_step(position, positionD, t_step)
    positionD_next = euler_step(positionD, positionDD, t_step)

    return angle_next, angleD_next, position_next, positionD_next


@Compile
def euler_step_tf(state, stateD, t_step):
    return state + stateD * t_step


@Compile
def cartpole_integration_tf(angle, angleD, angleDD, position, positionD, positionDD, t_step):
    angle_next = euler_step_tf(angle, angleD, t_step)
    angleD_next = euler_step_tf(angleD, angleDD, t_step)
    position_next = euler_step_tf(position, positionD, t_step)
    positionD_next = euler_step_tf(positionD, positionDD, t_step)

    return angle_next, angleD_next, position_next, positionD_next


if __name__ == '__main__':
    import timeit

    """
    On 9.02.2021 we saw a perfect coincidence (5 digits after coma) of Jacobian from Mathematica cartpole_model.nb
    with Jacobian calculated with this script for all non zero inputs, dtype=float32
    """

    # Set non-zero input
    s = s0
    s[POSITION_IDX] = -30.2
    s[POSITIOND_IDX] = 2.87
    s[ANGLE_IDX] = -0.32
    s[ANGLED_IDX] = 0.237
    u = -0.24


    # Calculate time necessary to evaluate cartpole ODE:

    f_to_measure = 'angleDD, positionDD = cartpole_ode(s, u)'
    number = 1  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 100000  # Gives how many times timeit should be repeated
    timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
    min_time = min(timings) / float(number)
    max_time = max(timings) / float(number)
    average_time = np.mean(timings) / float(number)
    print()
    print('----------------------------------------------------------------------------------')
    print('Min time to evaluate ODE is {} us'.format(min_time * 1.0e6))  # ca. 5 us
    print('Average time to evaluate ODE is {} us'.format(average_time * 1.0e6))  # ca 5 us
    # The max is of little relevance as it is heavily influenced by other processes running on the computer at the same time
    print('Max time to evaluate ODE is {} us'.format(max_time * 1.0e6))  # ca. 100 us
    print('----------------------------------------------------------------------------------')
    print()
