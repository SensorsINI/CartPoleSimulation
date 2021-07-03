from types import SimpleNamespace
from typing import Union
from CartPole.state_utilities import (
    create_cartpole_state, cartpole_state_varname_to_index,
    ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX
)
from others.p_globals import (
    k, M, m, g, J_fric, M_fric, L, v_max, u_max,
    sensorNoise, controlDisturbance, controlBias, TrackHalfLength,
    CARTPOLE_EQUATIONS
)

from numba import float32, jit
import numpy as np
from numpy.random import SFC64, Generator
rng = Generator(SFC64(123))

# -> PLEASE UPDATE THE cartpole_model.nb (Mathematica file) IF YOU DO ANY CHANAGES HERE (EXCEPT \
# FOR PARAMETERS VALUES), SO THAT THESE TWO FILES COINCIDE. AND LET EVERYBODY \
# INVOLVED IN THE PROJECT KNOW WHAT CHANGES YOU DID.

"""This script contains equations and parameters used currently in CartPole simulator."""

# You can choose CartPole dynamical equations you want to use in simulation by setting CARTPOLE_EQUATIONS variable
# The possible choices and their explanation are listed below
# Notice that any set of equation require setting the convention for the angle
# to draw a CartPole correctly in the CartPole GUI

""" 
Possible choices: 'Marcin-Sharpneat', (currently no more choices available)
'Marcin-Sharpneat' is derived by Marcin, checked by Krishna, coincide with:
https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
(The friction terms not compared attentively but calculated and checked carefully,
the rest should be the same up to the angle-direction-convention and notation changes.)

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


def _cartpole_ode(ca, sa, angleD, positionD, u):
    """
    Calculates current values of second derivative of angle and position
    from current value of angle and position, and their first derivatives

    :param angle, angleD, position, positionD: Essential state information of cart
    :param u: Force applied on cart in unnormalized range

    :returns: angular acceleration, horizontal acceleration
    """
    if CARTPOLE_EQUATIONS == 'Marcin-Sharpneat':
        # Clockwise rotation is defined as negative
        # force and cart movement to the right are defined as positive
        # g (gravitational acceleration) is positive (absolute value)
        # Checked independently by Marcin and Krishna

        A = m * (ca ** 2) - (k + 1) * (M + m)

        positionDD = (
            (
                + m * g * sa * ca  # Movement of the cart due to gravity
                - ((J_fric * (-angleD) * ca) / L)  # Movement of the cart due to pend' s friction in the joint
                - (k + 1) * (
                    + (m * L * (angleD ** 2) * sa)  # Keeps the Cart-Pole center of mass fixed when pole rotates
                    - M_fric * positionD  # Braking of the cart due its friction
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
                g * sa - positionDD * ca - (J_fric * (-angleD)) / (m * L) 
            ) / ((k + 1) * L)
        ) * (-1.0)

        # making M go to infinity makes angleDD = (g/k*L)sin(angle) - angleD*J_fric/(k*m*L^2)
        # This is the same as equation derived directly for a pendulum.
        # k is 4/3! It is the factor for pendulum with length 2L: I = k*m*L^2
    else:
        raise ValueError('An undefined name for Cartpole equations')

    return angleDD, positionDD, ca, -sa


_cartpole_ode_numba = jit(_cartpole_ode, nopython=True, cache=True, fastmath=True)


def cartpole_ode_namespace(s: SimpleNamespace, u: float):
    angleDD, positionDD, _, _ = _cartpole_ode(
        np.cos(-s.angle), np.sin(-s.angle), s.angleD, s.positionD, u
    )
    return angleDD, positionDD


def cartpole_ode(s: np.ndarray, u: float):
    angle = s[..., ANGLE_IDX]
    angleDD, positionDD, _, _ = _cartpole_ode_numba(
        np.cos(-angle), np.sin(-angle), s[..., ANGLED_IDX], s[..., POSITIOND_IDX], u
    )
    return angleDD, positionDD


@jit(nopython=True, cache=True, fastmath=True)
def edge_bounce(angle, angleD, position, positionD, t_step):
    if abs(position) >= TrackHalfLength:
        angleD -= 2 * (positionD * np.cos(angle)) / L
        angle += angleD * t_step
        positionD = -positionD
        position += positionD * t_step
    return angle, angleD, position, positionD


@jit(nopython=True, cache=True, fastmath=True)
def edge_bounce_wrapper(angle, angleD, position, positionD, t_step):
    for i in range(position.size):
        angle[i], angleD[i], position[i], positionD[i] = edge_bounce(angle[i], angleD[i], position[i], positionD[i], t_step)
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


@jit(nopython=True, cache=True, fastmath=True)
def euler_step(state, stateD, t_step):
    state += stateD * t_step
    return state


@jit(nopython=True, cache=True, fastmath=True)
def next_state_numba(angle, angleD, angleDD, angle_cos, angle_sin, position, positionD, positionDD, u, t_step, intermediate_steps):
    for _ in range(intermediate_steps):
        angle = euler_step(angle, angleD, t_step)
        angleD = euler_step(angleD, angleDD, t_step)
        position = euler_step(position, positionD, t_step)
        positionD = euler_step(positionD, positionDD, t_step)

        angle, angleD, position, positionD = edge_bounce_wrapper(angle, angleD, position, positionD, t_step)

        angleDD, positionDD, angle_cos, angle_sin = _cartpole_ode_numba(np.cos(-angle), np.sin(-angle), angleD, positionD, u)

    return angle, angleD, angleDD, position, positionD, positionDD, angle_cos, angle_sin


if __name__ == '__main__':
    import timeit
    """
    On 9.02.2021 we saw a perfect coincidence (5 digits after coma) of Jacobian from Mathematica cartpole_model.nb
    with Jacobian calculated with this script for all non zero inputs, dtype=float32
    """

    # Set non-zero input
    s = s0
    s[cartpole_state_varname_to_index('position')] = -30.2
    s[cartpole_state_varname_to_index('positionD')] = 2.87
    s[cartpole_state_varname_to_index('angle')] = -0.32
    s[cartpole_state_varname_to_index('angleD')] = 0.237
    u = -0.24


    # Calculate time necessary to evaluate cartpole ODE:

    f_to_measure = 'angleDD, positionDD = cartpole_ode(s, u)'
    number = 1  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 100000 # Gives how many times timeit should be repeated
    timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
    min_time = min(timings)/float(number)
    max_time = max(timings)/float(number)
    average_time = np.mean(timings)/float(number)
    print()
    print('----------------------------------------------------------------------------------')
    print('Min time to evaluate ODE is {} us'.format(min_time * 1.0e6))  # ca. 5 us
    print('Average time to evaluate ODE is {} us'.format(average_time*1.0e6))  # ca 5 us
    # The max is of little relevance as it is heavily influenced by other processes running on the computer at the same time
    print('Max time to evaluate ODE is {} us'.format(max_time * 1.0e6))          # ca. 100 us
    print('----------------------------------------------------------------------------------')
    print()