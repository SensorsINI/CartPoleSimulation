from types import SimpleNamespace
from typing import Union
from CartPole.state_utilities import create_cartpole_state, cartpole_state_varname_to_index

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
CARTPOLE_EQUATIONS = 'Marcin-Sharpneat'
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

# Parameters of the CartPole
P_GLOBALS = SimpleNamespace()  # "p" like parameters
P_GLOBALS.m = 0.087  # mass of pole, kg # Checked by Antonio & Tobi
P_GLOBALS.M = 0.230  # mass of cart, kg # Checked by Antonio
P_GLOBALS.L = 0.395/2.0  # HALF (!!!) length of pend, m # Checked by Antonio & Tobi
P_GLOBALS.u_max = 6.21  # max force produced by the motor, N # Checked by Marcin
P_GLOBALS.M_fric = 6.34  # cart friction on track, N/m/s # Checked by Marcin
P_GLOBALS.J_fric = 2.5e-4  # friction coefficient on angular velocity in pole joint, Nm/rad/s # Checked by Marcin
P_GLOBALS.v_max = 0.8  # max DC motor speed, m/s, in absense of friction, used for motor back EMF model # TODO: not implemented in model, but needed for MPC

cart_length = 4.4e-2  # m, checked by Marcin&Asude
usable_track_length = 44.0e-2  # m, checked by Marcin&Asude
P_GLOBALS.TrackHalfLength = (usable_track_length-cart_length)/2.0  # m, effective length, by which cart center can move

P_GLOBALS.controlDisturbance = 0.0  # disturbance, as factor of u_max
P_GLOBALS.controlBias = 0.0  # bias of control input
P_GLOBALS.sensorNoise = 0.0  # sensor noise added to output of the system TODO: not implemented yet

P_GLOBALS.g = 9.81  # absolute value of gravity acceleration, m/s^2
P_GLOBALS.k = 4.0 / 3.0  # Dimensionless factor of moment of inertia of the pole with length 2L: I = k*m*L^2 # FIXME: I think it should be 1/3
# (I = k*m*L^2) (with L being half if the length)

# Export variables as global
k, M, m, g, J_fric, M_fric, L, v_max, u_max, sensorNoise, controlDisturbance, controlBias, TrackHalfLength = (
    P_GLOBALS.k,
    P_GLOBALS.M,
    P_GLOBALS.m,
    P_GLOBALS.g,
    P_GLOBALS.J_fric,
    P_GLOBALS.M_fric,
    P_GLOBALS.L,
    P_GLOBALS.v_max,
    P_GLOBALS.u_max,
    P_GLOBALS.sensorNoise,
    P_GLOBALS.controlDisturbance,
    P_GLOBALS.controlBias,
    P_GLOBALS.TrackHalfLength
)

# Create initial state vector
s0 = create_cartpole_state()


def _cartpole_ode(angle, angleD, position, positionD, u):
    """
    Calculates current values of second derivative of angle and position
    from current value of angle and position, and their first derivatives

    :param angle, angleD, position, positionD: Essential state information of cart
    :param u: Force applied on cart in unnormalized range

    :returns: angular acceleration, horizontal acceleration
    """
    ca = np.cos(angle)
    sa = np.sin(angle)

    if CARTPOLE_EQUATIONS == 'Marcin-Sharpneat':
        # Clockwise rotation is defined as negative
        # force and cart movement to the right are defined as positive
        # g (gravitational acceleration) is positive (absolute value)
        # Checked independently by Marcin and Krishna

        A = k * (M + m) - m * (ca ** 2)

        positionDD = (
            (
                + m * g * sa * ca  # Movement of the cart due to gravity
                + ((J_fric * angleD * ca) / (L))  # Movement of the cart due to pend' s friction in the joint
                - k * (m * L * (angleD ** 2) * sa)  # Keeps the Cart-Pole center of mass fixed when pole rotates
                - k * M_fric * positionD  # Braking of the cart due its friction
            ) / A
            + (k / A) * u  # Effect of force applied to cart
        )

        # Making m go to 0 and setting J_fric=0 (fine for pole without mass)
        # positionDD = (u_max/M)*Q-(M_fric/M)*positionD
        # Compare this with positionDD = a*Q-b*positionD
        # u_max = M*a = 0.230*19.6 = 4.5, 0.317*19.6 = 6.21, (Second option is if I account for pole mass)
        # M_fric = M*b = 0.230*20 = 4.6, 0.317*20 = 6.34
        # From experiment b = 20, a = 28

        angleDD = (
            (
                + g * (m + M) * sa  # Movement of the pole due to gravity
                - ((J_fric * (m + M) * angleD) / (L * m))  # Braking of the pole due friction in its joint
                - m * L * (angleD ** 2) * sa * ca  # Keeps the Cart-Pole center of mass fixed when pole rotates
                - ca * M_fric * positionD  # Friction of the cart on the track causing deceleration of cart and acceleration of pole in opposite direction due to intertia
            ) / (A * L) 
            + (ca / (A * L)) * u  # Effect of force applied to cart
        )

        # making M go to infinity makes angleDD = (g/k*L)sin(angle) - angleD*J_fric/(k*m*L^2)
        # This is the same as equation derived directly for a pendulum.
        # k is 4/3! It is the factor for pendulum with length 2L: I = k*m*L^2

    elif CARTPOLE_EQUATIONS == 'Marcin-Sharpneat-Recommended':
        # Distribute pole mass uniformly across pole 
        # Eq. (56) & (57)
        positionDD = (
            (
                m * g * (-sa) * ca 
                - 7/3 * (
                    u 
                    + m * L * (angleD ** 2) * (-sa)
                    - M_fric * positionD
                )
                - J_fric * (-angleD) * ca / L
            ) / (
                m * (ca ** 2)
                - 7/3 * (M + m)
            )
        )
        angleDD = - (
            3 / (7 * L) * (
                g * (-sa)
                - positionDD * ca
                - J_fric * (-angleD) / (m * L)
            )
        )
    else:
        raise ValueError('An undefined name for Cartpole equations')

    return angleDD, positionDD


def cartpole_ode_namespace(s: SimpleNamespace, u: float):
    return _cartpole_ode(
        s.angle, s.angleD, s.position, s.positionD, u
    )


def cartpole_ode(s: np.ndarray, u: float):
    return _cartpole_ode(
        s[..., cartpole_state_varname_to_index('angle')], s[..., cartpole_state_varname_to_index('angleD')],
        s[..., cartpole_state_varname_to_index('position')], s[..., cartpole_state_varname_to_index('positionD')],
        u
    )


def cartpole_jacobian(s: Union[np.ndarray, SimpleNamespace], u: float):
    """
    Jacobian of cartpole ode with the following structure:

        # ______________|    position     |   positionD    | angle | angleD |       u       |
        # position  (x) |   xx -> J[0,0]        xv            xt       xo      xu -> J[0,4]
        # positionD (v) |       vx              vv            vt       vo         vu
        # angle     (t) |       tx              tv            tt       to         tu
        # angleD    (o) |   ox -> J[3,0]        ov            ot       oo      ou -> J[3,4]
    
    :param p: Namespace containing environment variables such track length, cart mass and pole mass
    :param s: State vector following the globally defined variable order
    :param u: Force applied on cart in unnormalized range

    :returns: A 4x5 numpy.ndarray with all partial derivatives
    """
    if isinstance(s, np.ndarray):
        angle = s[cartpole_state_varname_to_index('angle')]
        angleD = s[cartpole_state_varname_to_index('angleD')]
        position = s[cartpole_state_varname_to_index('position')]
        positionD = s[cartpole_state_varname_to_index('positionD')]
    elif isinstance(s, SimpleNamespace):
        angle = s.angle
        angleD = s.angleD
        position = s.position
        positionD = s.positionD
    
    J = np.zeros(shape=(4, 5), dtype=np.float32)  # Array to keep Jacobian
    ca = np.cos(angle)
    sa = np.sin(angle)

    if CARTPOLE_EQUATIONS == 'Marcin-Sharpneat':
        # Helper function
        A = k * (M + m) - m * (ca ** 2)

        # Jacobian entries
        J[0, 0] = 0.0  # xx

        J[0, 1] = 1.0  # xv

        J[0, 2] = 0.0  # xt

        J[0, 3] = 0.0  # xo

        J[0, 4] = 0.0  # xu

        J[1, 0] = 0.0  # vx

        J[1, 1] = -k * M_fric / A  # vv

        J[1, 2] = (    # vt
                     -2.0 * k * u * ca * sa * m
                     - 2.0 * ca * sa * m * (
                             -k * L * (angleD**2) * sa * m
                             + g * ca * sa * m - k * positionD * M_fric
                             + (angleD * ca * J_fric/L)
                                                                ))/(A**2) \
             + (
                     -k * L * (angleD**2) * ca * m
                     +  g * ((ca**2)-(sa**2)) * m
                     - (angleD * sa * J_fric)/L
                                                                )/ A

        J[1, 3] = (-2.0 * k * L * angleD * sa * m  # vo
              + (ca * J_fric / L)) / A

        J[1, 4] = k / A  # vu

        J[2, 0] = 0.0  # tx

        J[2, 1] = 0.0  # tv

        J[2, 2] = 0.0  # tt

        J[2, 3] = 1.0  # to

        J[2, 4] = 0.0  # tu

        J[3, 0] = 0.0  # ox

        J[3, 1] = -ca * M_fric / (L * A)  # ov

        J[3, 2] = (  # ot
                    - 2.0 * u * (ca**2) * sa * m
                    - 2.0 * ca * sa * m * (
                            -L * (angleD**2) * ca * sa * m
                            + g * sa * (M + m)
                            - positionD * ca * M_fric
                            - (angleD * (M+m) * J_fric)/(L*m))
                                    )/(L*(A**2)) \
             + (
                     -u * sa
                     + L * (angleD**2) * ((sa**2)-(ca**2)) * m
                     + g*ca*(M+m)
                     + positionD * sa * M_fric
                                    )/(L*A)

        J[3, 3] = (  # oo
                     -2.0*L*angleD*ca * sa * m
                     - ((M+m) * J_fric)/(L*m)
                                    ) / (L * A)

        J[3, 4] = ca / (L*A)  # ou

        return J



def Q2u(Q):
    """
    Converts dimensionless motor power [-1,1] to a physical force acting on a cart.

    In future there might be implemented here a more sophisticated model of a motor driving CartPole
    """
    u = u_max * (
        Q + controlDisturbance *  rng.standard_normal(size=np.shape(Q), dtype=np.float32) + P_GLOBALS.controlBias
    )  # Q is drive -1:1 range, add noise on control

    return u


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
    # Calculate time necessary for evaluation of a Jacobian:

    f_to_measure = 'Jacobian = cartpole_jacobian(s, u)'
    number = 1  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 100000 # Gives how many times timeit should be repeated
    timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
    min_time = min(timings)/float(number)
    max_time = max(timings)/float(number)
    average_time = np.mean(timings)/float(number)
    print('Min time to calculate Jacobian is {} us'.format(min_time * 1.0e6))  # ca. 14 us
    print('Average time to calculate Jacobian is {} us'.format(average_time*1.0e6))  # ca 16 us
    print('Max time to calculate Jacobian is {} us'.format(max_time * 1.0e6))          # ca. 150 us

    # Calculate once more to prrint the resulting matrix
    Jacobian = np.around(cartpole_jacobian(s, u), decimals=6)

    print()
    print(Jacobian.dtype)
    print(Jacobian)