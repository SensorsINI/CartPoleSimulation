import numpy as np
from types import SimpleNamespace

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
'Marcin-Sharpneat' is derived by Marcin, checked by Krishna, coinside with:
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
p_globals = SimpleNamespace()  # "p" like parameters
p_globals.m = 2.0  # mass of pend, kg
p_globals.M = 1.0  # mass of cart, kg
p_globals.L = 1.0  # HALF (!!!) length of pend, m
p_globals.u_max = 200.0  # max force produced by the motor, N
p_globals.M_fric = 1.0  # cart friction on track, N/m/s
p_globals.J_fric = 2.0  # friction coefficient on angular velocity in pole joint, Nm/rad/s
p_globals.v_max = 10.0  # max DC motor speed, m/s, in absense of friction, used for motor back EMF model # TODO: not implemented yet

p_globals.TrackHalfLength = 50.0  # m, length of the track on which CartPole can move, from 0 to edge, track is symmetric

p_globals.controlDisturbance = 0.0  # disturbance, as factor of u_max
p_globals.sensorNoise = 0.0  # sensor noise added to output of the system TODO: not implemented yet

p_globals.g = 9.81  # absolute value of gravity acceleration, m/s^2
p_globals.k = 4.0 / 3.0  # Dimensionless factor of moment of inertia of the pole
# (I = k*m*L^2) (with L being half if the length)

# Container for Cartpole state (augmented with second derivatives) filled with 0.0
s0 = SimpleNamespace()  # "s" like state
s0.position = 0.0  # position
s0.positionD = 0.0  # velocity
s0.positionDD = 0.0  # acceleration
s0.angle = 0.0  # angle
s0.angleD = 0.0  # angular speed
s0.angleDD = 0.0  # angular acceleration


def cartpole_ode(p, s, u):
    """Calculates current values of second derivative of angle and position
    from current value of angle and position, and their first derivatives


    """
    ca = np.cos(s.angle)
    sa = np.sin(s.angle)

    if CARTPOLE_EQUATIONS == 'Marcin-Sharpneat':
        # Clockwise rotation is defined as negative
        # force and cart movement to the right are defined as positive
        # g (gravitational acceleration) is positive (absolute value)
        # Checked independently by Marcin and Krishna

        A = (p.k + 1) * (p.M + p.m) - p.m * (ca ** 2)

        positionDD = (
                             + p.m * p.g * sa * ca  # Movement of the cart due to gravity
                             + ((p.J_fric * s.angleD * ca) / (p.L))  # Movement of the cart due to pend' s friction in the joint
                             - (p.k + 1) * (p.m * p.L * (s.angleD ** 2) * sa)  # Keeps the Cart-Pole center of mass fixed when pole rotates
                             - (p.k + 1) * p.M_fric * s.positionD  # Braking of the cart due its friction
                     ) / A \
                                + ((p.k + 1) / A) * u  # Effect of force applied to cart

        angleDD = (
                          + p.g * (p.m + p.M) * sa  # Movement of the pole due to gravity
                          - ((p.J_fric * (p.m + p.M) * s.angleD) / (p.L * p.m))  # Braking of the pole due friction in its joint
                          - p.m * p.L * (s.angleD ** 2) * sa * ca  # Keeps the Cart-Pole center of mass fixed when pole rotates
                          - ca * p.M_fric * s.positionD  # Friction of the cart on the track causing deceleration of cart and acceleration of pole in opposite direction due to intertia
                          ) / (A * p.L) \
                                + (ca / (A * p.L)) * u  # Effect of force applied to cart

    else:
        raise ValueError('An undefined name for Cartpole equations')

    return angleDD, positionDD


def cartpole_jacobian(p, s, u):
    """Jacobian of cartpole ode"""
    J = np.zeros(shape=(4, 5), dtype=np.float32)  # Array to keep Jacobian
    ca = np.cos(s.angle)
    sa = np.sin(s.angle)

    if CARTPOLE_EQUATIONS == 'Marcin-Sharpneat':

        #
        # ______________|    position     |   positionD    | angle | angleD |       u       |
        # position  (x) |   xx -> J[0,0]        xv            xt       xo      xu -> J[0,4]
        # positionD (v) |       vx              vv            vt       vo         vu
        # angle     (t) |       tx              tv            tt       to         tu
        # angleD    (o) |   ox -> J[3,0]        ov            ot       oo      ou -> J[3,4]

        # Helper function
        A = (p.k + 1.0) * (p.M + p.m) - p.m * (ca ** 2)

        # Jacobian entries
        J[0, 0] = 0.0  # xx

        J[0, 1] = 1.0  # xv

        J[0, 2] = 0.0  # xt

        J[0, 3] = 0.0  # xo

        J[0, 4] = 0.0  # xu

        J[1, 0] = 0.0  # vx

        J[1, 1] = -(1.0+p.k) * p.M_fric / A  # vv

        J[1, 2] = (    # vt
                     -2.0 * (1.0+p.k) * u * ca * sa * p.m
                     - 2.0 * ca * sa * p.m * (
                             -(1.0+p.k) * p.L * (s.angleD**2) * sa * p.m
                             + p.g * ca * sa * p.m - (1.0+p.k) * s.positionD * p.M_fric
                             + (s.angleD * ca * p.J_fric/p.L)
                                                                ))/(A**2) \
             + (
                     -(1.0+p.k) * p.L * (s.angleD**2) * ca * p.m
                     +  p.g * ((ca**2)-(sa**2)) * p.m
                     - (s.angleD * sa * p.J_fric)/p.L
                                                                )/ A

        J[1, 3] = (-2.0 * (1.0+p.k) * p.L * s.angleD * sa * p.m  # vo
              + (ca * p.J_fric / p.L)) / A

        J[1, 4] = (1.0+p.k) / A  # vu

        J[2, 0] = 0.0  # tx

        J[2, 1] = 0.0  # tv

        J[2, 2] = 0.0  # tt

        J[2, 3] = 1.0  # to

        J[2, 4] = 0.0  # tu

        J[3, 0] = 0.0  # ox

        J[3, 1] = -ca * p.M_fric / (p.L * A)  # ov

        J[3, 2] = (  # ot
                    - 2.0 * u * (ca**2) * sa * p.m
                    - 2.0 * ca * sa * p.m * (
                            -p.L * (s.angleD**2) * ca * sa * p.m
                            + p.g * sa * (p.M + p.m)
                            - s.positionD * ca * p.M_fric
                            - (s.angleD * (p.M+p.m) * p.J_fric)/(p.L*p.m))
                                    )/(p.L*(A**2)) \
             + (
                     -u * sa
                     + p.L * (s.angleD**2) * ((sa**2)-(ca**2)) * p.m
                     + p.g*ca*(p.M+p.m)
                     + s.positionD * sa * p.M_fric
                                    )/(p.L*A)

        J[3, 3] = (  # oo
                     -2.0*p.L*s.angleD*ca * sa * p.m
                     - ((p.M+p.m) * p.J_fric)/(p.L*p.m)
                                    ) / (p.L * A)

        J[3, 4] = ca / (p.L*A)  # ou

        return J



def Q2u(Q, p):
    """Converts dimensionless motor power [-1,1] to a physical force acting on a cart.

    In future there might be implemented here a more sophisticated model of a motor driving CartPole
    """
    u = p.u_max * Q + p.controlDisturbance * np.random.normal() * p.u_max  # Q is drive -1:1 range, add noise on control

    return u


if __name__ == '__main__':
    import timeit
    """
    On 9.02.2021 we saw a perfect coincidence (5 digits after coma) of Jacobian from Mathematica cartpole_model.nb
    with Jacobian calculated with this script for all non zero inputs, dtype=float32
    """

    # Set non-zero input
    s = s0
    s.position = -30.2
    s.positionD = 2.87
    s.angle = -0.32
    s.angleD = 0.237
    u = -0.24


    # Calculate time necessary to evaluate cartpole ODE:

    f_to_measure = 'angleDD, positionDD = cartpole_ode(p_globals, s, u)'
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

    f_to_measure = 'Jacobian = cartpole_jacobian(p_globals, s, u)'
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
    Jacobian = np.around(cartpole_jacobian(p_globals, s, u), decimals=6)

    print()
    print(Jacobian.dtype)
    print(Jacobian)