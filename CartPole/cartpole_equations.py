from SI_Toolkit.computation_library import NumpyLibrary, PyTorchLibrary, TensorFlowLibrary

from SI_Toolkit.Functions.TF.Compile import CompileAdaptive

from CartPole.cartpole_parameters import CartPoleParameters, TrackHalfLength
from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                      ANGLED_IDX, POSITION_IDX, POSITIOND_IDX)

from scipy.integrate import solve_ivp
import numpy as np


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



def _cartpole_ode(ca, sa, angleD, positionD, u,
                  k, m_cart, m_pole, g, J_fric, M_fric, L):

    """
    Calculates current values of second derivative of angle and position
    from current value of angle and position, and their first derivatives

    :param angle, angleD, position, positionD: Essential state information of cart.
        Angle is in radians, 0 vertical and increasing CCW.
        position is in meters, 0 in middle of track and increasing to right.
    :param m_cart and m_pole: masses in kg of cart and pole.
    :param ca and sa: sin and cos of angle of pole.
    :param g: gravity in m/s^2
    :param J_fric and M_fric: friction coefficients in Nm per rad/s of pole  TODO check correct
    :param  M_fric: friction coefficient of cart in N per m/s TODO check correct
    :param L: length of pole in meters.

    :param u: Force applied on cart in unnormalized range TODO what does this mean?

    :returns: angular acceleration, horizontal acceleration
    """

    # Clockwise rotation is defined as negative
    # force and cart movement to the right are defined as positive
    # g (gravitational acceleration) is positive (absolute value)
    # Checked independently by Marcin and Krishna

    A = (k + 1) * (m_cart + m_pole) - m_pole * (ca ** 2)
    F_fric = - M_fric * positionD  # Force resulting from cart friction, notice that the mass of the cart is not explicitly there
    T_fric = - J_fric * angleD  # Torque resulting from pole friction
    L_half = L/2.0

    positionDD = (
            (
                    m_pole * g * sa * ca  # Movement of the cart due to gravity
                    + ((T_fric * ca) / L_half)  # Movement of the cart due to pend' s friction in the joint
                    + (k + 1) * (
                            - (m_pole * L_half * (
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
                    g * sa + positionDD * ca + T_fric / (m_pole * L_half)
            ) / ((k + 1) * L_half)
    )

    # making M go to infinity makes angleDD = (g/k*L_half)sin(angle) - angleD*J_fric/(k*m*L_half^2)
    # This is the same as equation derived directly for a pendulum.
    # k is 1/3! It is the factor for pendulum with length 2L: I = k*m*L_half^2

    return angleDD, positionDD

def cartpole_energy(ca, angleD, positionD,
                  m_cart, m_pole, g, L):
    L_half = L/2.0
    T_cart = m_cart * positionD ** 2 / 2
    T_pole_trans = m_pole * (positionD ** 2 - 2 * L_half * angleD * positionD * ca) / 2
    T_pole_rot = 2/3 * m_pole * L_half ** 2 * angleD ** 2
    V_pole = m_pole * g * L_half * ca

    E_total = T_cart + T_pole_trans + T_pole_rot + V_pole

    return E_total, T_cart, T_pole_trans, T_pole_rot, V_pole

def Q2u(Q, u_max):
    """
    Converts dimensionless motor power [-1,1] to a physical force acting on a cart.

    In future there might be implemented here a more sophisticated model of a motor driving CartPole
    """
    u = u_max * Q  # Q is drive -1:1 range, add noise on control

    return u


def euler_step(state, stateD, t_step):
    return state + stateD * t_step


###
# FIXME: Currently these equations are not modeling edge bounce!
#  The function for edge bounce is separate, it is used by simulator but not by predictors
###
class CartPoleEquations:
    supported_computation_libraries = (NumpyLibrary, TensorFlowLibrary, PyTorchLibrary)

    def __init__(self, lib=NumpyLibrary(), get_parameters_from=None, numba_compiled=False):
        self.lib = lib
        self.params = CartPoleParameters(lib, get_parameters_from)
        self.euler_step = CompileAdaptive(self.lib)(euler_step)  # This is a nested function, still it was compiled separately before for TF. TODO: Check if it is needed to compile it separately

        if numba_compiled:
            self._cartpole_ode = _cartpole_ode_numba
            self.edge_bounce = edge_bounce_numba
            self.cartpole_integration = cartpole_integration_euler_cromer_numba
        else:
            self._cartpole_ode = _cartpole_ode
            self.edge_bounce = edge_bounce
            # if isinstance(lib, NumpyLibrary):
            #     self.cartpole_integration = self._cartpole_integration_scipy
            # else:
            self.cartpole_integration = self._cartpole_integration_euler_cromer


    @CompileAdaptive
    def Q2u(self, Q):
        Q = self.lib.to_tensor(Q, self.lib.float32)
        u = Q2u(Q, u_max=self.params.u_max)
        return u

    def cartpole_ode_interface(self, s, u, **kwargs):

        k = kwargs.get('k', self.params.k)
        m_cart = kwargs.get('m_cart', self.params.m_cart)
        m_pole = kwargs.get('m_pole', self.params.m_pole)
        g = kwargs.get('g', self.params.g)
        J_fric = kwargs.get('J_fric', self.params.J_fric)
        M_fric = kwargs.get('M_fric', self.params.M_fric)
        L = kwargs.get('L', self.params.L)

        angleDD, positionDD = self._cartpole_ode(
            s[..., ANGLE_COS_IDX], s[..., ANGLE_SIN_IDX], s[..., ANGLED_IDX], s[..., POSITIOND_IDX], u,
            k=k, m_cart=m_cart, m_pole=m_pole, g=g, J_fric=J_fric, M_fric=M_fric, L=L
        )
        return angleDD, positionDD

    def cartpole_fine_integration(self, s, u, t_step, intermediate_steps, **kwargs):
        """
        Just an upper wrapper changing the way data is provided to the function _cartpole_fine_integration_tf
        """

        k = kwargs.get('k', self.params.k)
        m_cart = kwargs.get('m_cart', self.params.m_cart)
        m_pole = kwargs.get('m_pole', self.params.m_pole)
        g = kwargs.get('g', self.params.g)
        J_fric = kwargs.get('J_fric', self.params.J_fric)
        M_fric = kwargs.get('M_fric', self.params.M_fric)
        L = kwargs.get('L', self.params.L)

        (
            angle, angleD, position, positionD, angle_cos, angle_sin
        ) = self._cartpole_fine_integration(
            angle=s[..., ANGLE_IDX],
            angleD=s[..., ANGLED_IDX],
            angle_cos=s[..., ANGLE_COS_IDX],
            angle_sin=s[..., ANGLE_SIN_IDX],
            position=s[..., POSITION_IDX],
            positionD=s[..., POSITIOND_IDX],
            u=u,
            t_step=t_step,
            intermediate_steps=intermediate_steps,
            L=L,
            k=k, m_cart=m_cart, m_pole=m_pole, g=g, J_fric=J_fric, M_fric=M_fric
        )

        ### TODO: This is ugly! But I don't know how to resolve it...
        s_next = self.lib.stack([angle, angleD, angle_cos, angle_sin, position, positionD], axis=1)
        return s_next

    def _cartpole_fine_integration(self,
                                      angle, angleD,
                                      angle_cos, angle_sin,
                                      position, positionD,
                                      u, t_step,
                                      intermediate_steps,
                                      **kwargs):
        k = kwargs.get('k', self.params.k)
        m_cart = kwargs.get('m_cart', self.params.m_cart)
        m_pole = kwargs.get('m_pole', self.params.m_pole)
        g = kwargs.get('g', self.params.g)
        J_fric = kwargs.get('J_fric', self.params.J_fric)
        M_fric = kwargs.get('M_fric', self.params.M_fric)
        L = kwargs.get('L', self.params.L)

        def _step(counter, angle, angleD, position, positionD, angle_cos, angle_sin):
            # Find second derivative for CURRENT "k" step (same as in input).
            # State and u in input are from the same timestep, output is belongs also to THE same timestep ("k")
            angleDD, positionDD = self._cartpole_ode(angle_cos, angle_sin, angleD, positionD, u,
                                                   k, m_cart, m_pole, g, J_fric, M_fric, L)

            # Find NEXT "k+1" state [angle, angleD, position, positionD]
            angle, angleD, position, positionD = self.cartpole_integration(angle, angleD, angleDD, position,
                                                                      positionD,
                                                                      positionDD, t_step, u,
                                                                      k, m_cart, m_pole, g, J_fric, M_fric, L)

            # The edge bounce calculation seems to be too much for a GPU to tackle
            # angle_cos = tf.cos(angle)
            # angle, angleD, position, positionD = edge_bounce_wrapper(angle, angle_cos, angleD, position, positionD, t_step, L)

            angle_cos = self.lib.cos(angle)
            angle_sin = self.lib.sin(angle)

            angle = self.wrap_angle_rad(angle_sin, angle_cos)

            return counter + 1, angle, angleD, position, positionD, angle_cos, angle_sin

        # run the backend-specific loop
        (_,
         angle, angleD, position, positionD,
         angle_cos, angle_sin) = self.lib.loop(
             _step,
             state=(angle, angleD, position, positionD, angle_cos, angle_sin),
             steps=intermediate_steps,
             counter=0)

        return angle, angleD, position, positionD, angle_cos, angle_sin

    # def cartpole_dynamics(self, t, y, u):
    #     angle, angleD, position, positionD  = y
    #
    #     angleDD, positionDD = _cartpole_ode(np.cos(angle), np.sin(angle), angleD, positionD, u,
    #                       self.params.k, self.params.m_cart, self.params.m_pole, self.params.g, self.params.J_fric, self.params.M_fric, self.params.L)
    #
    #     return [angleD, angleDD, positionD, positionDD]
    #
    #
    # def _cartpole_integration_scipy(self, angle, angleD, angleDD, position, positionD, positionDD, t_step, u):
    #     y0 = [angle, angleD, position, positionD]
    #     t_span = [0, t_step]
    #
    #     sol = solve_ivp(self.cartpole_dynamics, t_span, y0, args=(u,), method='RK45')
    #
    #     angle_next, angleD_next, position_next, positionD_next = sol.y[:, -1]
    #
    #     return angle_next, angleD_next, position_next, positionD_next

    @CompileAdaptive
    def _cartpole_integration(self, angle, angleD, angleDD, position, positionD, positionDD, t_step, u=None,
                              k=None, m_cart=None, m_pole=None, g=None, J_fric=None, M_fric=None, L=None):
        angle_next = self.euler_step(angle, angleD, t_step)
        angleD_next = self.euler_step(angleD, angleDD, t_step)
        position_next = self.euler_step(position, positionD, t_step)
        positionD_next = self.euler_step(positionD, positionDD, t_step)

        return angle_next, angleD_next, position_next, positionD_next

    @CompileAdaptive
    def _cartpole_integration_euler_cromer(self, angle, angleD, angleDD, position, positionD, positionDD, t_step, u=None,
                              k=None, m_cart=None, m_pole=None, g=None, J_fric=None, M_fric=None, L=None):
        # Update velocities first
        angleD_next = self.euler_step(angleD, angleDD, t_step)
        positionD_next = self.euler_step(positionD, positionDD, t_step)

        # Then update positions using the updated velocities
        angle_next = self.euler_step(angle, angleD_next, t_step)
        position_next = self.euler_step(position, positionD_next, t_step)

        return angle_next, angleD_next, position_next, positionD_next


    @CompileAdaptive
    def wrap_angle_rad(self, sin, cos):
        return self.lib.atan2(sin, cos)

    # The edge bounce functions were not used at the time I was refactoring the code
    # But if I remember correctly from before,
    # they were working just very slow for Tensoroflow 31.01.2024
    # The function is not library independent.

    # @CompileAdaptive
    # def edge_bounce_wrapper(self, angle, angle_cos, angleD, position, positionD, t_step, L=L):
    #     angle_bounced = tf.TensorArray(tf.float32, size=tf.size(angle), dynamic_size=False)
    #     angleD_bounced = tf.TensorArray(tf.float32, size=tf.size(angleD), dynamic_size=False)
    #     position_bounced = tf.TensorArray(tf.float32, size=tf.size(position), dynamic_size=False)
    #     positionD_bounced = tf.TensorArray(tf.float32, size=tf.size(positionD), dynamic_size=False)
    #
    #     for i in tf.range(tf.size(position)):
    #         angle_i, angleD_i, position_i, positionD_i = self.edge_bounce(angle[i], angle_cos[i], angleD[i], position[i],
    #                                                                  positionD[i],
    #                                                                  t_step, L)
    #         angle_bounced = angle_bounced.write(i, angle_i)
    #         angleD_bounced = angleD_bounced.write(i, angleD_i)
    #         position_bounced = position_bounced.write(i, position_i)
    #         positionD_bounced = positionD_bounced.write(i, positionD_i)
    #
    #     angle_bounced_tensor = angle_bounced.stack()
    #     angleD_bounced_tensor = angleD_bounced.stack()
    #     position_bounced_tensor = position_bounced.stack()
    #     positionD_bounced_tensor = positionD_bounced.stack()
    #
    #     return angle_bounced_tensor, angleD_bounced_tensor, position_bounced_tensor, positionD_bounced_tensor


# FIXME: Notice that TrackHalfLength is provided in a very different way than other parameters to their respective functions
#   This may lead to unwanted behaviours.
def edge_bounce(angle, angle_cos, angleD, position, positionD, t_step, L):
    if position >= TrackHalfLength or -position >= TrackHalfLength:  # Without abs to compile with tensorflow
        angleD -= 2 * (positionD * angle_cos) / (0.5*L)
        angle += angleD * t_step
        positionD = -positionD
        position += positionD * t_step
    return angle, angleD, position, positionD


from numba import jit
_cartpole_ode_numba = jit(_cartpole_ode, nopython=True, cache=True, fastmath=True)
euler_step_numba = jit(euler_step, nopython=True, cache=True, fastmath=True)
edge_bounce_numba = jit(edge_bounce, nopython=True, cache=True, fastmath=True)


@jit(nopython=True, cache=True, fastmath=True)
def cartpole_integration_numba(angle, angleD, angleDD, position, positionD, positionDD, t_step, u=None,
                               k=None, m_cart=None, m_pole=None, g=None, J_fric=None, M_fric=None, L=None):
    angle_next = euler_step_numba(angle, angleD, t_step)
    angleD_next = euler_step_numba(angleD, angleDD, t_step)
    position_next = euler_step_numba(position, positionD, t_step)
    positionD_next = euler_step_numba(positionD, positionDD, t_step)

    return angle_next, angleD_next, position_next, positionD_next


@jit(nopython=True, cache=True, fastmath=True)
def cartpole_integration_euler_cromer_numba(angle, angleD, angleDD, position, positionD, positionDD, t_step, u=None,
                                            k=None, m_cart=None, m_pole=None, g=None, J_fric=None, M_fric=None, L=None):
    # Update velocities first
    angleD_next = euler_step_numba(angleD, angleDD, t_step)
    positionD_next = euler_step_numba(positionD, positionDD, t_step)

    # Then update positions using the updated velocities
    angle_next = euler_step_numba(angle, angleD_next, t_step)
    position_next = euler_step_numba(position, positionD_next, t_step)

    return angle_next, angleD_next, position_next, positionD_next

@jit(nopython=True, cache=True, fastmath=True)
def cartpole_integration_leapfrog_numba(angle, angleD, angleDD, position, positionD, positionDD, t_step, u,
                                        k, m_cart, m_pole, g, J_fric, M_fric, L):
    # Half step for velocities
    angleD_half = euler_step_numba(angleD, angleDD, 0.5 * t_step)
    positionD_half = euler_step_numba(positionD, positionDD, 0.5 * t_step)

    # Full step for positions
    angle_next = euler_step_numba(angle, angleD_half, t_step)
    position_next = euler_step_numba(position, positionD_half, t_step)

    # Compute new accelerations
    angleDD_next, positionDD_next = _cartpole_ode_numba(np.cos(angle_next), np.sin(angle_next), angleD_half, positionD_half, u,
                                                        k, m_cart, m_pole, g, J_fric, M_fric, L)

    # Another half step for velocities
    angleD_next = euler_step_numba(angleD_half, angleDD_next, 0.5 * t_step)
    positionD_next = euler_step_numba(positionD_half, positionDD_next, 0.5 * t_step)

    return angle_next, angleD_next, position_next, positionD_next


if __name__ == '__main__':
    import timeit
    import numpy as np
    from CartPole.state_utilities import create_cartpole_state

    s0 = create_cartpole_state()

    # Set non-zero input
    s = s0
    s[POSITION_IDX] = -30.2
    s[POSITIOND_IDX] = 2.87
    s[ANGLE_IDX] = -0.32
    s[ANGLED_IDX] = 0.237
    u = -0.24

    # Calculate time necessary to evaluate cartpole ODE:
    initialisation = '''
from CartPole.cartpole_equations import CartPoleEquations
cpe = CartPoleEquations(numba_compiled=True)
'''

    f_to_measure = 'angleDD, positionDD = cpe.cartpole_ode_interface(s, u)'
    number = 1  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 100000  # Gives how many times timeit should be repeated
    timings = timeit.Timer(f_to_measure, setup=initialisation, globals=globals()).repeat(repeat_timeit, number)
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
