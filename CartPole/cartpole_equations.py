from types import SimpleNamespace

from SI_Toolkit.computation_library import NumpyLibrary, PyTorchLibrary, TensorFlowLibrary

from SI_Toolkit.Functions.TF.Compile import CompileAdaptive

from others.p_globals import export_parameters, TrackHalfLength
from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                      ANGLED_IDX, POSITION_IDX, POSITIOND_IDX)




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

    positionDD = (
            (
                    m_pole * g * sa * ca  # Movement of the cart due to gravity
                    + ((T_fric * ca) / L)  # Movement of the cart due to pend' s friction in the joint
                    + (k + 1) * (
                            - (m_pole * L * (
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
                    g * sa + positionDD * ca + T_fric / (m_pole * L)
            ) / ((k + 1) * L)
    )

    # making M go to infinity makes angleDD = (g/k*L)sin(angle) - angleD*J_fric/(k*m*L^2)
    # This is the same as equation derived directly for a pendulum.
    # k is 4/3! It is the factor for pendulum with length 2L: I = k*m*L^2

    return angleDD, positionDD





###
# FIXME: Currently these equations are not modeling edge bounce!
###


def euler_step(state, stateD, t_step):
    return state + stateD * t_step


class CartPoleEquations:
    supported_computation_libraries: set = {NumpyLibrary, TensorFlowLibrary, PyTorchLibrary}

    def __init__(self, lib=NumpyLibrary):
        self.lib = lib
        (self.k, self.m_cart, self.m_pole, self.g, self.J_fric,
         self.M_fric, self.L, self.v_max, self.u_max, self.controlDisturbance,
         self.controlBias, self.TrackHalfLength) = export_parameters(lib)

        # Compiling
        self.euler_step = CompileAdaptive(self.lib)(euler_step)

    def export_parameters(self):
        return (self.k, self.m_cart, self.m_pole, self.g, self.J_fric,
         self.M_fric, self.L, self.v_max, self.u_max, self.controlDisturbance,
         self.controlBias, self.TrackHalfLength)

    @CompileAdaptive
    def Q2u(self, Q):
        """
        Converts dimensionless motor power [-1,1] to a physical force acting on a cart.

        In future there might be implemented here a more sophisticated model of a motor driving CartPole
        """
        u = self.lib.to_tensor(
            self.u_max * Q,  # Q is drive -1:1 range, add noise on control
            self.lib.float32
        )

        return u

    def cartpole_fine_integration(self, s, u, t_step, intermediate_steps, **kwargs):
        """
        Just an upper wrapper changing the way data is provided to the function _cartpole_fine_integration_tf
        """

        k = kwargs.get('k', self.k)
        m_cart = kwargs.get('m_cart', self.m_cart)
        m_pole = kwargs.get('m_pole', self.m_pole)
        g = kwargs.get('g', self.g)
        J_fric = kwargs.get('J_fric', self.J_fric)
        M_fric = kwargs.get('M_fric', self.M_fric)
        L = kwargs.get('L', self.L)

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
        k = kwargs.get('k', self.k)
        m_cart = kwargs.get('m_cart', self.m_cart)
        m_pole = kwargs.get('m_pole', self.m_pole)
        g = kwargs.get('g', self.g)
        J_fric = kwargs.get('J_fric', self.J_fric)
        M_fric = kwargs.get('M_fric', self.M_fric)
        L = kwargs.get('L', self.L)

        for _ in self.lib.arange(0, intermediate_steps):
            # Find second derivative for CURRENT "k" step (same as in input).
            # State and u in input are from the same timestep, output is belongs also to THE same timestep ("k")
            angleDD, positionDD = _cartpole_ode(angle_cos, angle_sin, angleD, positionD, u,
                                                   k, m_cart, m_pole, g, J_fric, M_fric, L)

            # Find NEXT "k+1" state [angle, angleD, position, positionD]
            angle, angleD, position, positionD = self.cartpole_integration(angle, angleD, angleDD, position,
                                                                      positionD,
                                                                      positionDD, t_step, )

            # The edge bounce calculation seems to be too much for a GPU to tackle
            # angle_cos = tf.cos(angle)
            # angle, angleD, position, positionD = edge_bounce_wrapper(angle, angle_cos, angleD, position, positionD, t_step, L)

            angle_cos = self.lib.cos(angle)
            angle_sin = self.lib.sin(angle)

            angle = self.wrap_angle_rad(angle_sin, angle_cos)

        return angle, angleD, position, positionD, angle_cos, angle_sin

    @CompileAdaptive
    def cartpole_integration(self, angle, angleD, angleDD, position, positionD, positionDD, t_step):
        angle_next = self.euler_step(angle, angleD, t_step)
        angleD_next = self.euler_step(angleD, angleDD, t_step)
        position_next = self.euler_step(position, positionD, t_step)
        positionD_next = self.euler_step(positionD, positionDD, t_step)

        return angle_next, angleD_next, position_next, positionD_next


    @CompileAdaptive
    def wrap_angle_rad(self, sin, cos):
        return self.lib.atan2(sin, cos)

    def cartpole_ode_namespace(self, s: SimpleNamespace, u: float, **kwargs):

        k = kwargs.get('k', self.k)
        m_cart = kwargs.get('m_cart', self.m_cart)
        m_pole = kwargs.get('m_pole', self.m_pole)
        g = kwargs.get('g', self.g)
        J_fric = kwargs.get('J_fric', self.J_fric)
        M_fric = kwargs.get('M_fric', self.M_fric)
        L = kwargs.get('L', self.L)

        angleDD, positionDD = _cartpole_ode(
            self.lib.cos(s.angle), self.lib.sin(s.angle), s.angleD, s.positionD, u,
            k=k, m_cart=m_cart, m_pole=m_pole, g=g, J_fric=J_fric, M_fric=M_fric, L=L
        )
        return angleDD, positionDD

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
    #         angle_i, angleD_i, position_i, positionD_i = edge_bounce(angle[i], angle_cos[i], angleD[i], position[i],
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


def edge_bounce(angle, angle_cos, angleD, position, positionD, t_step, L):
    if position >= TrackHalfLength or -position >= TrackHalfLength:  # Without abs to compile with tensorflow
        angleD -= 2 * (positionD * angle_cos) / L
        angle += angleD * t_step
        positionD = -positionD
        position += positionD * t_step
    return angle, angleD, position, positionD