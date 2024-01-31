from SI_Toolkit.computation_library import NumpyLibrary, PyTorchLibrary, TensorFlowLibrary

from SI_Toolkit.Functions.TF.Compile import CompileAdaptive

from others.p_globals import export_parameters
from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                      ANGLED_IDX, POSITION_IDX, POSITIOND_IDX)

###
# FIXME: Currently these equations are not modeling edge bounce!
###

from CartPole.cartpole_model_tf import (_cartpole_ode, cartpole_integration_tf,
                                        cartpole_ode, edge_bounce,
                                        edge_bounce_wrapper)

_cartpole_ode_tf = _cartpole_ode

cartpole_ode_tf = cartpole_ode


class CartPoleEquations:
    def __init__(self, lib=NumpyLibrary):
        self.lib = lib
        k_global, m_cart_global, m_pole_global, g_global, J_fric_global, M_fric_global, L_global, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength = export_parameters(lib)

        def _wrap_angle_rad(sin, cos):
            return self.lib.atan2(sin, cos)

        wrap_angle_rad = CompileAdaptive(_wrap_angle_rad, computation_library=self.lib)

        def Q2u(Q):
            """
            Converts dimensionless motor power [-1,1] to a physical force acting on a cart.

            In future there might be implemented here a more sophisticated model of a motor driving CartPole
            """
            u = self.lib.to_tensor(
                u_max * Q,  # Q is drive -1:1 range, add noise on control
                self.lib.float32
            )

            return u

        self.Q2u = CompileAdaptive(Q2u, computation_library=self.lib)

        def cartpole_fine_integration(s, u, t_step, intermediate_steps,
                                         k=k_global, m_cart=m_cart_global, m_pole=m_pole_global, g=g_global, J_fric=J_fric_global, M_fric=M_fric_global, L=L_global):
            """
            Just an upper wrapper changing the way data is provided to the function _cartpole_fine_integration_tf
            """
            (
                angle, angleD, position, positionD, angle_cos, angle_sin
            ) = _cartpole_fine_integration_tf(
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

        self.cartpole_fine_integration = cartpole_fine_integration

        def _cartpole_fine_integration_tf(angle, angleD,
                                          angle_cos, angle_sin,
                                          position, positionD,
                                          u, t_step,
                                          intermediate_steps, k=k_global,
                                          m_cart=m_cart_global, m_pole=m_pole_global,
                                          g=g_global, J_fric=J_fric_global,
                                          M_fric=M_fric_global, L=L_global):
            # print('test 6')
            for _ in self.lib.arange(0, intermediate_steps):
                # Find second derivative for CURRENT "k" step (same as in input).
                # State and u in input are from the same timestep, output is belongs also to THE same timestep ("k")
                angleDD, positionDD = _cartpole_ode_tf(angle_cos, angle_sin, angleD, positionD, u,
                                                       k, m_cart, m_pole, g, J_fric, M_fric, L)

                # Find NEXT "k+1" state [angle, angleD, position, positionD]
                angle, angleD, position, positionD = cartpole_integration_tf(angle, angleD, angleDD, position,
                                                                             positionD,
                                                                             positionDD, t_step, )

                # The edge bounce calculation seems to be too much for a GPU to tackle
                # angle_cos = tf.cos(angle)
                # angle, angleD, position, positionD = edge_bounce_wrapper(angle, angle_cos, angleD, position, positionD, t_step, L)

                angle_cos = self.lib.cos(angle)
                angle_sin = self.lib.sin(angle)

                angle = wrap_angle_rad(angle_sin, angle_cos)
            # print('test 7')
            return angle, angleD, position, positionD, angle_cos, angle_sin



        # The edge bounce functions were not used at the time I was refactoring the code
        # But if I remember correctly from before,
        # they were working just very slow for Tensoroflow 31.01.2024
        # The function is not library independent.

        # def edge_bounce_wrapper(angle, angle_cos, angleD, position, positionD, t_step, L=L):
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
        #
        # self.edge_bounce_wrapper = CompileAdaptive(edge_bounce_wrapper, computation_library=self.lib)



