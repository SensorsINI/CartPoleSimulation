import tensorflow as tf
from CartPole.cartpole_model import _cartpole_ode, euler_step, edge_bounce, cartpole_ode, edge_bounce_wrapper, cartpole_integration
from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, STATE_INDICES

from others.p_globals import (
    k, M, m, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength
)

from SI_Toolkit.TF.TF_Functions.Compile import Compile


###
# FIXME: Currently tf predictor is not modeling edge bounce!
###


_cartpole_ode_tf = Compile(_cartpole_ode)


euler_step_tf = Compile(euler_step)


edge_bounce_tf = Compile(edge_bounce)


@Compile
def wrap_angle_rad(sin, cos):
    return tf.math.atan2(sin, cos)


cartpole_ode_tf = Compile(cartpole_ode)

edge_bounce_wrapper_tf = Compile(edge_bounce_wrapper)

# @Compile
def edge_bounce_wrapper(angle, angle_cos, angleD, position, positionD, t_step, L=L):
    angle_bounced = tf.TensorArray(tf.float32, size=tf.size(angle), dynamic_size=False)
    angleD_bounced = tf.TensorArray(tf.float32, size=tf.size(angleD), dynamic_size=False)
    position_bounced = tf.TensorArray(tf.float32, size=tf.size(position), dynamic_size=False)
    positionD_bounced = tf.TensorArray(tf.float32, size=tf.size(positionD), dynamic_size=False)

    for i in tf.range(tf.size(position)):
        angle_i, angleD_i, position_i, positionD_i = edge_bounce(angle[i], angle_cos[i], angleD[i], position[i], positionD[i],
                                                                     t_step, L)
        angle_bounced = angle_bounced.write(i, angle_i)
        angleD_bounced = angleD_bounced.write(i, angleD_i)
        position_bounced = position_bounced.write(i, position_i)
        positionD_bounced = positionD_bounced.write(i, positionD_i)

    angle_bounced_tensor = angle_bounced.stack()
    angleD_bounced_tensor = angleD_bounced.stack()
    position_bounced_tensor = position_bounced.stack()
    positionD_bounced_tensor = positionD_bounced.stack()

    return angle_bounced_tensor, angleD_bounced_tensor, position_bounced_tensor, positionD_bounced_tensor


@Compile
def Q2u_tf(Q):
    """
    Converts dimensionless motor power [-1,1] to a physical force acting on a cart.

    In future there might be implemented here a more sophisticated model of a motor driving CartPole
    """
    u = tf.convert_to_tensor(u_max, dtype=tf.float32) * (
        Q +
        tf.convert_to_tensor(controlDisturbance, dtype=tf.float32) * tf.random.normal(shape=tf.shape(Q), dtype=tf.float32) + tf.convert_to_tensor(controlBias, dtype=tf.float32)
    )  # Q is drive -1:1 range, add noise on control

    return u


cartpole_integration_tf = Compile(cartpole_integration)


# @Compile
def _cartpole_fine_integration_tf(angle, angleD, angle_cos, angle_sin, position, positionD, u, t_step, intermediate_steps,
                              k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric, L=L):
    for _ in tf.range(intermediate_steps):
        # Find second derivative for CURRENT "k" step (same as in input).
        # State and u in input are from the same timestep, output is belongs also to THE same timestep ("k")
        angleDD, positionDD = _cartpole_ode(angle_cos, angle_sin, angleD, positionD, u,
                                                  k, M, m, g, J_fric, M_fric, L)

        # Find NEXT "k+1" state [angle, angleD, position, positionD]
        angle, angleD, position, positionD = cartpole_integration(angle, angleD, angleDD, position, positionD,
                                                                  positionDD, t_step, )

        # The edge bounce calculation seems to be too much for a GPU to tackle
        # angle_cos = tf.cos(angle)
        # angle, angleD, position, positionD = edge_bounce_wrapper(angle, angle_cos, angleD, position, positionD, t_step, L)

        angle_cos = tf.cos(angle)
        angle_sin = tf.sin(angle)

        angle = wrap_angle_rad(angle_sin, angle_cos)

    return angle, angleD, position, positionD, angle_cos, angle_sin

def cartpole_fine_integration_tf(s, u, t_step, intermediate_steps,
                              k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric, L=L):
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
        k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric
    )

    ### TODO: This is ugly! But I don't know how to resolve it...
    s_next = tf.transpose(tf.stack([angle, angleD, angle_cos, angle_sin, position, positionD]))
    return s_next