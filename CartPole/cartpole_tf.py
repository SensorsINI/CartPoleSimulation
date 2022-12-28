import tensorflow as tf
from others.globals_and_utils import create_rng, load_config
from others.p_globals import (J_fric, L, m_cart, M_fric, TrackHalfLength,
                              controlBias, controlDisturbance, g, k, m_pole, u_max,
                              v_max)
from SI_Toolkit.Functions.TF.Compile import CompileTF

from CartPole.cartpole_model_tf import (_cartpole_ode, cartpole_integration_tf,
                                        cartpole_ode, edge_bounce,
                                        edge_bounce_wrapper)
from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                      ANGLED_IDX, POSITION_IDX, POSITIOND_IDX)

config = load_config("config.yml")

k = tf.convert_to_tensor(k)
m_cart = tf.convert_to_tensor(m_cart)
m_pole = tf.convert_to_tensor(m_pole)
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

###
# FIXME: Currently tf predictor is not modeling edge bounce!
###


# _cartpole_ode_tf = tf.function(_cartpole_ode, jit_compile=True)
_cartpole_ode_tf = _cartpole_ode

# euler_step_tf = tf.function(euler_step, jit_compile = True)


# edge_bounce_tf = tf.function(edge_bounce, jit_compile=True)
edge_bounce_tf = edge_bounce


@CompileTF
def wrap_angle_rad(sin, cos):
    return tf.math.atan2(sin, cos)


# cartpole_ode_tf = tf.function(cartpole_ode, jit_compile=True)
cartpole_ode_tf = cartpole_ode

# edge_bounce_wrapper_tf = tf.function(edge_bounce_wrapper, jit_compile=True)
edge_bounce_wrapper_tf = edge_bounce_wrapper


@CompileTF
def edge_bounce_wrapper(angle, angle_cos, angleD, position, positionD, t_step, L=L):
    angle_bounced = tf.TensorArray(tf.float32, size=tf.size(angle), dynamic_size=False)
    angleD_bounced = tf.TensorArray(tf.float32, size=tf.size(angleD), dynamic_size=False)
    position_bounced = tf.TensorArray(tf.float32, size=tf.size(position), dynamic_size=False)
    positionD_bounced = tf.TensorArray(tf.float32, size=tf.size(positionD), dynamic_size=False)

    for i in tf.range(tf.size(position)):
        angle_i, angleD_i, position_i, positionD_i = edge_bounce_tf(angle[i], angle_cos[i], angleD[i], position[i],
                                                                    positionD[i],
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


@CompileTF
def Q2u_tf(Q):
    """
    Converts dimensionless motor power [-1,1] to a physical force acting on a cart.

    In future there might be implemented here a more sophisticated model of a motor driving CartPole
    """
    u = tf.convert_to_tensor(u_max, dtype=tf.float32) * (
        Q
        + tf.convert_to_tensor(controlDisturbance * rng.standard_normal(size=Q.shape), dtype=tf.float32)
        + tf.convert_to_tensor(controlBias, dtype=tf.float32)
    )  # Q is drive -1:1 range, add noise on control

    return u


# cartpole_integration_tf = tf.function(cartpole_integration, jit_compile = True)


# @tf.function(jit_compile=True
#              ,input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.float32),
#                               tf.TensorSpec(shape=[None], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.float32),
#                               tf.TensorSpec(shape=[None], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.float32),
#                               tf.TensorSpec(shape=[None], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32),
#                               tf.TensorSpec(shape=[], dtype=tf.int32), tf.TensorSpec(shape=[], dtype=tf.float32),
#                               tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32),
#                               tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32),
#                               tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)])
@CompileTF
def _cartpole_fine_integration_tf(angle, angleD,
                                  angle_cos, angle_sin,
                                  position, positionD,
                                  u, t_step,
                                  intermediate_steps, k=k,
                                  m_cart=m_cart, m_pole=m_pole,
                                  g=g, J_fric=J_fric,
                                  M_fric=M_fric, L=L):
    #print('test 6')
    for _ in tf.range(intermediate_steps):
        # Find second derivative for CURRENT "k" step (same as in input).
        # State and u in input are from the same timestep, output is belongs also to THE same timestep ("k")
        angleDD, positionDD = _cartpole_ode_tf(angle_cos, angle_sin, angleD, positionD, u,
                                               k, m_cart, m_pole, g, J_fric, M_fric, L)

        # Find NEXT "k+1" state [angle, angleD, position, positionD]
        angle, angleD, position, positionD = cartpole_integration_tf(angle, angleD, angleDD, position, positionD,
                                                                     positionDD, t_step, )

        # The edge bounce calculation seems to be too much for a GPU to tackle
        # angle_cos = tf.cos(angle)
        # angle, angleD, position, positionD = edge_bounce_wrapper(angle, angle_cos, angleD, position, positionD, t_step, L)

        angle_cos = tf.cos(angle)
        angle_sin = tf.sin(angle)

        angle = wrap_angle_rad(angle_sin, angle_cos)
    #print('test 7')
    return angle, angleD, position, positionD, angle_cos, angle_sin


@CompileTF
def cartpole_fine_integration_tf(s, u, t_step, intermediate_steps,
                                 k=k, m_cart=m_cart, m_pole=m_pole, g=g, J_fric=J_fric, M_fric=M_fric, L=L):
    #print('test 5')
    """
    Calculates current values of second derivative of angle and position
    from current value of angle and position, and their first derivatives

    :param s: state of cartpole
        Angle is in radians, 0 vertical and increasing CCW.
        position is in meters, 0 in middle of track and increasing to right.
    :param u: cart accelerator in m/s^2 positive rightwards
    :param t_step: the timestep in seconds
    :param intermediate_steps: TODO what is this?
    :param k: TODO don't know what this is
    :param m_cart and m_pole: masses in kg of cart and pole.
    :param ca and sa: sin and cos of angle of pole.
    :param g: gravity in m/s^2
    :param J_fric and M_fric: friction coefficients in Nm per rad/s of pole  TODO check correct
    :param  M_fric: friction coefficient of cart in N per m/s TODO check correct
    :param L: length of pole in meters.

    :param u: Force applied on cart in Newtons TODO check is this correct?

    :returns: next state of s
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
    s_next = tf.stack([angle, angleD, angle_cos, angle_sin, position, positionD], axis=1)
    return s_next
