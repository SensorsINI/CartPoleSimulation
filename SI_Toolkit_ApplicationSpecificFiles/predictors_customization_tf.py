import tensorflow as tf

from CartPole.state_utilities import STATE_INDICES, STATE_VARIABLES, CONTROL_INPUTS, CONTROL_INDICES, create_cartpole_state
from CartPole.state_utilities import ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX

from CartPole.cartpole_tf import cartpole_fine_integration_tf, Q2u_tf
from CartPole.cartpole_model import L

STATE_INDICES_TF = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(STATE_INDICES.keys())), values=tf.constant(list(STATE_INDICES.values()))),
    default_value=-100, name=None
)


class next_state_predictor_ODE_tf():

    def __init__(self, dt, intermediate_steps):
        self.s = tf.convert_to_tensor(create_cartpole_state())

        self.intermediate_steps = tf.convert_to_tensor(intermediate_steps, dtype=tf.int32)
        self.t_step = tf.convert_to_tensor(dt / float(self.intermediate_steps), dtype=tf.float32)

    @tf.function(jit_compile = True
                 ,input_signature = [tf.TensorSpec(shape=[None,6], dtype=tf.float32),tf.TensorSpec(shape=[1,1], dtype=tf.float32)])

    def step(self, s, Q):

        # assers does not work with tf.function, but left here for information
        # assert Q.shape[0] == s.shape[0]
        # assert Q.ndim == 2
        # assert s.ndim == 2


        pole_half_length = tf.convert_to_tensor(L, dtype=tf.float32)


        Q = tf.squeeze(Q, axis=1)  # Removes features dimension, specific for cartpole as it has only one control input
        #print('test 3')
        u = Q2u_tf(Q)
        #print('test 4')
        (
            s_next
        ) = cartpole_fine_integration_tf(
            s,
            u=u,
            t_step=self.t_step,
            intermediate_steps=self.intermediate_steps,
            L=pole_half_length,
        )

        return s_next