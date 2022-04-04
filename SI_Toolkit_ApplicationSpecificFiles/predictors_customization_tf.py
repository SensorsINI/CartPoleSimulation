import tensorflow as tf

from CartPole.state_utilities import STATE_INDICES, STATE_VARIABLES, CONTROL_INPUTS, CONTROL_INDICES, create_cartpole_state
from CartPole.state_utilities import ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX

from CartPole.cartpole_tf import cartpole_fine_integration_tf, Q2u_tf
from CartPole.cartpole_model import L

from SI_Toolkit.TF.TF_Functions.Compile import Compile
import numpy as np

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

    @Compile
    def step(self, s, Q, params):

        # assers does not work with Compile, but left here for information
        # assert Q.shape[0] == s.shape[0]
        # assert Q.ndim == 2
        # assert s.ndim == 2

        if params is None:
            pole_half_length = tf.convert_to_tensor(L, dtype=tf.float32)
        else:
            pole_half_length = tf.convert_to_tensor(params, dtype=tf.float32)

        Q = tf.squeeze(Q, axis=1)  # Removes features dimension, specific for cartpole as it has only one control input

        u = Q2u_tf(Q)

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


class predictor_output_augmentation_tf:
    def __init__(self, net_info):
        self.net_output_indices = {key: value for value, key in enumerate(net_info.outputs)}
        indices_augmentation = []
        features_augmentation = []
        if 'angle' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['angle'])
            features_augmentation.append('angle')
        if 'angle_sin' not in net_info.outputs and 'angle' in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['angle_sin'])
            features_augmentation.append('angle_sin')
        if 'angle_cos' not in net_info.outputs and 'angle' in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['angle_cos'])
            features_augmentation.append('angle_cos')

        self.indices_augmentation = indices_augmentation
        self.features_augmentation = features_augmentation
        self.augmentation_len = len(self.indices_augmentation)

        if 'angle' in net_info.outputs:
            self.index_angle = tf.convert_to_tensor(self.net_output_indices['angle'])
        if 'angle_sin' in net_info.outputs:
            self.index_angle_sin = tf.convert_to_tensor(self.net_output_indices['angle_sin'])
        if 'angle_cos' in net_info.outputs:
            self.index_angle_cos = tf.convert_to_tensor(self.net_output_indices['angle_cos'])

    def get_indices_augmentation(self):
        return self.indices_augmentation

    def get_features_augmentation(self):
        return self.features_augmentation

    @Compile
    def augment(self, net_output):

        output = net_output
        if 'angle' in self.features_augmentation:
            angle = \
                tf.math.atan2(
                    net_output[..., self.index_angle_sin],
                    net_output[..., self.index_angle_cos])[:, :, tf.newaxis]
            output = tf.concat([output, angle], axis=-1)

        if 'angle_sin' in self.features_augmentation:
            angle_sin = \
                tf.sin(net_output[..., self.index_angle])
            output = tf.concat([output, angle_sin], axis=-1)

        if 'angle_cos' in self.features_augmentation:
            angle_cos = \
                tf.cos(net_output[..., self.index_angle])
            output = tf.concat([output, angle_cos], axis=-1)

        return output
