"""
A class for data augmentation.
The class is used to modify the input features and output targets of the dataset
for neural network training.
"""

import math
import tensorflow as tf


class DataAugmentation:
    def __init__(self, inputs, outputs, config_series_modification) -> None:

        self.inputs = inputs
        self.outputs = outputs

        self.noise_level_features = config_series_modification['NOISE_LEVEL']['FEATURES']
        self.mode = config_series_modification['MODE']

        self.columns_created = None

        if self.mode == 'None':
            self.columns_created = []
        elif self.mode == 'train_for_random_vertical_angle_shift':
            self.columns_created = ['angle_offset', 'angle_offset_cos', 'angle_offset_sin']

        if self.columns_created is None:
            raise ValueError('self.columns_created is None. This should not happen at the end of the function.')

    @staticmethod
    def _replace_column(tensor: tf.Tensor, index: int, new_vals: tf.Tensor) -> tf.Tensor:
        """
        Replace one column (last‑axis slice) with `new_vals` while preserving
        tensor immutability.  This avoids scatter operations and therefore
        executes efficiently on both eager and graph modes.
        """
        # Ensure new_vals has a singleton last‑dimension so it can be concatenated
        new_vals = tf.expand_dims(new_vals, axis=-1)
        return tf.concat([tensor[..., :index], new_vals, tensor[..., index + 1:]], axis=-1)

    def series_modification(self, features, targets):

        # Convert to tensors in case numpy arrays were passed
        features = tf.convert_to_tensor(features)
        targets = tf.convert_to_tensor(targets)

        if self.mode == 'train_for_random_vertical_angle_shift':
            index_cos = self.inputs.index('angle_cos')
            index_sin = self.inputs.index('angle_sin')

            angle_cos = features[..., index_cos]  # (B, T)
            angle_sin = features[..., index_sin]

            angles = tf.atan2(angle_sin, angle_cos)  # (B, T)

            # -------- 1. sample one offset per sequence (shape (B, 1)) -------------
            f_dtype = features.dtype
            B = tf.shape(features)[0]
            offset = tf.random.uniform([B, 1],  # (B, 1)
                                       minval=-math.pi,
                                       maxval=math.pi,
                                       dtype=f_dtype)

            # -------- 2. expand/broadcast along the time axis ----------------------
            random_angle_offset = tf.broadcast_to(offset, tf.shape(angles))  # (B, T)

            # -------- 3. write targets --------------------------------------------
            if 'angle_offset' in self.outputs:
                col = self.outputs.index('angle_offset')
                targets = self._replace_column(targets, col, random_angle_offset)

            if 'angle_offset_cos' in self.outputs:
                col = self.outputs.index('angle_offset_cos')
                targets = self._replace_column(targets, col, tf.cos(random_angle_offset))

            if 'angle_offset_sin' in self.outputs:
                col = self.outputs.index('angle_offset_sin')
                targets = self._replace_column(targets, col, tf.sin(random_angle_offset))

            # -------- 4. update features ------------------------------------------
            new_angles = angles + random_angle_offset  # (B, T)
            new_angle_cos = tf.cos(new_angles)
            new_angle_sin = tf.sin(new_angles)

            features = self._replace_column(features, index_cos, new_angle_cos)
            features = self._replace_column(features, index_sin, new_angle_sin)

        # Feature‑wise multiplicative noise
        noise = 1.0 + self.noise_level_features * tf.random.uniform(
            tf.shape(features), minval=-1.0, maxval=1.0, dtype=features.dtype
        )
        features = features * noise

        return features, targets
