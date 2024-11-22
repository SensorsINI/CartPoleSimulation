"""
A class for data augmentation.
The class is used to modify the input features and output targets of the dataset
for neural network training.
"""

import numpy as np


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

    def series_modification(self, features, targets):

        if self.mode == 'train_for_random_vertical_angle_shift':
            # # CARTPOLE ONLY - Training for shift 0-angle # TODO: Find better way to integrate this part and log the changes
            # # # Find the index for 'angle_cos'
            index_cos = self.inputs.index('angle_cos')
            index_sin = self.inputs.index('angle_sin')

            angle_cos = features[:, index_cos]
            angle_sin = features[:, index_sin]

            # Compute the original angles using arctan2, which takes into account the quadrant of the angle
            angles = np.arctan2(angle_sin, angle_cos)

            # Generate random values uniformly drawn from the interval [-π, π]
            random_angle_offset = np.random.uniform(-np.pi, np.pi)

            # Calculate sine and cosine of 'angle_offset' and add as new columns

            if 'angle_offset' in self.outputs:
                targets[:, self.outputs.index('angle_offset')] = random_angle_offset

            if 'angle_offset_cos' in self.outputs:
                targets[:, self.outputs.index('angle_offset_cos')] = np.cos(random_angle_offset)

            if 'angle_offset_sin' in self.outputs:
                targets[:, self.outputs.index('angle_offset_sin')] = np.sin(random_angle_offset)

            # Add the random values to the original angles
            new_angles = angles + random_angle_offset

            # Compute the new cosine and sine values from the modified angles
            new_angle_cos = np.cos(new_angles)
            new_angle_sin = np.sin(new_angles)

            features[:, index_cos] = new_angle_cos
            features[:, index_sin] = new_angle_sin

        features = features * (1 + self.noise_level_features * np.random.uniform(-1.0, 1.0, size=features.shape))

        return features, targets
