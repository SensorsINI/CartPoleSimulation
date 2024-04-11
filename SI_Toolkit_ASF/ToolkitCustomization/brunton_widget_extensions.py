"""
Adds application specific additions to brunton GUI.
get_feature_label allows replacing feature names with more descriptive labels.
convert_units_inplace converts units of the data to the desired units for display only.
All changes are done for GUI only, no changes to underlying dataset.
"""

import numpy as np


def get_feature_label(feature):

    if feature == 'angle':
        label = "Pole's Angle [deg]"
    elif feature == 'angleD':
        label = "Pole's Angular Velocity [deg/s]"
    elif feature == 'angle_cos':
        label = "Angle-Cosine"
    elif feature == 'angle_sin':
        label = "Angle-Sine"
    elif feature == 'position':
        label = "Cart's Position [cm]"
    elif feature == 'positionD':
        label = "Cart's Velocity [cm/s]"
    elif feature == 'L':
        label = 'Pole length [cm]'
    else:
        label = feature

    return label


def convert_units_inplace(ground_truth, predictions_list):
    ground_truth_dataset, ground_truth_features = ground_truth
    # Convert ground truth
    for feature in ground_truth_features:
        feature_idx, = np.where(ground_truth_features == feature)
        if feature == 'angle':
            ground_truth_dataset[:, feature_idx] *= 180.0 / np.pi
        elif feature == 'angleD':
            ground_truth_dataset[:, feature_idx] *= 180.0 / np.pi
        elif feature == 'angle_cos':
            pass
        elif feature == 'angle_sin':
            pass
        elif feature == 'position':
            ground_truth_dataset[:, feature_idx] *= 100.0
        elif feature == 'positionD':
            ground_truth_dataset[:, feature_idx] *= 100.0
        elif feature == 'L':
            ground_truth_dataset[:, feature_idx] *= 200.0  # Recalculating from pole half length to full length and from m to cm
        else:
            pass

    # Convert predictions
    for i in range(len(predictions_list)):
        predictions_array, features, _ = predictions_list[i]
        for feature in features:
            feature_idx, = np.where(features == feature)

            if feature == 'angle':
                predictions_array[:, :, feature_idx] *= 180.0/np.pi
            elif feature == 'angleD':
                predictions_array[:, :, feature_idx] *= 180.0 / np.pi
            elif feature == 'angle_cos':
                pass
            elif feature == 'angle_sin':
                pass
            elif feature == 'position':
                predictions_array[:, :, feature_idx] *= 100.0
            elif feature == 'positionD':
                predictions_array[:, :, feature_idx] *= 100.0
            elif feature == 'L':
                predictions_array[:, :, feature_idx] *= 200.0
            else:
                pass

            predictions_list[i][0] = predictions_array
