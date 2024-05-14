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


def calculete_additional_metrics(ground_truth, predictions_list):
    from CartPole.cartpole_equations import cartpole_energy
    from CartPole.cartpole_parameters import m_cart, m_pole, g, L

    # Ground truth
    ground_truth_dataset, ground_truth_features = ground_truth

    feature_idx, = np.where(ground_truth_features == 'angle_cos')
    ca = ground_truth_dataset[:, feature_idx]

    feature_idx, = np.where(ground_truth_features == 'angleD')
    angleD = ground_truth_dataset[:, feature_idx]

    feature_idx, = np.where(ground_truth_features == 'positionD')
    positionD = ground_truth_dataset[:, feature_idx]

    E_total, T_cart, T_pole_trans, T_pole_rot, V_pole = cartpole_energy(ca, angleD, positionD, m_cart, m_pole, g, L)

    new_data = np.concatenate((E_total, T_cart, T_pole_trans, T_pole_rot, V_pole), axis=-1)

    ground_truth_dataset = np.concatenate((ground_truth_dataset, new_data), axis=-1)
    ground_truth_features = np.array(list(ground_truth_features) + ['E_total', 'T_cart', 'T_pole', 'T_pole_rot', 'V_pole'])

    ground_truth = (ground_truth_dataset, ground_truth_features)

    # Predictions
    for i in range(len(predictions_list)):
        predictions_array, features, time_axis = predictions_list[i]

        feature_idx, = np.where(features == 'angle_cos')
        ca = predictions_array[:, :, feature_idx]

        feature_idx, = np.where(features == 'angleD')
        angleD = predictions_array[:, :, feature_idx]

        feature_idx, = np.where(features == 'positionD')
        positionD = predictions_array[:, :, feature_idx]

        E_total, T_cart, T_pole_trans, T_pole_rot, V_pole = cartpole_energy(ca, angleD, positionD, m_cart, m_pole, g, L)

        new_data = np.concatenate((E_total, T_cart, T_pole_trans, T_pole_rot, V_pole), axis=-1)

        predictions_array = np.concatenate((predictions_array, new_data), axis=-1)
        features = np.array(list(features) + ['E_total', 'T_cart', 'T_pole', 'T_pole_rot', 'V_pole'])

        predictions_list[i] = (predictions_array, features, time_axis)

    return ground_truth, predictions_list
