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
    else:
        label = feature

    return label


def convert_units_inplace(ground_truth, predictions_list, features):

    for i in range(len(predictions_list)):
        for feature in features:
            feature_idx = features.index(feature)

            predictions_array = predictions_array[:, :, feature_idx]

            if feature == 'angle':
                ground_truth[:, feature_idx] *= 180.0/np.pi
                predictions_array[:, :, feature_idx] *= 180.0/np.pi
            elif feature == 'angleD':
                ground_truth[:, feature_idx] *= 180.0 / np.pi
                predictions_array[:, :, feature_idx] *= 180.0 / np.pi
            elif feature == 'angle_cos':
                pass
            elif feature == 'angle_sin':
                pass
            elif feature == 'position':
                ground_truth[:, feature_idx] *= 100.0
                predictions_array[:, :, feature_idx] *= 100.0
            elif feature == 'positionD':
                ground_truth[:, feature_idx] *= 100.0
                predictions_array[:, :, feature_idx] *= 100.0
            else:
                pass

            predictions_list[i] = predictions_array
