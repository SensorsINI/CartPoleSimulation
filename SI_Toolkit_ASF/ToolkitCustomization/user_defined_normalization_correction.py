"""
This is auxiliary file used while running ./SI_Toolkit_ASF/run/Create_normalization_file.py.
If apply_user_defined_normalization_correction is defined it allows overwriting selected data statistics,
mean, std, max, min, replacing values calculated from data with user defined values.
This is useful if you a priori know that some features should be symmetric or cover some particular range.
"""

import numpy as np
from CartPole.cartpole_parameters import TrackHalfLength, u_max

def apply_user_defined_normalization_correction(df_norm_info):

    df_norm_info.loc['mean', 'angle'] = 0.0
    df_norm_info.loc['std', 'angle'] = df_norm_info.loc['std', 'angle']
    df_norm_info.loc['max', 'angle'] = np.pi
    df_norm_info.loc['min', 'angle'] = -np.pi

    df_norm_info.loc['mean', 'angleD'] = 0.0
    df_norm_info.loc['std', 'angleD'] = df_norm_info.loc['std', 'angleD']
    df_norm_info.loc['max', 'angleD'] = max(abs(df_norm_info.loc['max', 'angleD']),
                                            abs(df_norm_info.loc['min', 'angleD']))
    df_norm_info.loc['min', 'angleD'] = - df_norm_info.loc['max', 'angleD']

    df_norm_info.loc['mean', 'angleDD'] = 0.0
    df_norm_info.loc['std', 'angleDD'] = df_norm_info.loc['std', 'angleDD']
    df_norm_info.loc['max', 'angleDD'] = max(abs(df_norm_info.loc['max', 'angleDD']),
                                             abs(df_norm_info.loc['min', 'angleDD']))
    df_norm_info.loc['min', 'angleDD'] = - df_norm_info.loc['max', 'angleDD']

    df_norm_info.loc['mean', 'angle_cos'] = 0.0
    df_norm_info.loc['std', 'angle_cos'] = df_norm_info.loc['std', 'angle_cos']
    df_norm_info.loc['max', 'angle_cos'] = 1.0
    df_norm_info.loc['min', 'angle_cos'] = - 1.0

    df_norm_info.loc['mean', 'angle_sin'] = 0.0
    df_norm_info.loc['std', 'angle_sin'] = df_norm_info.loc['std', 'angle_sin']
    df_norm_info.loc['max', 'angle_sin'] = 1.0
    df_norm_info.loc['min', 'angle_sin'] = - 1.0

    df_norm_info.loc['mean', 'position'] = 0.0
    df_norm_info.loc['std', 'position'] = df_norm_info.loc['std', 'position']  # no change
    df_norm_info.loc['max', 'position'] = np.float32(TrackHalfLength)
    df_norm_info.loc['min', 'position'] = -np.float32(TrackHalfLength)

    df_norm_info.loc['mean', 'positionD'] = 0.0
    df_norm_info.loc['std', 'positionD'] = df_norm_info.loc['std', 'positionD']
    df_norm_info.loc['max', 'positionD'] = max(abs(df_norm_info.loc['max', 'positionD']),
                                               abs(df_norm_info.loc['min', 'positionD']))
    df_norm_info.loc['min', 'positionD'] = - df_norm_info.loc['max', 'positionD']

    df_norm_info.loc['mean', 'positionDD'] = 0.0
    df_norm_info.loc['std', 'positionDD'] = df_norm_info.loc['std', 'positionDD']
    df_norm_info.loc['max', 'positionDD'] = max(abs(df_norm_info.loc['max', 'positionDD']),
                                                abs(df_norm_info.loc['min', 'positionDD']))
    df_norm_info.loc['min', 'positionDD'] = - df_norm_info.loc['max', 'positionDD']

    df_norm_info.loc['mean', 'u'] = 0.0
    df_norm_info.loc['std', 'u'] = df_norm_info.loc['std', 'u']
    df_norm_info.loc['max', 'u'] = np.float32(u_max)
    df_norm_info.loc['min', 'u'] = - np.float32(u_max)

    df_norm_info.loc['mean', 'Q'] = 0.0
    df_norm_info.loc['std', 'Q'] = df_norm_info.loc['std', 'Q']
    df_norm_info.loc['max', 'Q'] = 1.0
    df_norm_info.loc['min', 'Q'] = - 1.0

    df_norm_info.loc['mean', 'target_position'] = 0.0
    df_norm_info.loc['std', 'target_position'] = df_norm_info.loc['std', 'target_position']
    df_norm_info.loc['max', 'target_position'] = np.float32(TrackHalfLength)
    df_norm_info.loc['min', 'target_position'] = -np.float32(TrackHalfLength)


    try:  # This is a separate try-except clause because at this time the correct calculation of angle derivative (unwrapping!) was not implemented
        df_norm_info.loc['mean', 'D_angle'] = 0.0
        df_norm_info.loc['std', 'D_angle'] = df_norm_info.loc['std', 'D_angle']
        df_norm_info.loc['max', 'D_angle'] = max(abs(df_norm_info.loc['max', 'D_angle']),
                                                abs(df_norm_info.loc['min', 'D_angle']))
        df_norm_info.loc['min', 'D_angle'] = - df_norm_info.loc['max', 'D_angle']
    except KeyError:
        pass

    try:
        df_norm_info.loc['mean', 'D_angleD'] = 0.0
        df_norm_info.loc['std', 'D_angleD'] = df_norm_info.loc['std', 'D_angleD']
        df_norm_info.loc['max', 'D_angleD'] = max(abs(df_norm_info.loc['max', 'D_angleD']),
                                                abs(df_norm_info.loc['min', 'D_angleD']))
        df_norm_info.loc['min', 'D_angleD'] = - df_norm_info.loc['max', 'D_angleD']

        df_norm_info.loc['mean', 'D_angle_cos'] = 0.0
        df_norm_info.loc['std', 'D_angle_cos'] = df_norm_info.loc['std', 'D_angle_cos']
        df_norm_info.loc['max', 'D_angle_cos'] = max(abs(df_norm_info.loc['max', 'D_angle_cos']),
                                                abs(df_norm_info.loc['min', 'D_angle_cos']))
        df_norm_info.loc['min', 'D_angle_cos'] = - df_norm_info.loc['max', 'D_angle_cos']

        df_norm_info.loc['mean', 'D_angle_sin'] = 0.0
        df_norm_info.loc['std', 'D_angle_sin'] = df_norm_info.loc['std', 'D_angle_sin']
        df_norm_info.loc['max', 'D_angle_sin'] = max(abs(df_norm_info.loc['max', 'D_angle_sin']),
                                                abs(df_norm_info.loc['min', 'D_angle_sin']))
        df_norm_info.loc['min', 'D_angle_sin'] = - df_norm_info.loc['max', 'D_angle_sin']

        df_norm_info.loc['mean', 'D_position'] = 0.0
        df_norm_info.loc['std', 'D_position'] = df_norm_info.loc['std', 'D_position']  # no change
        df_norm_info.loc['max', 'D_position'] = max(abs(df_norm_info.loc['max', 'D_position']),
                                                   abs(df_norm_info.loc['min', 'D_position']))
        df_norm_info.loc['min', 'D_position'] = - df_norm_info.loc['max', 'D_position']

        df_norm_info.loc['mean', 'D_positionD'] = 0.0
        df_norm_info.loc['std', 'D_positionD'] = df_norm_info.loc['std', 'D_positionD']
        df_norm_info.loc['max', 'D_positionD'] = max(abs(df_norm_info.loc['max', 'D_positionD']),
                                                   abs(df_norm_info.loc['min', 'D_positionD']))
        df_norm_info.loc['min', 'D_positionD'] = - df_norm_info.loc['max', 'D_positionD']

    except KeyError:
        pass


    return df_norm_info