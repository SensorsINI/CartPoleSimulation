import numpy as np
from others.p_globals import P_GLOBALS


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
    df_norm_info.loc['max', 'position'] = P_GLOBALS.TrackHalfLength
    df_norm_info.loc['min', 'position'] = -P_GLOBALS.TrackHalfLength

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
    df_norm_info.loc['max', 'u'] = P_GLOBALS.u_max
    df_norm_info.loc['min', 'u'] = - P_GLOBALS.u_max

    df_norm_info.loc['mean', 'Q'] = 0.0
    df_norm_info.loc['std', 'Q'] = df_norm_info.loc['std', 'Q']
    df_norm_info.loc['max', 'Q'] = 1.0
    df_norm_info.loc['min', 'Q'] = - 1.0

    df_norm_info.loc['mean', 'target_position'] = 0.0
    df_norm_info.loc['std', 'target_position'] = df_norm_info.loc['std', 'target_position']
    df_norm_info.loc['max', 'target_position'] = P_GLOBALS.TrackHalfLength
    df_norm_info.loc['min', 'target_position'] = -P_GLOBALS.TrackHalfLength

    return df_norm_info