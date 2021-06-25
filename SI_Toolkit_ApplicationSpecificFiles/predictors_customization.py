import numpy as np

from CartPole.state_utilities import STATE_INDICES, STATE_VARIABLES, CONTROL_INPUTS

def augment_predictor_output(output_array, net_info):

    if 'angle' not in net_info.outputs:
        output_array[..., STATE_INDICES['angle']] = \
            np.arctan2(
                output_array[..., STATE_INDICES['angle_sin']],
                output_array[..., STATE_INDICES['angle_cos']])
    if 'angle_sin' not in net_info.outputs:
        output_array[..., STATE_INDICES['angle_sin']] = \
            np.sin(output_array[..., STATE_INDICES['angle']])
    if 'angle_cos' not in net_info.outputs:
        output_array[..., STATE_INDICES['angle_cos']] = \
            np.sin(output_array[..., STATE_INDICES['angle']])

    return output_array
