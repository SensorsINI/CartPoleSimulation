import numpy as np

from matplotlib import colors

from tqdm import trange

import copy


from Modeling.load_and_normalize import denormalize_numpy_array, normalize_numpy_array
from Modeling.TF.TF_Functions.Network import load_internal_states, get_internal_states

from CartPole.state_utilities import STATE_INDICES_REDUCED

# This import mus go before pyplot so also before our scripts
from matplotlib import use, get_backend
# Use Agg if not in scientific mode of Pycharm
from Modeling.TF.TF_Functions.Initialization import get_net_and_norm_info

if get_backend() != 'module://backend_interagg':
    use('Agg')


cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

def get_data_for_gui_TF(a, dataset, dataset_sampling_dt, net_name):

    output_array = np.zeros(shape=(a.test_max_horizon + 1, a.test_len, len(a.features) + 1))
    output_array[0,:, :-1] = dataset[a.features].to_numpy()[:-a.test_max_horizon, :]
    # Create a copy of the network suitable for inference (stateful and with sequence length one)
    a = copy.deepcopy(a)
    a.net_name = net_name
    net_for_inference, net_for_inference_info, normalization_info = \
        get_net_and_norm_info(a, time_series_length=1,
                              batch_size=1, stateful=True)

    # Get features, target, and time axis
    # Format the experiment data
    features = dataset[net_for_inference_info.inputs].drop(columns=['Q']).to_numpy()[:-a.test_max_horizon, :]

    Q = dataset['Q'].to_numpy()
    Q_array = [Q[i:-a.test_max_horizon + i] for i in range(a.test_max_horizon)]
    Q_array = np.vstack(Q_array)
    output_array[:-1, :, -1] = Q_array


    features_normalized = normalize_numpy_array(features, net_for_inference_info.outputs, normalization_info)
    # Make a prediction
    normalized_net_output = np.zeros(shape=(a.test_max_horizon, a.test_len, len(net_for_inference_info.outputs)))

    internal_states = None
    mode = 'sequential'
    # mode = 'batch'
    if mode=='sequential':

        for timestep in trange(a.test_len):
            if timestep != 0:
                load_internal_states(net_for_inference, internal_states)
            for i in range(0, a.test_max_horizon):
                if i == 0:
                    net_input = np.hstack((Q_array[0, timestep, np.newaxis], features_normalized[timestep,:]))
                else:
                    net_input = np.hstack((Q_array[i, timestep, np.newaxis], normalized_net_output[i - 1, timestep, :]))

                if net_for_inference_info.net_type != 'Dense':
                    if net_input.ndim == 1:
                        net_input = np.expand_dims(net_input, axis=(0, 1))  # Add dimension for time for RNN
                    elif net_input.ndim == 2:
                        net_input = np.expand_dims(net_input, axis=1)  # Add time series dimension
                else:
                    if net_input.ndim == 1:
                        net_input = np.expand_dims(net_input, axis=0)  # Add batch dimension

                normalized_net_output[i, timestep, :] = np.squeeze(net_for_inference.predict_on_batch(net_input))

                if i == 0: # Update RNN internal states
                    internal_states = get_internal_states(net_for_inference)

    elif mode=='batch':
        for i in range(0, a.test_max_horizon):

            if i == 0:
                net_input = np.hstack((Q_array[0, :, np.newaxis], features_normalized))
            else:
                net_input = np.hstack((Q_array[i, :, np.newaxis], normalized_net_output[i-1,:,:]))

            if net_for_inference_info.net_type != 'Dense':
                if net_input.ndim == 1:
                    net_input = np.expand_dims(net_input, axis=(0, 1))  # Add dimension for time for RNN
                elif net_input.ndim == 2:
                    net_input = np.expand_dims(net_input, axis=1)

            normalized_net_output[i, :, :] = np.squeeze(net_for_inference.predict_on_batch(net_input))



    net_outputs_denormalized = denormalize_numpy_array(normalized_net_output, net_for_inference_info.outputs, normalization_info)

    for output_idx in range(len(net_for_inference_info.outputs)):
        output_array[1:,:,STATE_INDICES_REDUCED[net_for_inference_info.outputs[output_idx]]] = net_outputs_denormalized[:, :, 
                                                                                               output_idx]

    # Augment
    if 'angle' not in net_for_inference_info.outputs:
        output_array[1:,:, STATE_INDICES_REDUCED['angle']] = \
            np.arctan2(
                output_array[1:,:, STATE_INDICES_REDUCED['angle_sin']],
                output_array[1:,:, STATE_INDICES_REDUCED['angle_cos']])
    if 'angle_sin' not in net_for_inference_info.outputs:
        output_array[1:,:, STATE_INDICES_REDUCED['angle_sin']] =\
            np.sin(output_array[1:,:, STATE_INDICES_REDUCED['angle']])
    if 'angle_cos' not in net_for_inference_info.outputs:
        output_array[1:,:, STATE_INDICES_REDUCED['angle_cos']] =\
            np.sin(output_array[1:,:, STATE_INDICES_REDUCED['angle']])

    # time_axis is a time axis for ground truth
    return output_array.transpose((1,0,2))
