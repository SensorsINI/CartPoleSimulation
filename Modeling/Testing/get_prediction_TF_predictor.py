
import copy
import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors

from tqdm import trange

import copy
from time import sleep

from Modeling.load_and_normalize import denormalize_df, get_paths_to_datafiles, load_data, load_normalization_info, \
    normalize_df, denormalize_numpy_array, normalize_numpy_array

from CartPole.state_utilities import STATE_INDICES_REDUCED, STATE_VARIABLES, cartpole_state_varnames_to_indices

from Predictores.predictor_autoregressive_tf import predictor_autoregressive_tf

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

    states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]

    Q = dataset['Q'].to_numpy()
    Q_array = [Q[i:-a.test_max_horizon+i] for i in range(a.test_max_horizon)]
    Q_array = np.vstack(Q_array)

    a = copy.deepcopy(a)
    a.net_name = net_name
    predictor = predictor_autoregressive_tf(a, horizon=a.test_max_horizon)

    # All at once
    predictor.setup(initial_state=states_0, prediction_denorm=True)
    output_array = predictor.predict(Q_array)  # Should be shape=(a.test_max_horizon, a.test_len, len(outputs))

    # # Iteratively (to test internal state update)
    # for timestep in trange(a.test_len):
    #     Q_current_timestep = Q_array[:, timestep]
    #     s_current_timestep = states_0[timestep]
    #     predictor.setup(initial_state=s_current_timestep, prediction_denorm=True)
    #     output_array[timestep,:,:] = predictor.predict(Q_current_timestep)
    #     predictor.update_internal_state(Q_current_timestep[0])

    output_array = output_array[..., list(cartpole_state_varnames_to_indices(a.features))+[-1]]

    # time_axis is a time axis for ground truth
    return output_array
