
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
    normalize_df, denormalize_numpy_array
from Modeling.TF.TF_Functions.Network import get_internal_states, load_internal_states

# Import functions from PyQt5 module (creating GUI)
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QApplication, QVBoxLayout, \
    QHBoxLayout, QLabel, QPushButton, QWidget, QCheckBox, \
    QLineEdit, QMessageBox, QComboBox, QButtonGroup, QSlider
from PyQt5.QtCore import QThreadPool, QTimer, Qt
# The main drawing functionalities are implemented in CartPole Class
# Some more functions needed for interaction of matplotlib with PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# This import mus go before pyplot so also before our scripts
from matplotlib import use, get_backend
# Use Agg if not in scientific mode of Pycharm
from .Initialization import get_net_and_norm_info

if get_backend() != 'module://backend_interagg':
    use('Agg')

# endregion

# region Other imports for GUI

import sys

# endregion


from .Dataset import Dataset

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

def get_predictions_TF(net_for_inference,
                       net_for_inference_info,
                       dataset=None,
                       normalization_info=None,
                       experiment_length=None,
                       max_horizon=None
                       ):

    # Get features, target, and time axis
    # Format the experiment data
    features = dataset[net_for_inference_info.inputs]
    time_axis = dataset['time'].to_numpy()[:experiment_length]

    features_normalized = normalize_df(features, normalization_info).to_numpy()
    # Make a prediction
    normalized_net_output = np.zeros(shape=(max_horizon, experiment_length, len(net_for_inference_info.outputs)))

    print('Calculating predictions...')
    for timestep in trange(experiment_length):

        # Make prediction based on true data
        net_input = copy.deepcopy(features_normalized[np.newaxis, np.newaxis, timestep, :])
        # t2 = timeit.default_timer()
        normalized_net_output[0, timestep, :] = np.squeeze(net_for_inference.predict_on_batch(net_input))
        # t3 = timeit.default_timer()
        # print('t3 evaluation {} ms'.format((t3 - t2) * 1000.0))

        # Save internal state
        states = get_internal_states(net_for_inference)

        # Progress max_horison-1 steps in closed loop
        # save the data for every step in a third dimension of an array
        for i in range(1, max_horizon):
            # We assume control input is the first variable
            # All other variables are in closed loop
            net_input[..., 1:] = copy.deepcopy(normalized_net_output[i - 1, timestep, :])
            net_input[..., 0] = copy.deepcopy(features_normalized[np.newaxis, np.newaxis, timestep + i, 0])
            normalized_net_output[i, timestep, :] = \
                np.squeeze(net_for_inference.predict_on_batch(net_input))


        #  Reset state to where it was after the first step
        net_for_inference.reset_states()
        load_internal_states(net_for_inference, states)

    net_outputs_denormalized = denormalize_numpy_array(normalized_net_output, net_for_inference_info.outputs, normalization_info)

    # Data tailored for plotting
    ground_truth = features.to_numpy()[:experiment_length, :]
    net_outputs = net_outputs_denormalized

    # time_axis is a time axis for ground truth
    return ground_truth, net_outputs, time_axis


def get_data_for_gui_TF(a):
    # Create a copy of the network suitable for inference (stateful and with sequence length one)
    net_for_inference, net_for_inference_info, normalization_info = \
        get_net_and_norm_info(a, time_series_length=1,
                              batch_size=1, stateful=True)

    # region In either case testing is done on a data collected offline
    paths_to_datafiles_test = get_paths_to_datafiles(a.test_files)
    test_dfs = load_data(paths_to_datafiles_test)
    dataset = test_dfs[0].iloc[a.test_start_idx:a.test_start_idx+a.test_len+a.test_max_horizon, :]

    ground_truth, net_outputs, time_axis = \
        get_predictions_TF(net_for_inference, net_for_inference_info,
                           dataset, normalization_info,
                           experiment_length=a.test_len,
                           max_horizon=a.test_max_horizon)

    return net_for_inference_info.inputs, net_for_inference_info.outputs, net_for_inference_info.net_full_name,\
           ground_truth, net_outputs, time_axis