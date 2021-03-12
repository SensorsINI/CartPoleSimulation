
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

from Modeling.load_and_normalize import denormalize_df, load_data, load_normalization_info, normalize_df, denormalize_numpy_array
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

def open_loop_prediction_experiment(net_for_inference,
                                    net_for_inference_info,
                                    dataset=None,
                                    normalization_info=None,
                                    experiment_length=None
                                    ):

    # How far NN should predict
    max_horizon = 10

    # Reset dataset to change experiment length
    dataset.reset_exp_len(experiment_length+max_horizon)

    # Get features, target, and time axis
    # Format the experiment data
    features, targets, time_axis = dataset.get_experiment(1)  # Put number in brackets to get the same idx at every run
    time_axis = time_axis[:-1]

    # Make a prediction
    net_outputs = np.zeros(shape=(max_horizon, targets.shape[0], targets.shape[1]))
    normalized_net_output = np.zeros(shape=(max_horizon, targets.shape[0], targets.shape[1]))

    print('Calculating predictions...')
    for timestep in trange(features.shape[0]-max_horizon):

        # Make prediction based on true data
        net_input = copy.deepcopy(features[np.newaxis, np.newaxis, timestep, :])
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
            net_input = copy.deepcopy(features[np.newaxis, np.newaxis, timestep + i, :])
            net_input[..., 1:] = copy.deepcopy(normalized_net_output[i-1, timestep, :])
            normalized_net_output[i, timestep, :] = \
                np.squeeze(net_for_inference.predict_on_batch(net_input))


        #  Reset state to where it was after the first step
        net_for_inference.reset_states()
        load_internal_states(net_for_inference, states)


    features_denormalized = denormalize_numpy_array(features, net_for_inference_info.inputs, normalization_info)
    targets_denormalized = denormalize_numpy_array(targets, net_for_inference_info.outputs, normalization_info)
    net_outputs_denormalized = denormalize_numpy_array(normalized_net_output, net_for_inference_info.outputs, normalization_info)

    # Data tailored for plotting
    ground_truth = features_denormalized
    net_outputs = net_outputs_denormalized

    # time_axis is a time axis for ground truth
    return ground_truth[:-max_horizon, :], net_outputs[:, :-max_horizon, :], time_axis[:-max_horizon]

from PyQt5.QtWidgets import QApplication

def run_test_gui(inputs, outputs, ground_truth, net_outputs, time_axis):
    # Creat an instance of PyQt5 application
    # Every PyQt5 application has to contain this line
    app = QApplication(sys.argv)
    # Create an instance of the GUI window.
    window = MainWindow(inputs, outputs, ground_truth, net_outputs, time_axis)
    window.show()
    # Next line hands the control over to Python GUI
    app.exec_()

# Class implementing the main window of CartPole GUI
class MainWindow(QMainWindow):

    def __init__(self,
                 inputs, outputs, ground_truth, net_outputs, time_axis,
                 *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.inputs = inputs
        self.outputs = outputs
        self.ground_truth = ground_truth
        self.net_outputs = net_outputs
        self.time_axis = time_axis

        self.max_horizon = self.net_outputs.shape[0]
        self.horizon = self.max_horizon//2

        self.show_all = False
        self.downsample = False
        self.current_point_at_timeaxis = (self.time_axis.shape[0]-self.max_horizon)//2
        self.feature_to_display = outputs[0]

        # region - Create container for top level layout
        layout = QVBoxLayout()
        # endregion

        # region - Change geometry of the main window
        self.setGeometry(300, 300, 2500, 1000)
        # endregion

        # region - Feature selection

        # endregion

        # region - Matplotlib figures (CartPole drawing and Slider)
        # Draw Figure
        self.fig = Figure(figsize=(25, 10))  # Regulates the size of Figure in inches, before scaling to window size.
        self.canvas = FigureCanvas(self.fig)
        self.fig.Ax = self.canvas.figure.add_subplot(111)
        self.redraw_canvas()

        # Attach figure to the layout
        lf = QVBoxLayout()
        lf.addWidget(self.canvas)
        layout.addLayout(lf)

        # endregion

        l_sl = QVBoxLayout()

        # region - Slider position
        l_sl.addWidget(QLabel('"Current" point in time:'))
        self.sl_p = QSlider(Qt.Horizontal)
        self.sl_p.setMinimum(0)
        self.sl_p.setMaximum(self.time_axis.shape[0]-self.max_horizon)
        self.sl_p.setValue((self.time_axis.shape[0]-self.max_horizon)//2)
        self.sl_p.setTickPosition(QSlider.TicksBelow)
        # self.sl_p.setTickInterval(5)

        l_sl.addWidget(self.sl_p)
        self.sl_p.valueChanged.connect(self.slider_position_f)
        # endregion

        # region - Slider horizon
        l_sl.addWidget(QLabel('Prediction horizon:'))
        self.sl_h = QSlider(Qt.Horizontal)
        self.sl_h.setMinimum(0)
        self.sl_h.setMaximum(self.max_horizon)
        self.sl_h.setValue(self.max_horizon//2)
        self.sl_h.setTickPosition(QSlider.TicksBelow)
        # self.sl_h.setTickInterval(5)
        # endregion

        l_sl.addWidget(self.sl_h)
        self.sl_h.valueChanged.connect(self.slider_horizon_f)

        layout.addLayout(l_sl)


        # region - Make strip of layout for checkboxes and compobox
        l_cb = QHBoxLayout()

        # region -- Checkbox: Show all
        self.cb_show_all = QCheckBox('Show all', self)
        if self.show_all:
            self.cb_show_all.toggle()
        self.cb_show_all.toggled.connect(self.cb_show_all_f)
        l_cb.addWidget(self.cb_show_all)
        # endregion

        # region -- Checkbox: Save/don't save experiment recording
        self.cb_downsample = QCheckBox('Downsample predictions (X2)', self)
        if self.downsample:
            self.cb_downsample.toggle()
        self.cb_downsample.toggled.connect(self.cb_downsample_f)
        l_cb.addWidget(self.cb_downsample)
        # endregion

        l_cb.addStretch(1)

        # region -- Combobox: Select feature to plot
        l_cb.addWidget(QLabel('Feature to plot:'))
        self.cb_select_feature = QComboBox()
        self.cb_select_feature.addItems(outputs)
        self.cb_select_feature.currentIndexChanged.connect(self.cb_select_feature_f)
        self.cb_select_feature.setCurrentText(outputs[0])
        l_cb.addWidget(self.cb_select_feature)

        # region - Add checkboxes to layout
        layout.addLayout(l_cb)
        # endregion

        # endregion

        # region - QUIT button
        bq = QPushButton("QUIT")
        bq.pressed.connect(self.quit_application)
        lb = QVBoxLayout()  # Layout for buttons
        lb.addWidget(bq)
        layout.addLayout(lb)
        # endregion

        # region - Create an instance of a GUI window
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.show()
        self.setWindowTitle('Testing TF model')

        # endregion


    def slider_position_f(self, value):
        self.current_point_at_timeaxis = int(value)

        self.redraw_canvas()

    def slider_horizon_f(self, value):
        self.horizon = int(value)

        self.redraw_canvas()

    def cb_show_all_f(self, state):
        if state:
            self.show_all = True
        else:
            self.show_all = False

        self.redraw_canvas()

    def cb_downsample_f(self, state):
        if state:
            self.downsample = True
        else:
            self.downsample = False

        self.redraw_canvas()

    def cb_select_feature_f(self):
        self.feature_to_display = self.cb_select_feature.currentText()
        self.redraw_canvas()

    # The actions which has to be taken to properly terminate the application
    # The method is evoked after QUIT button is pressed
    # TODO: Can we connect it somehow also the the default cross closing the application?
    def quit_application(self):
        # Closes the GUI window
        self.close()
        # The standard command
        # It seems however not to be working by its own
        # I don't know how it works
        QApplication.quit()


    def redraw_canvas(self):

        self.fig.Ax.clear()

        brunton_widget(self.inputs, self.outputs, self.ground_truth, self.net_outputs, self.time_axis,
                       axs=self.fig.Ax,
                       current_point_at_timeaxis=self.current_point_at_timeaxis,
                       feature_to_display=self.feature_to_display,
                       max_horizon=self.max_horizon,
                       horizon=self.horizon,
                       show_all=self.show_all,
                       downsample=self.downsample)

        self.canvas.draw()



def brunton_widget(inputs, outputs, ground_truth, net_outputs, time_axis, axs=None,
                   current_point_at_timeaxis=None,
                   feature_to_display=None,
                   max_horizon=10, horizon=None,
                   show_all=True,
                   downsample=False):

    # Start at should be done bu widget (slider)
    if current_point_at_timeaxis is None:
        current_point_at_timeaxis = ground_truth.shape[0]//2

    if feature_to_display is None:
        feature_to_display = 's.angle.cos'

    if horizon is None:
        horizon = max_horizon

    feature_idx = inputs.index(feature_to_display)
    target_idx = outputs.index(feature_to_display)

    # Brunton Plot
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(18, 10), sharex=True)

    axs.plot(time_axis, ground_truth[:, feature_idx], 'k:', markersize=12, label='Ground Truth')
    y_lim = axs.get_ylim()
    prediction_distance = []
    axs.set_ylabel(feature_to_display, fontsize=18)
    axs.set_xlabel('Time [s]', fontsize=18)
    for i in range(horizon):

        if not show_all:
            axs.plot(time_axis[current_point_at_timeaxis], ground_truth[current_point_at_timeaxis, feature_idx],
                     'g.', markersize=16, label='Start')
            prediction_distance.append(net_outputs[i, current_point_at_timeaxis, target_idx])
            if downsample:
                if (i % 2) == 0:
                    continue
            axs.plot(time_axis[current_point_at_timeaxis+i+1], prediction_distance[i],
                        c=cmap(float(i)/max_horizon),
                        marker='.')

        else:
            prediction_distance.append(net_outputs[i, :-(i+1), target_idx])
            if downsample:
                if (i % 2) == 0:
                    continue
            axs.plot(time_axis[i+1:], prediction_distance[i],
                        c=cmap(float(i)/max_horizon),
                        marker='.', linestyle = '')

    axs.set_ylim(y_lim)

    plt.show()