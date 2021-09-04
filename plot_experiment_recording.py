# Uncomment if you want to get interactive plots for MPPI in Pycharm on MacOS
# On other OS you have to chose a different interactive backend.
from matplotlib import use
# # use('TkAgg')
use('macOSX')
from cycler import cycler
import numpy as np

from CartPole import CartPole

import matplotlib.pyplot as plt

# csv_name = ['Experiment-GT-Smooth.csv', 'Experiment-8-Eq-Frozen-Smooth.csv', 'Experiment-9-Dense-Smooth.csv']
csv_name = ['Experiment.csv']
final_index = -1
dict_datasets = {}

for experiment_name in csv_name:

    CartPoleInstance = CartPole()

    # Load experiment history
    history_pd, filepath = CartPoleInstance.load_history_csv(csv_name=experiment_name)

    # Augment the experiment history with simulation time step size
    dt = []
    row_iterator = history_pd.iterrows()
    _, last = next(row_iterator)  # take first item from row_iterator
    for i, row in row_iterator:
        dt.append(row['time'] - last['time'])
        last = row
    dt.append(dt[-1])
    history_pd['dt'] = np.array(dt)

    CartPoleInstance.dict_history = history_pd.to_dict(orient='list')

    fontsize_labels = 12
    fontsize_ticks = 12
    title = 'LQR-Observer'

    CartPoleInstance.summary_plots(title=title)

    fig, axs = plt.subplots(4, 1, figsize=(16, 9), sharex=True)  # share x axis so zoom zooms all plots
    fig.suptitle(title, fontsize=16)

    # Plot angle error
    axs[0].set_ylabel("Angle (deg)", fontsize=fontsize_labels)
    axs[0].plot(np.array(CartPoleInstance.dict_history['time']), np.array(CartPoleInstance.dict_history['angle']) * 180.0 / np.pi,
                'black', markersize=12, label='Ground Truth')
    axs[0].plot(np.array(CartPoleInstance.dict_history['time']), np.array(CartPoleInstance.dict_history['angle_measurement']) * 180.0 / np.pi,
                'green', markersize=12, label='Measurement')
    axs[0].plot(np.array(CartPoleInstance.dict_history['time']), np.array(CartPoleInstance.dict_history['angle_estimate']) * 180.0 / np.pi,
                'red', markersize=12, label='Estimate')
    axs[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    axs[0].legend(fontsize=fontsize_labels)

    # Plot position
    axs[1].set_ylabel("position (m)", fontsize=fontsize_labels)
    axs[1].plot(CartPoleInstance.dict_history['time'], CartPoleInstance.dict_history['position'], 'black', markersize=12,
                label='Ground Truth')
    axs[1].plot(CartPoleInstance.dict_history['time'], CartPoleInstance.dict_history['position_measurement'], 'green', markersize=12,
                label='Measurement')
    axs[1].plot(CartPoleInstance.dict_history['time'], CartPoleInstance.dict_history['position_estimate'], 'red', markersize=12,
                label='Estimate')
    axs[1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    axs[1].legend(fontsize=fontsize_labels)

    axs[2].set_ylabel("angleD (deg/s)", fontsize=fontsize_labels)
    axs[2].plot(CartPoleInstance.dict_history['time'], CartPoleInstance.dict_history['angleD'], 'black', markersize=12,
                label='Ground Truth')
    axs[2].plot(CartPoleInstance.dict_history['time'], CartPoleInstance.dict_history['angleD_estimate'], 'red', markersize=12,
                label='Estimate')
    axs[2].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    axs[2].legend(fontsize=fontsize_labels)

    axs[3].set_ylabel("positionD (m/s)", fontsize=fontsize_labels)
    axs[3].plot(CartPoleInstance.dict_history['time'], CartPoleInstance.dict_history['positionD'], 'black', markersize=12,
                label='Ground Truth')
    axs[3].plot(CartPoleInstance.dict_history['time'], CartPoleInstance.dict_history['positionD_estimate'], 'red', markersize=12,
                label='Estimate')
    axs[3].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    axs[3].legend(fontsize=fontsize_labels)

    axs[3].set_xlabel('Time (s)', fontsize=fontsize_labels)

    fig.align_ylabels()

    plt.show()