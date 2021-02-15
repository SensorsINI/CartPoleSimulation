"""
This file contains methods related to experiment history: saving, loading, plotting
"""

import matplotlib.pyplot as plt
import numpy as np

# to keep the loaded data
import pandas as pd
# Import module to interact with OS
import os
# Import module to save history of the simulation as csv file
import csv
# Import module to get a current time and date used to name the files containing the history of simulations
from datetime import datetime
# To detect the latest csv file
import glob


# This method saves the dictionary keeping the history of simulation to a .csv file
def save_history_csv(self, csv_name=None, mode='init'):

    if mode=='init':

        # Make folder to save data (if not yet existing)
        try:
            os.makedirs('./data')
        except FileExistsError:
            pass

        # Set path where to save the data
        if csv_name is None or csv_name == '':
            self.csv_filepath = './data/' + 'CP_' + self.controller_name + str(
                datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')) + '.csv'
        else:
            self.csv_filepath = './data/' + csv_name
            if csv_name[-4:] != '.csv':
                self.csv_filepath += '.csv'

            # If such file exists, append index to the end (do not overwrite)
            net_index = 1
            logpath_new = self.csv_filepath
            while True:
                if os.path.isfile(logpath_new):
                    logpath_new = self.csv_filepath[:-4]
                else:
                    self.csv_filepath = logpath_new
                    break
                logpath_new = logpath_new + '-' + str(net_index) + '.csv'
                net_index += 1

        # Write the .csv file
        with open(self.csv_filepath, "a") as outfile:
            writer = csv.writer(outfile)

            writer.writerow(['# ' + 'This is CartPole experiment from {} at time {}'
                            .format(datetime.now().strftime('%d.%m.%Y'), datetime.now().strftime('%H:%M:%S'))])
            if iter:
                writer.writerow(['# Number of data points: {}'.format(len(self.dict_history['time']))])
            else:
                writer.writerow(['# Number of data points: data saved online'])
            writer.writerow(['# Controller: {}'.format(self.controller_name)])

            writer.writerow(['#'])
            writer.writerow(['# Parameters:'])
            for k in self.p.__dict__:
                writer.writerow(['# ' + k + ': ' + str(getattr(self.p, k))])
            writer.writerow(['#'])

            writer.writerow(['# Data:'])
            writer.writerow(self.dict_history.keys())

    elif mode == 'save online':

        # Save this dict
        with open(self.csv_filepath, "a") as outfile:
            writer = csv.writer(outfile)
            writer.writerows(zip(*self.dict_history.values()))
        self.save_now = False

    elif mode == 'save offline':
        # Round data to a set precision
        DF_history = pd.DataFrame.from_dict(self.dict_history).round(self.rounding_decimals)
        DF_history.to_csv(self.csv_filepath, index=False, header=False, mode='a') # Mode (a)ppend


# load csv file with experiment recording (e.g. for replay)
def load_history_csv(self, csv_name=None):
    # Set path where to save the data
    if csv_name is None or csv_name == '':
        # get the latest file
        try:
            list_of_files = glob.glob('./data/' + '/*.csv')
            file_path = max(list_of_files, key=os.path.getctime)
        except FileNotFoundError:
            print('Cannot load: No experiment recording found in data folder ' + './data/')
            return False
    else:
        filename = csv_name
        if csv_name[-4:] != '.csv':
            filename += '.csv'

        # check if file found in DATA_FOLDER_NAME or at local starting point
        if not os.path.isfile(filename):
            file_path = os.path.join('data', filename)
            if not os.path.isfile(file_path):
                print(
                    'Cannot load: There is no experiment recording file with name {} at local folder or in {}'.format(
                        filename, './data/'))
                return False

    # Get race recording
    print('Loading file {}'.format(file_path))
    try:
        data: pd.DataFrame = pd.read_csv(file_path, comment='#')  # skip comment lines starting with #
    except Exception as e:
        print('Cannot load: Caught {} trying to read CSV file {}'.format(e, file_path))
        return False

    return data



# Method printing the parameters of the CartPole over time after an experiment
def summary_plots(self):
    fig, axs = plt.subplots(4, 1, figsize=(18, 14), sharex=True)  # share x axis so zoom zooms all plots

    # Plot angle error
    axs[0].set_ylabel("Angle (deg)", fontsize=18)
    axs[0].plot(np.array(self.dict_history['time']), np.array(self.dict_history['s.angle']) * 180.0 / np.pi,
                'b', markersize=12, label='Ground Truth')
    axs[0].tick_params(axis='both', which='major', labelsize=16)

    # Plot position
    axs[1].set_ylabel("position (m)", fontsize=18)
    axs[1].plot(self.dict_history['time'], self.dict_history['s.position'], 'g', markersize=12,
                label='Ground Truth')
    axs[1].tick_params(axis='both', which='major', labelsize=16)

    # Plot motor input command
    axs[2].set_ylabel("motor (N)", fontsize=18)
    axs[2].plot(self.dict_history['time'], self.dict_history['u'], 'r', markersize=12,
                label='motor')
    axs[2].tick_params(axis='both', which='major', labelsize=16)

    # Plot target position
    axs[3].set_ylabel("position target (m)", fontsize=18)
    axs[3].plot(self.dict_history['time'], self.dict_history['target_position'], 'k')
    axs[3].tick_params(axis='both', which='major', labelsize=16)

    axs[3].set_xlabel('Time (s)', fontsize=18)

    fig.align_ylabels()

    plt.show()

    return fig, axs