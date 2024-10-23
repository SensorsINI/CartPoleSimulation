"""
This module provides a class to manage data during the simulation and physical experiment.
It is used to store data and save to csv.
"""

import copy
import csv
import threading

import numpy as np


class DataManager:
    def __init__(self, create_csv_file):

        self.create_csv_file = create_csv_file

        self._store_data = False  # Weather to store the history data
        self._save_mode = 'online'  # disabled (no saving to csv), "online" (save at each step), "offline"(save in the end)
        self.history = {}

        self.csv_filepath = None
        self.csv_file = None
        self.csv_writer = None

        self.rounding_decimals = np.inf

        self.start_csv_recording_thread = None
        self.finish_experiment_thread = None

        self.recording_counter = 0
        self.recording_counter_target = np.inf

        self.starting_recording = False
        self.recording_running = False

    # The getter-setter below make sure that the impossible combination
    # - store_data=False and save_mode='offline' is not possible to select
    @property
    def store_data(self):
        return self._store_data

    @store_data.setter
    def store_data(self, value):
        if not isinstance(value, bool):
            raise ValueError("store_data must be a boolean value")
        if not value and self._save_mode == 'offline':
            self._save_mode = 'online'
        self._store_data = value
        self.reset_history()  # it is needed in case self.store changes between initialize_history and update_history

    @property
    def save_mode(self):
        return self._save_mode

    @save_mode.setter
    def save_mode(self, value):
        if value not in ['disabled', 'online', 'offline']:
            raise ValueError("save_mode must be 'disabled', 'online', or 'offline'")
        if value == 'offline':
            self._store_data = True
        self._save_mode = value
        self.reset_history() # it is needed in case self.store changes between initialize_history and update_history

    def initialize_history(self, keys):
        """Initialize history with given keys without data."""
        if not self.history:
            for key in keys:
                self.history[key] = [] if self.store_data else None

    def update_history(self, list_data_dicts):
        # If history is empty, initialize it with the data
        if self.history == {}:
            for d in list_data_dicts:
                for key, value in d.items():
                    self.history[key] = [value] if self.store_data else value
        else:
            # Check if keys are the same
            input_keys = {key for d in list_data_dicts for key in d.keys()}
            history_keys = set(self.history.keys())

            if input_keys != history_keys:
                raise ValueError("Keys of input dictionaries do not match the keys in history")

            # If keys match, update the history
            for d in list_data_dicts:
                for key, value in d.items():
                    if self.store_data:
                        self.history[key].append(value)
                    else:
                        self.history[key] = value

    def reset_history(self):
        self.history = {}

    def start_csv_recording(self, csv_name, keys, title, header, path_to_experiment_recordings, mode='online',
                            wait_till_complete=True, recording_length=np.inf):

        self.save_mode = mode
        self.recording_counter_target = recording_length

        def start_csv_recording_thread_function():
            self.initialize_history(keys)
            # Create the csv file
            if self.save_mode != 'disabled':
                self.csv_filepath = self.create_csv_file(csv_name, keys, path_to_experiment_recordings=path_to_experiment_recordings,
                                                    title=title, header=header)
            if self.save_mode == 'online':
                self.csv_file = open(self.csv_filepath, 'a', newline='')
                self.csv_writer = csv.writer(self.csv_file, delimiter=',')

            self.starting_recording = False
            self.recording_running = True

            print(f"Recording started to {self.csv_filepath}")

        self.start_csv_recording_thread = threading.Thread(target=start_csv_recording_thread_function)
        self.starting_recording = True
        self.start_csv_recording_thread.start()
        if wait_till_complete:
            self.finish_experiment_thread.join()

    def save_csv_online(self):
        self.csv_writer.writerow(self.history.values())

    def finish_experiment(self, return_history=False, wait_till_complete=True):
        def finish_experiment_thread_function():
            if self.save_mode == 'offline':
                with open(self.csv_filepath, "a", newline='') as outfile:
                    writer = csv.writer(outfile)
                    if self.rounding_decimals == np.inf:
                        pass
                    else:
                        dict_history = {key: np.around(value, self.rounding_decimals)
                                        for key, value in self.history.items()}
                    writer.writerows(zip(*dict_history.values()))
            elif self.save_mode == 'online':
                self.csv_file.close()

            print(f"Recording finished to {self.csv_filepath}")

            self.csv_filepath = None
            self.csv_file = None
            self.csv_writer = None

        self.finish_experiment_thread = threading.Thread(target=finish_experiment_thread_function)
        self.finish_experiment_thread.start()
        if wait_till_complete:
            self.finish_experiment_thread.join()

        self.recording_running = False

        if return_history:
            history = copy.deepcopy(self.history)  # Be careful, it might take a while causing physical cartpole to fall
        else:
            history = None

        self.reset_history()

        self.recording_counter_target = np.inf
        self.recording_counter = 0
        self.recording_running = False

        return history

    def step(self, list_data_dicts):
        if self.recording_running:
            self.recording_counter += 1
            self.update_history(list_data_dicts)
            if self.save_mode == 'online':
                self.save_csv_online()

        if self.recording_counter == self.recording_counter_target:
            self.finish_experiment()
