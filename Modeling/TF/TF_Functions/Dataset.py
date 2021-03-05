import tensorflow.keras as keras

import numpy as np
import pandas as pd

import copy


class Dataset(keras.utils.Sequence):
    def __init__(self,
                 dfs,
                 args,
                 inputs_list=None, outputs_list=None,
                 batch_size=None,
                 time_axes=None,
                 exp_len=None,
                 warm_up_len=None,
                 shuffle=True):
        'Initialization - divide data in features and labels'

        if inputs_list is None and args.inputs_list is not None:
            inputs_list = args.inputs_list
        if outputs_list is None and args.outputs_list is not None:
            outputs_list = args.outputs_list

        self.data = []
        self.labels = []

        for df in dfs:
            # Get Raw Data
            features = copy.deepcopy(df)
            targets = copy.deepcopy(df)

            features.drop(features.tail(1).index, inplace=True)  # Drop last row
            targets.drop(targets.head(1).index, inplace=True)
            features = features.reset_index()  # Reset index
            targets = targets.reset_index()

            features = features[inputs_list]
            targets = targets[outputs_list]

            self.data.append(features)
            self.labels.append(targets)

        self.args = args

        self.exp_len = None
        self.warm_up_len = self.args.wash_out_len
        self.df_lengths = []
        self.df_lengths_cs = []
        self.number_of_samples = 0

        self.time_axes = time_axes

        # NEW PART FOR TENSORFLOW:
        # __get_item__ must return a batch
        self.batch_size = 1
        self.number_of_batches = 1
        self.shuffle = shuffle
        self.indexes = []

        self.reset_exp_len(exp_len=exp_len)
        self.reset_batch_size(batch_size=batch_size)

        # Here we imnplement a trial to change the target position and current positions
        # for ease of implementation we assume that both input and output has s.position, only input has target position
        self.idx_pos_in = inputs_list.index('s.position')
        self.idx_pos_out = outputs_list.index('s.position')
        if 'target_position' in inputs_list:
            self.idx_target_pos_in = inputs_list.index('target_position')

    def reset_exp_len(self, exp_len=None):
        """
        This method should be used if the user wants to change the exp_len without creating new Dataset
        Please remember that one can reset it again to come back to old configuration (from ParseArgs)
        :param exp_len: Gives new user defined exp_len. Call empty to come back to default.
        """
        if exp_len is None:
            self.exp_len = self.args.exp_len  # Sequence length
        else:
            self.exp_len = exp_len

        self.df_lengths = []
        self.df_lengths_cs = []
        if type(self.data) == list:
            for data_set in self.data:
                self.df_lengths.append(data_set.shape[0] - self.exp_len)
                if not self.df_lengths_cs:
                    self.df_lengths_cs.append(self.df_lengths[0])
                else:
                    self.df_lengths_cs.append(self.df_lengths_cs[-1] + self.df_lengths[-1])
            self.number_of_samples = self.df_lengths_cs[-1]

        else:
            self.number_of_samples = self.data.shape[0] - self.exp_len

        self.number_of_batches = int(np.ceil(self.number_of_samples / float(self.batch_size)))

        self.indexes = np.arange(self.number_of_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def reset_batch_size(self, batch_size=None):

        if batch_size is None:
            self.batch_size = self.args.batch_size
        else:
            self.batch_size = batch_size

        self.number_of_batches = int(np.ceil(self.number_of_samples / float(self.batch_size)))

        self.indexes = np.arange(self.number_of_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # In TF it must return the number of batches
        return self.number_of_batches

    def get_series(self, idx, get_time_axis=False, targets_type='first_after_warm_up'):
        """
        Requires the self.data to be a list of pandas dataframes
        """
        # Find index of the dataset in self.data and index of the starting point in this dataset
        idx_data_set = next(i for i, v in enumerate(self.df_lengths_cs) if v > idx)
        if idx_data_set == 0:
            pass
        else:
            idx -= self.df_lengths_cs[idx_data_set - 1]

        # Get data
        features = None
        targets = None

        if targets_type == 'first_after_warm_up':
            features = self.data[idx_data_set].to_numpy()[idx:idx + self.warm_up_len, :]
            # After feeding the whole sequence we just compare the final output of the RNN with the state following afterwards
            targets = self.labels[idx_data_set].to_numpy()[idx + self.warm_up_len-1]
        elif targets_type == 'all':
            features = self.data[idx_data_set].to_numpy()[idx:idx + self.exp_len, :]
            # Every point in features has its target value corresponding to the next time step:
            targets = self.labels[idx_data_set].to_numpy()[idx:idx + self.exp_len]
        elif targets_type == 'all after warm-up':
            features = self.data[idx_data_set].to_numpy()[idx:idx + self.exp_len, :]
            # Every point in features has its target value corresponding to the next time step:
            targets = self.labels[idx_data_set].to_numpy()[idx+self.warm_up_len:idx + self.exp_len]
        else:
            raise('Non-existent target_type')

        # mix_position = True
        # if mix_position and targets_type == 'all after warm-up':
        #     random_pos = np.random.uniform(-40.0, 40.0)
        #     features[:,self.idx_pos_in] = features[:,self.idx_pos_in]+random_pos
        #     features[:,self.idx_target_pos_in] = features[:,self.idx_target_pos_in]+random_pos
        #     targets[self.idx_pos_out] = targets[self.idx_pos_out]+random_pos

        # If get_time_axis try to obtain a vector of time data for the chosen sample
        if get_time_axis:
            try:
                # As targets and features are shifted by one timestep we have to make time_axis accordingly longer to cover both
                time_axis = self.time_axes[idx_data_set].to_numpy()[idx:idx + self.exp_len + 1]
            except IndexError:
                time_axis = []

        # Return results
        if get_time_axis:
            return features, targets, time_axis
        else:
            return features, targets

    def __getitem__(self, idx_batch):

        features_batch = []
        targets_batch = []
        sample_idx = self.indexes[self.batch_size * idx_batch: self.batch_size * (idx_batch + 1)]
        for i in sample_idx:
            features, targets = self.get_series(i)
            features_batch.append(features)
            targets_batch.append(targets)
        features_batch = np.stack(features_batch)
        targets_batch = np.stack(targets_batch)

        return features_batch, targets_batch

    def get_experiment(self, idx=None, targets_type='all'):
        if self.time_axes is None:
            raise Exception('No time information available!')
        if idx is None:
            idx = np.random.randint(0, self.number_of_samples)
        return self.get_series(idx, get_time_axis=True, targets_type=targets_type)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)




from types import SimpleNamespace
from Predictores.predictor_ideal import predictor_ideal
class DatasetRandom(keras.utils.Sequence):
    def __init__(self,
                 args=None,
                 inputs_list=None, outputs_list=None,
                 number_of_batches=1000):
        'Initialization - divide data in features and labels'

        if inputs_list is None and args.inputs_list is not None:
            inputs_list = args.inputs_list
        if outputs_list is None and args.outputs_list is not None:
            outputs_list = args.outputs_list

        self.inputs_list = inputs_list
        self.outputs_list = outputs_list

        self.data = []
        self.labels = []

        self.args = args

        self.exp_len = self.args.exp_len
        self.warm_up_len = self.args.warm_up_len

        self.number_of_batches = number_of_batches
        self.batch_size = self.args.batch_size



    def __len__(self):
        # In TF it must return the number of batches
        return self.number_of_batches

    def get_series(self, idx, targets_type='first_after_warm_up'):
        """
        Requires the self.data to be a list of pandas dataframes
        """
        s = SimpleNamespace()

        s.position = np.random.uniform(low=-50.0,
                                              high=50.0)

        s.positionD = np.random.uniform(low=-12.0,
                                               high=12.0)

        s.angle = np.random.uniform(low=-np.pi,
                                           high= np.pi)

        s.angleD = np.random.uniform(low=-3.0 * np.pi,
                                            high=3.0 * np.pi)

        initial_state = pd.DataFrame(0, index=np.arange(1),
                                          columns=['s.angle.cos', 's.angle.sin', 's.angleD', 's.position',
                                                   's.positionD'])

        initial_state['s.angle.cos'] = [np.cos(s.angle)]
        initial_state['s.angle.sin'] = [np.sin(s.angle)]
        initial_state['s.angleD'] = [s.angleD]
        initial_state['s.position'] = [s.position]
        initial_state['s.positionD'] = [s.positionD]

        Predictor = predictor_ideal((self.exp_len+1) * 5, 0.02) # This results in exp_len+2 timesteps
        Predictor.setup(initial_state=initial_state, prediction_denorm=False)

        Q = np.random.uniform(low=-1, high=1, size=self.exp_len+1)
        Q_hat = np.repeat(Q, 5)
        df = Predictor.predict(Q_hat)
        df = df[::5]
        df.drop(df.tail(1).index, inplace=True)  # Drop last row

        df['Q'] = Q

        features = copy.deepcopy(df)
        targets = copy.deepcopy(df)

        features.drop(features.tail(1).index, inplace=True)  # Drop last row
        targets.drop(targets.head(1).index, inplace=True)
        features.reset_index(inplace=True)  # Reset index
        targets.reset_index(inplace=True)

        data = features[self.inputs_list]
        labels = targets[self.outputs_list]

        if targets_type == 'first_after_warm_up':
            features = data.to_numpy()[0:self.warm_up_len, :]
            # After feeding the whole sequence we just compare the final output of the RNN with the state following afterwards
            targets = labels.to_numpy()[self.warm_up_len-1]
        elif targets_type == 'all':
            features = data.to_numpy()[0:self.exp_len, :]
            # Every point in features has its target value corresponding to the next time step:
            targets = labels.to_numpy()[0:self.exp_len]
        elif targets_type == 'all after warm-up':
            features = data.to_numpy()[:self.exp_len, :]
            # Every point in features has its target value corresponding to the next time step:
            targets = labels.to_numpy()[self.warm_up_len:self.exp_len]
        else:
            raise('Non-existent target_type')


        return features, targets

    def __getitem__(self, idx_batch):

        features_batch = []
        targets_batch = []
        for i in range(self.batch_size):
            features, targets = self.get_series(i)
            features_batch.append(features)
            targets_batch.append(targets)
        features_batch = np.stack(features_batch)
        targets_batch = np.stack(targets_batch)

        return features_batch, targets_batch

