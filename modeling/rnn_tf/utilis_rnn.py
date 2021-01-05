

from datetime import datetime

import collections
import os

import random as rnd

import copy

from modeling.rnn_tf.utilis_rnn_specific import *

from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras

import pandas as pd

import timeit


# Set seeds everywhere required to make results reproducible
def set_seed(args):
    seed = args.seed
    rnd.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Print parameter count
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def print_parameter_count(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams
    print('::: # network all parameters: ' + str(totalParams))
    print('::: # network trainable parameters: ' + str(trainableParams))
    print('')


def load_pretrained_rnn(net, ckpt_path):
    """
    A function loading parameters (weights and biases) from a previous training to a net RNN instance
    :param net: An instance of RNN
    :param ckpt_path: path to .ckpt file storing weights and biases
    :return: No return. Modifies net in place.
    """
    print("Loading Model: ", ckpt_path)
    print('')

    net.load_weights(ckpt_path)


# Initialize weights and biases - should be only applied if no pretrained net loaded
# Seems not be automatic in current implementation
# def initialize_weights_and_biases(net):
#     print('Initialize weights and biases')
#     for name, param in net.named_parameters():
#         print('Initialize {}'.format(name))
#         if 'gru' in name:
#             if 'weight' in name:
#                 nn.init.orthogonal_(param)
#         if 'linear' in name:
#             if 'weight' in name:
#                 nn.init.orthogonal_(param)
#                 # nn.init.xavier_uniform_(param)
#         if 'bias' in name:  # all biases
#             nn.init.constant_(param, 0)
#     print('')


def create_rnn_instance(args=None, rnn_name=None, inputs_list=None, outputs_list=None, load_rnn=None, path_save=None,
                        warm_up_len=1, return_sequence=False, stateful=False, batchSize=None):
    if rnn_name is None and args is not None:
        rnn_name = args.rnn_name
    if inputs_list is None and args is not None:
        inputs_list = args.inputs_list
    if outputs_list is None and args is not None:
        outputs_list = args.outputs_list
    if load_rnn is None and args is not None:
        load_rnn = args.load_rnn
    if path_save is None and args is not None:
        path_save = args.path_save
    if warm_up_len == 1 and args is not None:
        warm_up_len = args.warm_up_len
    # if return_sequence is False and args is not None:
    #     return_sequence = args.return_sequence
    # if stateful is False and args is not None:
    #     stateful = args.stateful

    if load_rnn is not None and load_rnn != 'last':
        # 1) Find csv with this name if exists load name, inputs and outputs list
        #       if it does not exist raise error
        # 2) Create corresponding net
        # 3) Load parameters from corresponding ckpt file

        filename = load_rnn
        print('Loading a pretrained RNN with the full name: {}'.format(filename))
        print('')
        txt_filename = filename + '.txt'
        ckpt_filename = filename + '.ckpt'
        txt_path = path_save + txt_filename
        ckpt_path = path_save + ckpt_filename

        if not os.path.isfile(txt_path):
            raise ValueError(
                'The corresponding .txt file is missing (information about inputs and outputs) at the location {}'.format(
                    txt_path))
        if not os.path.isfile(ckpt_path + '.index'):
            raise ValueError(
                'The corresponding .ckpt file is missing (information about weights and biases) at the location {}'.format(
                    ckpt_path))

        f = open(txt_path, 'r')
        lines = f.readlines()
        rnn_name = lines[1].rstrip("\n")
        inputs_list = lines[7].rstrip("\n").split(sep=', ')
        outputs_list = lines[10].rstrip("\n").split(sep=', ')
        f.close()

        print('Inputs to the loaded RNN: {}'.format(', '.join(map(str, inputs_list))))
        print('Outputs from the loaded RNN: {}'.format(', '.join(map(str, outputs_list))))
        print('')

        # Construct the requested RNN
        net = myNN(rnn_name=rnn_name, inputs_list=inputs_list, outputs_list=outputs_list,
                   warm_up_len=warm_up_len, return_sequence=return_sequence,
                   stateful=stateful, batchSize=batchSize)
        net.rnn_full_name = load_rnn

        # Load the parameters
        load_pretrained_rnn(net, ckpt_path)

        ## TODO: Load Normalization
        normalization_info = load_normalization_info(path_save, filename)


    elif load_rnn == 'last':
        files_found = False
        while (not files_found):
            try:
                import glob
                list_of_files = glob.glob(path_save + '/*.txt')
                txt_path = max(list_of_files, key=os.path.getctime)
            except ValueError:
                raise ValueError('No information about any pretrained network found at {}'.format(path_save))

            f = open(txt_path, 'r')
            lines = f.readlines()
            rnn_name = lines[1].rstrip("\n")
            pre_rnn_full_name = lines[4].rstrip("\n")
            inputs_list = lines[7].rstrip("\n").split(sep=', ')
            outputs_list = lines[10].rstrip("\n").split(sep=', ')
            f.close()

            ckpt_path = path_save + pre_rnn_full_name + '.ckpt'
            if not os.path.isfile(ckpt_path + '.index'):
                print('The .ckbt file is missing (information about weights and biases) at the location {}'.format(
                    ckpt_path))
                print('I delete the corresponding .txt file and try to search again')
                print('')
                os.remove(txt_path)
            else:
                files_found = True

        print('Full name of the loaded RNN is {}'.format(pre_rnn_full_name))
        print('Inputs to the loaded RNN: {}'.format(', '.join(map(str, inputs_list))))
        print('Outputs from the loaded RNN: {}'.format(', '.join(map(str, outputs_list))))
        print('')

        # Construct the requested RNN
        net = myNN(rnn_name=rnn_name, inputs_list=inputs_list, outputs_list=outputs_list,
                   warm_up_len=warm_up_len, return_sequence=return_sequence,
                   stateful=stateful, batchSize=batchSize)
        net.rnn_full_name = pre_rnn_full_name

        # Load the parameters
        load_pretrained_rnn(net, ckpt_path)

        ## TODO: Load Normalization
        normalization_info = load_normalization_info(path_save, pre_rnn_full_name)


    else:  # args.load_rnn is None
        print('No pretrained network specified. I will train a network from scratch.')
        print('')
        # Construct the requested RNN
        net = myNN(rnn_name=rnn_name, inputs_list=inputs_list, outputs_list=outputs_list,
                   warm_up_len=warm_up_len, return_sequence=return_sequence,
                   stateful=stateful, batchSize=batchSize)
        normalization_info = None


    return net, rnn_name, inputs_list, outputs_list, normalization_info


def create_log_file(rnn_name, inputs_list, outputs_list, path_save):
    rnn_full_name = rnn_name[:4] + str(len(inputs_list)) + 'IN-' + rnn_name[4:] + '-' + str(len(outputs_list)) + 'OUT'

    net_index = 0
    while True:

        txt_path = path_save + rnn_full_name + '-' + str(net_index) + '.txt'
        if os.path.isfile(txt_path):
            pass
        else:
            rnn_full_name += '-' + str(net_index)
            f = open(txt_path, 'w')
            f.write('RNN NAME: \n' + rnn_name + '\n\n')
            f.write('RNN FULL NAME: \n' + rnn_full_name + '\n\n')
            f.write('INPUTS: \n' + ', '.join(map(str, inputs_list)) + '\n\n')
            f.write('OUTPUTS: \n' + ', '.join(map(str, outputs_list)) + '\n\n')
            f.close()
            break

        net_index += 1

    print('Full name given to the currently trained network is {}.'.format(rnn_full_name))
    print('')
    return rnn_full_name


class myNN(keras.Sequential):
    """"
    Our RNN class.
    """

    def __init__(self,
                 rnn_name,
                 inputs_list,
                 outputs_list,
                 warm_up_len,
                 return_sequence=False,
                 batchSize=None,
                 stateful=False):
        super(myNN, self).__init__()
        """Initialization of an RNN instance
        We assume that inputs may be both commands and state variables, whereas outputs are always state variables
        """

        self.rnn_name = rnn_name
        self.rnn_full_name = None
        self.warm_up_len = warm_up_len

        self.inputs_list = inputs_list
        self.outputs_list = outputs_list

        self.skip_warm_up_in_output = True

        # Get the information about network architecture from the network name
        # Split the names into "LSTM/GRU", "128H1", "64H2" etc.
        names = rnn_name.split('-')
        layers = ['H1', 'H2', 'H3', 'H4', 'H5']
        self.h_size = []  # Hidden layers sizes
        for name in names:
            for index, layer in enumerate(layers):
                if layer in name:
                    # assign the variable with name obtained from list layers.
                    self.h_size.append(int(name[:-2]))

        if not self.h_size:
            raise ValueError('You have to provide the size of at least one hidden layer in rnn name')

        self.h_number = len(self.h_size)

        if 'GRU' in names:
            self.rnn_type = 'GRU'
            self.rnn_layer = keras.layers.GRU
        elif 'LSTM' in names:
            self.rnn_type = 'LSTM'
            self.rnn_layer = keras.layers.LSTM
        else:
            self.rnn_type = 'RNN-Basic'
            self.rnn_layer = keras.layers.SimpleRNN

        # Construct network
        # Define first layer
        if self.h_number == 1:
            self.add(self.rnn_layer(
                units=self.h_size[0],
                batch_input_shape=(batchSize, warm_up_len, len(inputs_list)),
                return_sequences=return_sequence,
                stateful=stateful
            ))
        else:
            self.add(self.rnn_layer(
                units=self.h_size[0],
                batch_input_shape=(batchSize, warm_up_len, len(inputs_list)),
                return_sequences=True,
                stateful=stateful
            ))
            # Define following layers
            # The for loop will only executed if there is MORE than 2 hidden layers
            for i in range(len(self.h_size) - 2):
                self.add(self.rnn_layer(
                    units=self.h_size[i + 1],
                    return_sequences=True,
                    stateful=stateful
                ))
            # Last RNN layer
            self.add(self.rnn_layer(
                units=self.h_size[-1],
                return_sequences=return_sequence,
                stateful=stateful
            ))

        self.add(keras.layers.Dense(units=len(outputs_list), activation='tanh'))

        print('Constructed a neural network of type {}, with {} hidden layers with sizes {} respectively.'
              .format(self.rnn_type, len(self.h_size), ', '.join(map(str, self.h_size))))
        print('The inputs are (in this order): {}'.format(', '.join(map(str, inputs_list))))
        print('The outputs are (in this order): {}'.format(', '.join(map(str, outputs_list))))

    def get_internal_states(self):
        states_list = []
        for layer in self.layers:
            if (('gru' in layer.name) or
                    ('lstm' in layer.name) or
                        ('rnn' in layer.name)):
                single_states = []
                for single_state in layer.states:
                    single_state = single_state.numpy()
                    single_states.append(single_state)

                states_list.append(single_states)
            else:
                states_list.append(None)
        return states_list

    def load_internal_states(self, states):

        for layer, state in zip(self.layers, states):
            # print(layer)
            # print(state)
            if (('gru' in layer.name) or
                    ('lstm' in layer.name) or
                        ('rnn' in layer.name)):
                layer.reset_states(state[0])


def load_data(args, filepath=None, columns_list=None, norm_inf=False, rnn_full_name=None):
    if filepath is None and args.filepath is not None:
        filepath = args.val_file_name

    if columns_list is None and (args.inputs_list is not None or args.outputs_list is not None):
        columns_list = list(set(args.inputs_list).union(set(args.outputs_list)))

    if type(filepath) == list:
        filepaths = filepath
    else:
        filepaths = [filepath]

    all_dfs = []  # saved separately to get normalization
    all_time_axes = []

    for one_filepath in filepaths:
        # Load dataframe
        print('loading data from ' + str(one_filepath))
        print('')
        df_dense = pd.read_csv(one_filepath, comment='#')
        max_pos = df_dense['s.position'].abs().max()
        print('Max_pos: {}'.format(max_pos))
        # Here time calculation


        if 'time' in df_dense.columns and 'dt' not in df_dense.columns:
            dt = [0.0]
            row_iterator = df_dense.iterrows()
            _, last = row_iterator.next()  # take first item from row_iterator
            for i, row in row_iterator:
                dt.append(row['value']-last['value'])
                last = row
            df_dense['dt'] = np.array(dt)
        elif 'dt' in df_dense.columns and 'time' not in df_dense.columns:
            dt = df_dense['dt']
            t = dt.cumsum()
            df_dense['time'] = t

        # You can shift dt by one time step to know "now" the timestep till the next row
        if args.cheat_dt:
            if 'dt' in df_dense:
                df_dense['dt'] = df_dense['dt'].shift(-1)
                df_dense = df_dense[:-1]

        for i in range(args.downsampling):
            df = df_dense.iloc[i::args.downsampling].reset_index()

            # Get time axis as separate Dataframe
            if 'time' in df.columns:
                t = df['time']
            else:
                t = pd.Series([])
                t.rename('time', inplace=True)

            all_time_axes.append(t)

            # Get only relevant subset of columns
            if columns_list == 'all':
                pass
            else:
                df = df[columns_list]

            all_dfs.append(df)

    return all_dfs, all_time_axes


# This way of doing normalization is fine for long data sets and (relatively) short sequence lengths
# The points from the edges of the datasets count too little
def calculate_normalization_info(df, path_save, rnn_full_name):
    if type(df) is list:
        df_total = pd.concat(df)
    else:
        df_total = df

    if 'time' in df_total.columns:
        df_total.drop('time',
                      axis='columns', inplace=True)

    df_mean = df_total.mean(axis=0)
    df_std = df_total.std(axis=0)
    df_max = df_total.max(axis=0)
    df_min = df_total.min(axis=0)
    frame = {'mean': df_mean, 'std': df_std, 'max': df_max, 'min': df_min}
    df_norm_info = pd.DataFrame(frame).transpose()

    df_norm_info.to_csv(path_save + rnn_full_name + '-norm' + '.csv')

    # Plot historgrams to make the firs check about gaussian assumption
    # for feature in df_total.columns:
    #     plt.hist(df_total[feature].to_numpy(), 50, density=True, facecolor='g', alpha=0.75)
    #     plt.title(feature)
    #     plt.show()

    return df_norm_info


def load_normalization_info(path_save, rnn_full_name):
    return pd.read_csv(path_save + rnn_full_name + '-norm' + '.csv', index_col=0)


def normalize_feature(feature, normalization_info, normalization_type='minmax_sym', name=None):
    """feature needs to have atribute name!!!"""

    if hasattr(feature, 'name'):
        name = feature.name
    else:
        pass

    if normalization_type == 'gaussian':
        col_mean = normalization_info.loc['mean', name]
        col_std = normalization_info.loc['std', name]
        if col_std == 0:
            return 0
        else:
            return (feature - col_mean) / col_std
    elif normalization_type == 'minmax_pos':
        col_min = normalization_info.loc['min', name]
        col_max = normalization_info.loc['max', name]
        if (col_max - col_min) == 0:
            return 0
        else:
            return (feature - col_min) / (col_max - col_min)
    elif normalization_type == 'minmax_sym':
        col_min = normalization_info.loc['min', name]
        col_max = normalization_info.loc['max', name]
        if (col_max - col_min) == 0:
            return 0
        else:
            return -1.0 + 2.0 * (feature - col_min) / (col_max - col_min)


def normalize_df(dfs, normalization_info, normalization_type='minmax_sym'):

    if type(dfs) is list:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].apply(normalize_feature, axis=0,
                                  normalization_info=normalization_info,
                                  normalization_type=normalization_type)
    else:
        dfs = dfs.apply(normalize_feature, axis=0,
                                  normalization_info=normalization_info,
                                  normalization_type=normalization_type)

    return dfs

def denormalize_feature(feature, normalization_info, normalization_type='minmax_sym', name=None):
    """feature needs to have atribute name!!!"""

    if hasattr(feature, 'name'):
        name = feature.name
    else:
        pass

    if normalization_type == 'gaussian':
        col_mean = normalization_info.loc['mean', name]
        col_std = normalization_info.loc['std', name]
        return feature * col_std + col_mean
    elif normalization_type == 'minmax_pos':
        col_min = normalization_info.loc['min', name]
        col_max = normalization_info.loc['max', name]
        return feature * (col_max - col_min) + col_min
    elif normalization_type == 'minmax_sym':
        col_min = normalization_info.loc['min', name]
        col_max = normalization_info.loc['max', name]
        return ((feature + 1.0) / 2.0) * (col_max - col_min) + col_min

def denormalize_df(dfs, normalization_info, normalization_type='minmax_sym'):

    if type(dfs) is list:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].apply(denormalize_feature, axis=0,
                                  normalization_info=normalization_info,
                                  normalization_type=normalization_type)
    else:
        dfs = dfs.apply(denormalize_feature, axis=0,
                                  normalization_info=normalization_info,
                                  normalization_type=normalization_type)

    return dfs


class Dataset(keras.utils.Sequence):
    def __init__(self, dfs,
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
            features.reset_index(inplace=True)  # Reset index
            targets.reset_index(inplace=True)

            features = features[inputs_list]
            targets = targets[outputs_list]

            self.data.append(features)
            self.labels.append(targets)

        self.args = args

        self.exp_len = None
        self.warm_up_len = self.args.warm_up_len
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

    def get_all_targets(self):
        all_targets = []
        for i in range(self.number_of_samples):
            _, targets = self.get_series(i)
            all_targets.append(targets)
        all_targets = np.stack(all_targets)
        return all_targets

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


def plot_results(net,
                 args,
                 dataset=None,
                 normalization_info=None,
                 time_axes=None,
                 testset_filepath=None,
                 inputs_list=None,
                 outputs_list=None,
                 closed_loop_list=None,
                 exp_len=None,
                 warm_up_len=None,
                 closed_loop_enabled=False,
                 comment='',
                 rnn_full_name=None,
                 save=False,
                 close_loop_idx=None,
                 path_save=None):
    """
    This function accepts RNN instance, arguments and CartPole instance.
    It runs one random experiment with CartPole,
    inputs the data into RNN and check how well RNN predicts CartPole state one time step ahead of time
    """

    rnn_full_name = net.rnn_full_name
    rnn_name = net.rnn_name
    inputs_list = net.inputs_list
    outputs_list = net.outputs_list

    if path_save is None and args is not None:
        path_save = args.path_save

    if testset_filepath is None:
        testset_filepath = args.val_file_name
        if type(testset_filepath) == list:
            testset_filepath = testset_filepath[0]

    if warm_up_len is None:
        warm_up_len = args.warm_up_len

    if exp_len is None:
        exp_len = args.exp_len

    columns_list = list(set(inputs_list).union(set(outputs_list)))

    if closed_loop_enabled and (closed_loop_list is None):
        closed_loop_list = args.close_loop_for
        if closed_loop_list is None:
            raise ValueError('RNN closed-loop-inputs not provided!')

    if close_loop_idx is None:
        close_loop_idx = exp_len//2

    # Create new model which will return after every time step

    net_predict = myNN(rnn_name,
                       inputs_list,
                       outputs_list,
                       warm_up_len=1,
                       return_sequence=False,
                       batchSize=1,
                       stateful=True
                       )
    net_predict.rnn_full_name = rnn_full_name

    net_predict.set_weights(net.get_weights())

    # net_predict.summary()

    if normalization_info is None:
        normalization_info = load_normalization_info(path_save, rnn_full_name)

    if dataset is None or dataset.time_axes is None:
        test_dfs, time_axes = load_data(args, testset_filepath, columns_list=columns_list)
        test_dfs_norm = normalize_df(test_dfs, normalization_info)
        test_set = Dataset(test_dfs_norm, args,
                           time_axes=time_axes, exp_len=exp_len,
                           inputs_list=inputs_list, outputs_list=outputs_list)
        del test_dfs
    else:
        test_set = copy.deepcopy(dataset)
        test_set.reset_exp_len(exp_len=exp_len)

    # Format the experiment data
    features, targets, time_axis = test_set.get_experiment(0)  # Put number in brackets to get the same idx at every run

    features_pd = pd.DataFrame(data=features, columns=inputs_list)
    targets_pd = pd.DataFrame(data=targets, columns=outputs_list)

    rnn_outputs = pd.DataFrame(columns=outputs_list)

    idx_cl = 0
    close_the_loop = False

    for index, row in features_pd.iterrows():
        # states = net_predict.get_internal_states()
        # net_predict.reset_states()
        # net_predict.load_internal_states(states)
        rnn_input = pd.DataFrame(copy.deepcopy(row)).transpose().reset_index(drop=True)

        if idx_cl == close_loop_idx:
            close_the_loop = True
        if closed_loop_enabled and close_the_loop and (normalized_rnn_output is not None):
            rnn_input[closed_loop_list] = normalized_rnn_output[closed_loop_list]
        rnn_input = np.squeeze(rnn_input.to_numpy())
        rnn_input = rnn_input[np.newaxis, np.newaxis, :]
        # t2 = timeit.default_timer()
        normalized_rnn_output = net_predict.predict_on_batch(rnn_input)
        # t3 = timeit.default_timer()
        # print('t3 evaluation {} ms'.format((t3 - t2) * 1000.0))
        normalized_rnn_output = np.squeeze(normalized_rnn_output).tolist()
        normalized_rnn_output = copy.deepcopy(pd.DataFrame(data=[normalized_rnn_output], columns=outputs_list))

        rnn_outputs = rnn_outputs.append(copy.deepcopy(normalized_rnn_output), ignore_index=True)
        idx_cl += 1

    features_pd_denorm = denormalize_df(features_pd, normalization_info)
    targets_pd_denorm = denormalize_df(targets_pd, normalization_info)
    rnn_outputs_denorm = denormalize_df(rnn_outputs, normalization_info)
    figs = plot_results_specific(targets_pd_denorm, rnn_outputs_denorm, features_pd_denorm, time_axis, comment, closed_loop_enabled,
                                     close_loop_idx)

    plt.show()

    if save:
        # Make folders if not yet exist
        try:
            os.makedirs('save_plots_tf')
        except FileExistsError:
            pass
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("-%d%b%Y_%H%M%S")

        for i in range(len(figs)):
            fig = figs[i]
            figNrStr = '-'+str(i)+''
            if rnn_full_name is not None:
                fig.savefig('./save_plots_tf/' + rnn_full_name + figNrStr +timestampStr + '.png')
            else:
                fig.savefig('./save_plots_tf/' + figNrStr + timestampStr + '.png')
