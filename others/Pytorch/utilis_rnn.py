import torch
import torch.nn as nn
from torch.utils import data

from datetime import datetime

import collections
import os

import random as rnd

import copy

from Modeling.Pytorch.utilis_rnn_specific import *
from SI_Toolkit.load_and_normalize import load_normalization_info, load_data, normalize_df, denormalize_df


def get_device():
    """
    Small function to correctly send data to GPU or CPU depending what is available
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


# Set seeds everywhere required to make results reproducible
def set_seed(args):
    seed = args.seed
    rnd.seed(seed)
    np.random.seed(seed)



# Print parameter count
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def print_parameter_count(net):
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('::: # network all parameters: ' + str(pytorch_total_params))
    print('::: # network trainable parameters: ' + str(pytorch_trainable_params))
    print('')


def load_pretrained_rnn(net, pt_path, device):
    """
    A function loading parameters (weights and biases) from a previous training to a net RNN instance
    :param net: An instance of RNN
    :param pt_path: path to .pt file storing weights and biases
    :return: No return. Modifies net in place.
    """
    pre_trained_model = torch.load(pt_path, map_location=device)
    print("Loading Model: ", pt_path)
    print('')

    pre_trained_model = list(pre_trained_model.items())
    new_state_dict = collections.OrderedDict()
    count = 0
    num_param_key = len(pre_trained_model)
    for key, value in net.state_dict().items():
        if count >= num_param_key:
            break
        layer_name, weights = pre_trained_model[count]
        new_state_dict[key] = weights
        # print("Pre-trained Layer: %s - Loaded into new layer: %s" % (layer_name, key))
        count += 1
    print('')
    net.load_state_dict(new_state_dict)


# Initialize weights and biases - should be only applied if no pretrained net loaded
def initialize_weights_and_biases(net):
    print('Initialize weights and biases')
    for name, param in net.named_parameters():
        print('Initialize {}'.format(name))
        if 'gru' in name:
            if 'weight' in name:
                nn.init.orthogonal_(param)
        if 'linear' in name:
            if 'weight' in name:
                nn.init.orthogonal_(param)
                # nn.init.xavier_uniform_(param)
        if 'bias' in name:  # all biases
            nn.init.constant_(param, 0)
    print('')


def create_rnn_instance(rnn_name=None, inputs_list=None, outputs_list=None, load_rnn=None, path_save=None, device=None):
    if load_rnn is not None and load_rnn != 'last':
        # 1) Find csv with this name if exists load name, inputs and outputs list
        #       if it does not exist raise error
        # 2) Create corresponding net
        # 3) Load parameters from corresponding pt file

        filename = load_rnn
        print('Loading a pretrained RNN with the full name: {}'.format(filename))
        print('')
        txt_filename = filename + '.txt'
        pt_filename = filename + '.pt'
        txt_path = path_save + txt_filename
        pt_path = path_save + pt_filename

        if not os.path.isfile(txt_path):
            raise ValueError(
                'The corresponding .txt file is missing (information about inputs and outputs) at the location {}'.format(
                    txt_path))
        if not os.path.isfile(pt_path):
            raise ValueError(
                'The corresponding .pt file is missing (information about weights and biases) at the location {}'.format(
                    pt_path))

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
        net = Sequence(rnn_name=rnn_name, inputs_list=inputs_list, outputs_list=outputs_list)
        net.rnn_full_name = load_rnn

        # Load the parameters
        load_pretrained_rnn(net, pt_path, device)

    elif load_rnn == 'last':
        files_found = False
        while (not files_found):
            try:
                import glob
                list_of_files = glob.glob(path_save + '/*.txt')
                txt_path = max(list_of_files, key=os.path.getctime)
            except FileNotFoundError:
                raise ValueError('No information about any pretrained network found at {}'.format(path_save))

            f = open(txt_path, 'r')
            lines = f.readlines()
            rnn_name = lines[1].rstrip("\n")
            pre_rnn_full_name = lines[4].rstrip("\n")
            inputs_list = lines[7].rstrip("\n").split(sep=', ')
            outputs_list = lines[10].rstrip("\n").split(sep=', ')
            f.close()

            pt_path = path_save + pre_rnn_full_name + '.pt'
            if not os.path.isfile(pt_path):
                print('The .pt file is missing (information about weights and biases) at the location {}'.format(
                    pt_path))
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
        net = Sequence(rnn_name=rnn_name, inputs_list=inputs_list, outputs_list=outputs_list)
        net.rnn_full_name = pre_rnn_full_name

        # Load the parameters
        load_pretrained_rnn(net, pt_path, device)


    else:  # a.load_rnn is None
        print('No pretrained network specified. I will train a network from scratch.')
        print('')
        # Construct the requested RNN
        net = Sequence(rnn_name=rnn_name, inputs_list=inputs_list, outputs_list=outputs_list)
        initialize_weights_and_biases(net)

    return net, rnn_name, inputs_list, outputs_list


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


# FIXME: To tailor this sequence class according to the commands and state_variables of cartpole
class Sequence(nn.Module):
    """"
    Our RNN class.
    """

    def __init__(self, rnn_name, inputs_list, outputs_list):
        super(Sequence, self).__init__()
        """Initialization of an RNN instance
        We assume that inputs may be both commands and state variables, whereas outputs are always state variables
        """

        # Check if GPU is available. If yes device='cuda:0' if not device='cpu'
        self.device = get_device()

        self.rnn_name = rnn_name
        self.rnn_full_name = None

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

        if 'GRU' in names:
            self.rnn_type = 'GRU'
        elif 'LSTM' in names:
            self.rnn_type = 'LSTM'
        else:
            self.rnn_type = 'RNN-Basic'

        # Construct network

        if self.rnn_type == 'GRU':
            self.rnn_cell = [nn.GRUCell(len(inputs_list), self.h_size[0]).to(get_device())]
            for i in range(len(self.h_size) - 1):
                self.rnn_cell.append(nn.GRUCell(self.h_size[i], self.h_size[i + 1]).to(get_device()))
        elif self.rnn_type == 'LSTM':
            self.rnn_cell = [nn.LSTMCell(len(inputs_list), self.h_size[0]).to(get_device())]
            for i in range(len(self.h_size) - 1):
                self.rnn_cell.append(nn.LSTMCell(self.h_size[i], self.h_size[i + 1]).to(get_device()))
        else:
            self.rnn_cell = [nn.RNNCell(len(inputs_list), self.h_size[0]).to(get_device())]
            for i in range(len(self.h_size) - 1):
                self.rnn_cell.append(nn.RNNCell(self.h_size[i], self.h_size[i + 1]).to(get_device()))

        self.linear = nn.Linear(self.h_size[-1], len(outputs_list))  # RNN out

        self.layers = nn.ModuleList([])
        for cell in self.rnn_cell:
            self.layers.append(cell)
        self.layers.append(self.linear)

        # Count data samples (=time steps)
        self.sample_counter = 0
        # Declaration of the variables keeping internal state of GRU hidden layers
        self.h = [None] * len(self.h_size)
        self.c = [None] * len(self.h_size)  # Internal state cell - only matters for LSTM
        # Variable keeping the most recent output of RNN
        self.output = None
        # List storing the history of RNN outputs
        self.outputs = []

        # Send the whole RNN to GPU if available, otherwise send it to CPU
        self.to(self.device)

        print('Constructed a neural network of type {}, with {} hidden layers with sizes {} respectively.'
              .format(self.rnn_type, len(self.h_size), ', '.join(map(str, self.h_size))))
        print('The inputs are (in this order): {}'.format(', '.join(map(str, inputs_list))))
        print('The outputs are (in this order): {}'.format(', '.join(map(str, outputs_list))))

    def reset(self):
        """
        Reset the network (not the weights!)
        """
        self.sample_counter = 0
        self.h = [None] * len(self.h_size)
        self.c = [None] * len(self.h_size)
        self.output = None
        self.outputs = []

    def forward(self, rnn_input):

        """
        Predicts future CartPole states IN "OPEN LOOP"
        (at every time step prediction for the next time step is done based on the true CartPole state)
        """

        # Initialize hidden layers - this change at every call as the batch size may vary
        for i in range(len(self.h_size)):
            self.h[i] = torch.zeros(rnn_input.size(1), self.h_size[i], dtype=torch.float).to(self.device)
            self.c[i] = torch.zeros(rnn_input.size(1), self.h_size[i], dtype=torch.float).to(self.device)

        # The for loop takes the consecutive time steps from input plugs them into RNN and save the outputs into a list
        # THE NETWORK GETS ALWAYS THE GROUND TRUTH, THE REAL STATE OF THE CARTPOLE, AS ITS INPUT
        # IT PREDICTS THE STATE OF THE CARTPOLE ONE TIME STEP AHEAD BASED ON TRUE STATE NOW
        for iteration, input_t in enumerate(rnn_input.chunk(rnn_input.size(0), dim=0)):

            # Propagate input through RNN layers
            if self.rnn_type == 'LSTM':
                self.h[0], self.c[0] = self.layers[0](input_t.squeeze(0), (self.h[0], self.c[0]))
                for i in range(len(self.h_size) - 1):
                    self.h[i + 1], self.c[i + 1] = self.layers[i + 1](self.h[i], (self.h[i + 1], self.c[i + 1]))
            else:
                self.h[0] = self.layers[0](input_t.squeeze(0), self.h[0])
                for i in range(len(self.h_size) - 1):
                    self.h[i + 1] = self.layers[i + 1](self.h[i], self.h[i + 1])
            self.output = self.layers[-1](self.h[-1])

            self.outputs += [self.output]
            self.sample_counter = self.sample_counter + 1

        # In the train mode we want to continue appending the outputs by calling forward function
        # The outputs will be saved internally in the network instance as a list
        # Otherwise we want to transform outputs list to a tensor and return it
        return self.output

    def return_outputs_history(self):
        return torch.stack(self.outputs, 1)


import pandas as pd

#
# def load_data(a, filepath=None, columns_list=None, norm_inf=False, rnn_full_name=None, downsample=1):
#     if filepath is None:
#         filepath = a.val_file_name
#
#     if columns_list is None:
#         columns_list = list(set(a.inputs_list).union(set(a.outputs_list)))
#
#     if type(filepath) == list:
#         filepaths = filepath
#     else:
#         filepaths = [filepath]
#
#     all_dfs = []  # saved separately to get normalization
#     all_time_axes = []
#
#     for one_filepath in filepaths:
#         # Load dataframe
#         print('loading data from ' + str(one_filepath))
#         print('')
#         df = pd.read_csv(one_filepath, comment='#')
#         df=df.iloc[::downsample].reset_index()
#
#         # You can shift dt by one time step to know "now" the timestep till the next row
#         if a.cheat_dt:
#             if 'dt' in df:
#                 df['dt'] = df['dt'].shift(-1)
#                 df = df[:-1]
#
#         # FIXME: Make calculation of dt compatible with downsampling
#         # Get time axis as separate Dataframe
#         if 'time' in df.columns:
#             t = df['time']
#         elif 'dt' in df.columns:
#             dt = df['dt']
#             t = dt.cumsum()
#             t.rename('time', inplace=True)
#         else:
#             t = pd.Series([])
#             t.rename('time', inplace=True)
#
#         time_axis = t
#         all_time_axes.append(time_axis)
#
#         # Get only relevant subset of columns
#         if columns_list == 'all':
#             pass
#         else:
#             df = df[columns_list]
#
#         all_dfs.append(df)
#
#
#     return all_dfs, all_time_axes


#
# # This way of doing normalization is fine for long data sets and (relatively) short sequence lengths
# # The points from the edges of the datasets count too little
# def calculate_normalization_info(df, PATH_TO_EXPERIMENT_RECORDINGS, rnn_full_name):
#     if type(df) is list:
#         df_total = pd.concat(df)
#     else:
#         df_total = df
#
#     if 'time' in df_total.columns:
#         df_total.drop('time',
#                       axis='columns', inplace=True)
#
#     df_mean = df_total.mean(axis=0)
#     df_std = df_total.std(axis=0)
#     df_max = df_total.max(axis=0)
#     df_min = df_total.min(axis=0)
#     frame = {'mean': df_mean, 'std': df_std, 'max': df_max, 'min': df_min}
#     df_norm_info = pd.DataFrame(frame).transpose()
#
#     df_norm_info.to_csv(PATH_TO_EXPERIMENT_RECORDINGS + rnn_full_name + '-norm' + '.csv')
#
#     # Plot historgrams to make the firs check about gaussian assumption
#     # for feature in df_total.columns:
#     #     plt.hist(df_total[feature].to_numpy(), 50, density=True, facecolor='g', alpha=0.75)
#     #     plt.title(feature)
#     #     plt.show()
#
#     return df_norm_info
#
#
# def load_normalization_info(PATH_TO_EXPERIMENT_RECORDINGS, rnn_full_name):
#     return pd.read_csv(PATH_TO_EXPERIMENT_RECORDINGS + rnn_full_name + '-norm' + '.csv', index_col=0)
#
#
# def normalize_df(dfs, normalization_info, normalization_type='minmax_sym'):
#     if normalization_type == 'gaussian':
#         def normalize_feature(col):
#             col_mean = normalization_info.loc['mean', col.name]
#             col_std = normalization_info.loc['std', col.name]
#             return (col - col_mean) / col_std
#     elif normalization_type == 'minmax_pos':
#         def normalize_feature(col):
#             col_min = normalization_info.loc['min', col.name]
#             col_max = normalization_info.loc['max', col.name]
#             return (col - col_min) / (col_max - col_min)
#     elif normalization_type == 'minmax_sym':
#         def normalize_feature(col):
#             col_min = normalization_info.loc['min', col.name]
#             col_max = normalization_info.loc['max', col.name]
#             return -1.0 + 2.0 * (col - col_min) / (col_max - col_min)
#
#     if type(dfs) is list:
#         for i in range(len(dfs)):
#             dfs[i] = dfs[i].apply(normalize_feature, axis=0)
#     else:
#         dfs = dfs.apply(normalize_feature, axis=0)
#
#     return dfs
#
#
# def denormalize_df(dfs, normalization_info, normalization_type='minmax_sym'):
#     if normalization_type == 'gaussian':
#         def denormalize_feature(col):
#             col_mean = normalization_info.loc['mean', col.name]
#             col_std = normalization_info.loc['std', col.name]
#             return col * col_std + col_mean
#     elif normalization_type == 'minmax_pos':
#         def denormalize_feature(col):
#             col_min = normalization_info.loc['min', col.name]
#             col_max = normalization_info.loc['max', col.name]
#             return col * (col_max - col_min) + col_min
#     elif normalization_type == 'minmax_sym':
#         def denormalize_feature(col):
#             col_min = normalization_info.loc['min', col.name]
#             col_max = normalization_info.loc['max', col.name]
#             return ((col + 1.0) / 2.0) * (col_max - col_min) + col_min
#
#     if type(dfs) is list:
#         for i in range(len(dfs)):
#             dfs[i] = dfs[i].apply(denormalize_feature, axis=0)
#     else:
#         dfs = dfs.apply(denormalize_feature, axis=0)
#
#     return dfs


class Dataset(data.Dataset):
    def __init__(self, dfs, args, time_axes=None, seq_len=None):
        'Initialization - divide data in features and labels'

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

            features = features[args.inputs_list]
            targets = targets[args.outputs_list]

            self.data.append(features)
            self.labels.append(targets)

        self.args = args

        self.seq_len = None
        self.df_lengths = []
        self.df_lengths_cs = []
        self.number_of_samples = 0

        self.time_axes = time_axes

        self.reset_seq_len(seq_len=seq_len)

    def reset_seq_len(self, seq_len=None):
        """
        This method should be used if the user wants to change the seq_len without creating new Dataset
        Please remember that one can reset it again to come back to old configuration
        :param seq_len: Gives new user defined seq_len. Call empty to come back to default.
        """
        if seq_len is None:
            self.seq_len = self.args.seq_len  # Sequence length
        else:
            self.seq_len = seq_len

        self.df_lengths = []
        self.df_lengths_cs = []
        if type(self.data) == list:
            for data_set in self.data:
                self.df_lengths.append(data_set.shape[0] - self.seq_len)
                if not self.df_lengths_cs:
                    self.df_lengths_cs.append(self.df_lengths[0])
                else:
                    self.df_lengths_cs.append(self.df_lengths_cs[-1] + self.df_lengths[-1])
            self.number_of_samples = self.df_lengths_cs[-1]

        else:
            self.number_of_samples = self.data.shape[0] - self.seq_len

    def __len__(self):
        'Total number of samples'
        return self.number_of_samples

    def __getitem__(self, idx, get_time_axis=False):
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
        features = self.data[idx_data_set].to_numpy()[idx:idx + self.seq_len, :]
        # Every point in features has its target value corresponding to the next time step:
        targets = self.labels[idx_data_set].to_numpy()[idx:idx + self.seq_len]
        # After feeding the whole sequence we just compare the final output of the RNN with the state following afterwards
        # targets = self.labels[idx_data_set].to_numpy()[idx + self.seq_len-1]

        # If get_time_axis try to obtain a vector of time data for the chosen sample
        if get_time_axis:
            try:
                time_axis = self.time_axes[idx_data_set].to_numpy()[idx:idx + self.seq_len + 1]
            except IndexError:
                time_axis = []

        # Return results
        if get_time_axis:
            return features, targets, time_axis
        else:
            return features, targets

    def get_experiment(self, idx=None):
        if self.time_axes is None:
            raise Exception('No time information available!')
        if idx is None:
            idx = np.random.randint(0, self.number_of_samples)
        return self.__getitem__(idx, get_time_axis=True)


def plot_results(net,
                 args,
                 dataset=None,
                 normalization_info = None,
                 time_axes=None,
                 filepath=None,
                 inputs_list=None,
                 outputs_list=None,
                 closed_loop_list=None,
                 seq_len=None,
                 warm_up_len=None,
                 closed_loop_enabled=False,
                 comment='',
                 rnn_full_name=None,
                 save=False,
                 close_loop_idx=512):
    """
    This function accepts RNN instance, arguments and CartPole instance.
    It runs one random experiment with CartPole,
    inputs the data into RNN and check how well RNN predicts CartPole state one time step ahead of time
    """

    rnn_full_name = net.rnn_full_name

    if filepath is None:
        filepath = args.val_file_name
        if type(filepath) == list:
            filepath = filepath[0]

    if warm_up_len is None:
        warm_up_len = args.warm_up_len

    if seq_len is None:
        seq_len = args.seq_len

    if inputs_list is None:
        inputs_list = args.inputs_list
        if inputs_list is None:
            raise ValueError('RNN inputs not provided!')

    if outputs_list is None:
        outputs_list = args.outputs_list
        if outputs_list is None:
            raise ValueError('RNN outputs not provided!')

    if closed_loop_enabled and (closed_loop_list is None):
        closed_loop_list = args.close_loop_for
        if closed_loop_list is None:
            raise ValueError('RNN closed-loop-inputs not provided!')

    net.reset()
    net.eval()
    device = get_device()

    if normalization_info is None:
        normalization_info = load_normalization_info(args.PATH_TO_EXPERIMENT_RECORDINGS, rnn_full_name)

    if dataset is None or time_axes is None:
        test_dfs, time_axes = load_data(args, filepath)
        test_dfs_norm = normalize_df(test_dfs, normalization_info)
        test_set = Dataset(test_dfs_norm, args, time_axes=time_axes, seq_len=seq_len)
        del test_dfs
    else:
        test_set = copy.deepcopy(dataset)
        test_set.reset_seq_len(seq_len=seq_len)

    # Format the experiment data
    features, targets, time_axis = test_set.get_experiment(1)  # Put number in brackets to get the same idx at every run

    features_pd = pd.DataFrame(data=features, columns=inputs_list)
    targets_pd = pd.DataFrame(data=targets, columns=outputs_list)

    rnn_outputs = pd.DataFrame(columns=outputs_list)

    warm_up_idx = 0
    rnn_input_0 = copy.deepcopy(features_pd.iloc[0])
    # Does not bring anything. Why? 0-state shouldn't have zero internal state due to biases...
    while warm_up_idx < warm_up_len:
        rnn_input = rnn_input_0
        rnn_input = np.squeeze(rnn_input.to_numpy())
        rnn_input = torch.from_numpy(rnn_input).float().unsqueeze(0).unsqueeze(0).to(device)
        net(rnn_input=rnn_input)
        warm_up_idx += 1
    net.outputs = []
    net.sample_counter = 0

    idx_cl = 0
    close_the_loop = False

    for index, row in features_pd.iterrows():
        rnn_input = pd.DataFrame(copy.deepcopy(row)).transpose().reset_index(drop=True)
        if idx_cl == close_loop_idx:
            close_the_loop = True
        if closed_loop_enabled and close_the_loop and (normalized_rnn_output is not None):
            rnn_input[closed_loop_list] = normalized_rnn_output[closed_loop_list]
        rnn_input = np.squeeze(rnn_input.to_numpy())
        rnn_input = torch.from_numpy(rnn_input).float().unsqueeze(0).unsqueeze(0).to(device)
        normalized_rnn_output = net(rnn_input=rnn_input)
        normalized_rnn_output = np.squeeze(normalized_rnn_output.detach().cpu().numpy()).tolist()
        normalized_rnn_output = copy.deepcopy(pd.DataFrame(data=[normalized_rnn_output], columns=outputs_list))
        rnn_outputs = rnn_outputs.append(copy.deepcopy(normalized_rnn_output), ignore_index=True)
        idx_cl += 1

    targets_pd_denorm = denormalize_df(targets_pd, normalization_info)
    rnn_outputs_denorm = denormalize_df(rnn_outputs, normalization_info)
    fig, axs = plot_results_specific(targets_pd_denorm, rnn_outputs_denorm, time_axis, comment, closed_loop_enabled, close_loop_idx)

    plt.show()

    if save:
        # Make folders if not yet exist
        try:
            os.makedirs('save_plots')
        except FileExistsError:
            pass
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("-%d%b%Y_%H%M%S")
        if rnn_full_name is not None:
            fig.savefig('./save_plots/' + rnn_full_name + timestampStr + '.png')
        else:
            fig.savefig('./save_plots/' + timestampStr + '.png')
