import torch
import torch.nn as nn
from torch.utils import data
from datetime import datetime
from IPython.display import Image

import matplotlib.pyplot as plt
import numpy as np

from src.utilis import Generate_Experiment
import collections
import os

import random as rnd

import copy

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
        print("Pre-trained Layer: %s - Loaded into new layer: %s" % (layer_name, key))
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

        # Load the parameters
        load_pretrained_rnn(net, pt_path, device)

    elif load_rnn == 'last':
        files_found = False
        while(not files_found):
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

        # Load the parameters
        load_pretrained_rnn(net, pt_path, device)


    else:  # args.load_rnn is None
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


#FIXME: To tailor this sequence class according to the commands and state_variables of cartpole
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

def load_data(args, filepath=None, inputs_list=None, outputs_list=None):

    if filepath is None:
        filepath = args.val_file_name

    if inputs_list is None:
        inputs_list = args.inputs_list

    if outputs_list is None:
        outputs_list = args.outputs_list

    if type(filepath) == list:
        filepaths = filepath
    else:
        filepaths = [filepath]

    all_features = []
    all_targets = []

    for one_filepath in filepaths:
        # Load dataframe
        print('loading data from ' + str(one_filepath))
        print('')
        df = pd.read_csv(one_filepath, comment='#')

        if args.cheat_dt:
            df['dt'] = df['dt'].shift(-1)
            df = df[:-1]

        # Get Raw Data
        inputs = copy.deepcopy(df)
        outputs = copy.deepcopy(df)

        inputs.drop(inputs.tail(1).index, inplace=True) # Drop last row
        outputs.drop(outputs.head(1).index, inplace=True)
        inputs.reset_index(inplace=True)  # Reset index
        outputs.reset_index(inplace=True)

        inputs = inputs[inputs_list]
        outputs = outputs[outputs_list]


        features = np.array(inputs)
        targets = np.array(outputs)
        all_features.append(features)
        all_targets.append(targets)

    if type(filepath) == list:
        return all_features, all_targets
    else:
        return features, targets


class Dataset(data.Dataset):
    def __init__(self, df, labels, args, seq_len=None):
        'Initialization'
        self.data = df
        self.labels = labels
        self.args = args

        self.seq_len = None
        self.df_lengths = []
        self.df_lengths_cs = []
        self.number_of_samples = 0

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
                    self.df_lengths_cs.append(self.df_lengths_cs[-1]+self.df_lengths[-1])
            self.number_of_samples = self.df_lengths_cs[-1]

        else:
            self.number_of_samples = self.data.shape[0] - self.seq_len



    def __len__(self):
        'Total number of samples'
        return self.number_of_samples

    def __getitem__(self, idx):
        if type(self.data) == list:
            idx_data_set = next(i for i, v in enumerate(self.df_lengths_cs) if v > idx)
            if idx_data_set == 0:
                pass
            else:
                idx -= self.df_lengths_cs[idx_data_set-1]
            return self.data[idx_data_set][idx:idx + self.seq_len, :], self.labels[idx_data_set][idx:idx + self.seq_len]
        else:
            return self.data[idx:idx + self.seq_len, :], self.labels[idx:idx + self.seq_len]

def plot_results(net,
                 args,
                 dataset=None,
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
                 close_loop_idx=150):
    """
    This function accepts RNN instance, arguments and CartPole instance.
    It runs one random experiment with CartPole,
    inputs the data into RNN and check how well RNN predicts CartPole state one time step ahead of time
    """

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

    # normalization_info = NORMALIZATION_INFO

#     # Here in contrary to ghoast car implementation I have
#     # rnn_input[name] /= normalization_info.iloc[0][column]
#     # and not
#     # rnn_input.iloc[0][column] /= normalization_info.iloc[0][column]
#     # It is because rnn_input is just row (type = Series) and not the whole DataFrame (type = DataFrame)
#
    # def denormalize_output(output_series):
    #     for name in output_series.index:
    #         if normalization_info.iloc[0][name] is not None:
    #             output_series[name] *= normalization_info.iloc[0][name]
    #     return output_series

#
#     # Reset the internal state of RNN cells, clear the output memory, etc.
    net.reset()
    net.eval()
    device = get_device()
#
    if dataset is None:
        dev_features, dev_targets = load_data(args, filepath, inputs_list=inputs_list, outputs_list=outputs_list)
        dev_set = Dataset(dev_features, dev_targets, args, seq_len=seq_len)
    else:
        dev_set = copy.deepcopy(dataset)
        dev_set.reset_seq_len(seq_len=seq_len)

    # Format the experiment data
    features, targets = dev_set[0]
#
    features_pd = pd.DataFrame(data=features, columns=inputs_list)
    targets_pd = pd.DataFrame(data=targets, columns=outputs_list)
    #FIXME: Add denormalization by uncommenting the next line
    # targets_pd = pd.DataFrame(data=targets, columns=outputs_list).apply(denormalize_output, axis=1)
    rnn_outputs = pd.DataFrame(columns=outputs_list)
    rnn_output = None
#
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

    close_the_loop = False
    idx_cl = 0

    for index, row in features_pd.iterrows():
        rnn_input = copy.deepcopy(row)
        if idx_cl == close_loop_idx:
            close_the_loop = True
        if closed_loop_enabled and close_the_loop and (rnn_output is not None):
            rnn_input[closed_loop_list] = normalized_rnn_output[closed_loop_list]
        rnn_input = np.squeeze(rnn_input.to_numpy())
        rnn_input = torch.from_numpy(rnn_input).float().unsqueeze(0).unsqueeze(0).to(device)
        normalized_rnn_output = net(rnn_input=rnn_input)
        normalized_rnn_output = list(np.squeeze(normalized_rnn_output.detach().cpu().numpy()))
        normalized_rnn_output = pd.Series(data=normalized_rnn_output, index=outputs_list)
        rnn_output = copy.deepcopy(normalized_rnn_output)
        #FIXME : Enable denormalization
        # denormalize_output(rnn_output)
        rnn_outputs = rnn_outputs.append(rnn_output, ignore_index=True)
        idx_cl += 1


#     # If RNN was given sin and cos of body angle calculate back the body angle
#     if ('body_angle.cos' in rnn_outputs) and ('body_angle.sin' in rnn_outputs) and ('body_angle_deg' not in rnn_outputs):
#         rnn_outputs['body_angle_deg'] = rnn_outputs.apply(SinCos2Angle_wrapper, axis=1)
#     if ('body_angle.cos' in targets_pd) and ('body_angle.sin' in targets_pd) and ('body_angle_deg' not in targets_pd):
#         targets_pd['body_angle_deg'] = targets_pd.apply(SinCos2Angle_wrapper, axis=1)
#
#     # Get the time or # samples axes
    experiment_length  = seq_len
#
    if 'time' in features_pd.columns:
        t = features_pd['time'].to_numpy()
        time_axis = t
        time_axis_string = 'Time [s]'
    elif 'dt' in features_pd.columns:
        dt = features_pd['dt'].to_numpy()
        t = np.cumsum(dt)
        time_axis = t
        time_axis_string = 'Time [s]'
    else:
        samples = np.arange(0, experiment_length)
        time_axis = samples
        time_axis_string = 'Sample number'

    number_of_plots = 0



    if ('s.angle' in targets_pd) and ('s.angle' in rnn_outputs) and ('s.position' in targets_pd) and ('s.position' in rnn_outputs):
        x_target = targets_pd['s.angle'].to_numpy()
        y_target = targets_pd['s.position'].to_numpy()
        x_output = rnn_outputs['s.angle'].to_numpy()
        y_output = rnn_outputs['s.position'].to_numpy()
        number_of_plots += 1

    #FIXME: For number of plots = 1, TypeError: 'AxesSubplot' object is not subscriptable
    number_of_plots=2
#
#     if ('body_angle_deg' in targets_pd) and ('body_angle_deg' in rnn_outputs):
#         body_angle_target = targets_pd['body_angle_deg'].to_numpy()
#         body_angle_output = rnn_outputs['body_angle_deg'].to_numpy()
#         number_of_plots += 1
#
#     if ('velocity_m_per_sec.x' in targets_pd) and ('velocity_m_per_sec.x' in rnn_outputs) and ('velocity_m_per_sec.y' in targets_pd) and ('velocity_m_per_sec.y' in rnn_outputs):
#         vel_x_target = targets_pd['velocity_m_per_sec.x'].to_numpy()
#         vel_y_target = targets_pd['velocity_m_per_sec.y'].to_numpy()
#         vel_x_output = rnn_outputs['velocity_m_per_sec.x'].to_numpy()
#         vel_y_output = rnn_outputs['velocity_m_per_sec.y'].to_numpy()
#         speed_target = np.sqrt((vel_x_target**2)+(vel_y_target**2))
#         speed_output = np.sqrt((vel_x_output ** 2) + (vel_y_output ** 2))
#         number_of_plots += 1
#
#     # Create a figure instance
    fig, axs = plt.subplots(number_of_plots, 1, figsize=(18, 10)) #, sharex=True)  # share x axis so zoom zooms all plots
    plt.subplots_adjust(hspace=0.4)
    start_idx = 0
    axs[0].set_title(comment, fontsize=20)

    axs[0].set_ylabel("Position", fontsize=18)
    axs[0].plot(x_target, pixels2meters(SCREEN_HEIGHT_PIXELS)-y_target, 'k:', markersize=12, label='Ground Truth')
    axs[0].plot(x_output, pixels2meters(SCREEN_HEIGHT_PIXELS)-y_output, 'b', markersize=12, label='Predicted position')

    axs[0].plot(x_target[start_idx], pixels2meters(SCREEN_HEIGHT_PIXELS)-y_target[start_idx], 'g.', markersize=16, label='Start')
    axs[0].plot(x_output[start_idx], pixels2meters(SCREEN_HEIGHT_PIXELS)-y_output[start_idx], 'g.', markersize=16)
    axs[0].plot(x_target[-1], pixels2meters(SCREEN_HEIGHT_PIXELS)-y_target[-1], 'r.', markersize=16, label='End')
    axs[0].plot(x_output[-1], pixels2meters(SCREEN_HEIGHT_PIXELS)-y_output[-1], 'r.', markersize=16)
    if closed_loop_enabled:
        axs[0].plot(x_target[close_loop_idx], pixels2meters(SCREEN_HEIGHT_PIXELS)-y_target[close_loop_idx], '.', color='darkorange', markersize=16, label='connect output->input')
        axs[0].plot(x_output[close_loop_idx], pixels2meters(SCREEN_HEIGHT_PIXELS)-y_output[close_loop_idx], '.', color='darkorange', markersize=16)

    axs[0].tick_params(axis='both', which='major', labelsize=16)

    axs[0].set_xlabel('Angle', fontsize=18)
    axs[0].legend()
#
#
#
#     axs[1].set_ylabel("Body angle (deg)", fontsize=18)
#     axs[1].plot(time_axis, body_angle_target, 'k:', markersize=12, label='Ground Truth')
#     axs[1].plot(time_axis, body_angle_output, 'b', markersize=12, label='Predicted speed')
#
#     axs[1].plot(time_axis[start_idx], body_angle_target[start_idx], 'g.', markersize=16, label='Start')
#     axs[1].plot(time_axis[start_idx], body_angle_output[start_idx], 'g.', markersize=16)
#     axs[1].plot(time_axis[-1], body_angle_target[-1], 'r.', markersize=16, label='End')
#     axs[1].plot(time_axis[-1], body_angle_output[-1], 'r.', markersize=16)
#     if closed_loop_enabled:
#         axs[1].plot(time_axis[close_loop_idx], body_angle_target[close_loop_idx], '.', color='darkorange', markersize=16, label='Connect output->input')
#         axs[1].plot(time_axis[close_loop_idx], body_angle_output[close_loop_idx], '.', color='darkorange', markersize=16)
#
#     axs[1].tick_params(axis='both', which='major', labelsize=16)
#
#     axs[1].set_xlabel(time_axis_string, fontsize=18)
#
#     axs[1].legend()
#
#
#     axs[2].set_ylabel("Speed (m/s)", fontsize=18)
#     axs[2].plot(time_axis, speed_target, 'k:', markersize=12, label='Ground Truth')
#     axs[2].plot(time_axis, speed_output, 'b', markersize=12, label='Predicted speed')
#
#     axs[2].plot(time_axis[start_idx], speed_target[start_idx], 'g.', markersize=16, label='Start')
#     axs[2].plot(time_axis[start_idx], speed_output[start_idx], 'g.', markersize=16)
#     axs[2].plot(time_axis[-1], speed_target[-1], 'r.', markersize=16, label='End')
#     axs[2].plot(time_axis[-1], speed_output[-1], 'r.', markersize=16)
#     if closed_loop_enabled:
#         axs[2].plot(time_axis[close_loop_idx], speed_target[close_loop_idx], '.', color='darkorange', markersize=16, label='Connect output->input')
#         axs[2].plot(time_axis[close_loop_idx], speed_output[close_loop_idx], '.', color='darkorange', markersize=16)
#
#     axs[2].tick_params(axis='both', which='major', labelsize=16)
#
#     axs[2].set_xlabel(time_axis_string, fontsize=18)
#     axs[2].legend()
#
#     plt.ioff()
#     # plt.show()
#     plt.pause(1)
#
#     # Make name settable and with time-date stemp
#     # Save figure to png
    if save:
        # Make folders if not yet exist
        try:
            os.makedirs('save_plots')
        except FileExistsError:
            pass
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d%b%Y_%H%M%S")
        if rnn_full_name is not None:
            fig.savefig('./save_plots/'+rnn_full_name+'.png')
        else:
            fig.savefig('./save_plots/'+timestampStr + '.png')



#FIXME: The M_PER_PIXEL was imported from globals in case of l2race. I am hardcoding it for now
M_PER_PIXEL = 0.10
SCREEN_HEIGHT_PIXELS=768
SCREEN_WIDTH_PIXELS=1024

def pixels2meters(x_map: float):
    """
    The function converts a value in the map units (pixels) to the physical units (meters).
    It is suitable to convert position, velocity or acceleration.
    :param x_map: value in map units (pixels, not necessarily integer)
    :return x_track: Value converted to physical units (meters)
    """
    x_track = x_map * M_PER_PIXEL
    return x_track


def meters2pixels(x_track: float):
    """
    The function converts a value in the map units (pixels) to the physical units (meters).
    In contrast to get_position_on_map() it DOES NOT round the results down to nearest integer.
    It is suitable to convert position, velocity or acceleration.
    :param x_track: Value converted to physical units (meters)
    :return x_map: Value in map units (pixels, not necessarily integer!)
    """
    x_map = x_track / M_PER_PIXEL
    return x_map