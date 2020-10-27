import torch
import torch.nn as nn
from torch.utils import data

from IPython.display import Image

import matplotlib.pyplot as plt
import numpy as np

from src.utilis import Generate_Experiment
import collections
import os

import random as rnd

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




def norm(x):
    m = np.mean(x)
    s = np.std(x)
    y = (x - m) / s
    return y

class Dataset(data.Dataset):
    """
    This is a Dataset class providing a proper data format for Pytorch applications
    It inherits from the standard Pytorch dataset class
    """

    def __init__(self, MyCart, args, train=True, inputs_list=None, outputs_list=None):
        """Initialization"""

        self.MyCart = MyCart

        # Take data set of different size for training and testing of RNN
        if train:
            self.exp_len = args.exp_len_train
        else:
            self.exp_len = args.exp_len_test

        # Recalculate simulation time step from milliseconds to seconds
        self.dt = args.dt / 1000.0  # s
        self.epoch_len = args.epoch_len

        if inputs_list is None:
            inputs_list = args.inputs_list

        if outputs_list is None:
            outputs_list = args.outputs_list

        self.inputs_list = inputs_list
        self.outputs_list = outputs_list

    def __len__(self):
        """
        Total number of samples.
        In this implementation it is meaningless however needs to be preserved due to Pytorch requirements
        """

        return int(self.epoch_len)

    def __getitem__(self, idx):
        """
        When called this function generate a random CartPole experiment
        and returns its recording in a form suitable to be used to train/test RNN
        (inputs list to RNN in features array and expected outputs from RNN )
        :param idx: Normally this parameter let you choose the element of the Dataset.
        In this case every time __getitem__ is called a new random experiment is generated.
        One still need this argument due to construction of Pytorch Dataset class on which this class is based.
        idx can be any integer <epoch_len without any impact on what this function returns.
        We recommend to call __getitem__ method of mydataset instance always with mydataset[0]
        """

        # Generate_Experiment function returns:
        #   states: position of the cart, angle of the pole, etc...
        #   u_effs: control input to the cart
        #   target_positions: The desired position of the cart
        data = Generate_Experiment(MyCart=self.MyCart,  # MyCart contain CartPole dynamics
                                                               exp_len=self.exp_len,
                                                               # How many data points should be generated
                                                               dt=self.dt)  # Simulation time step size

        features = data[self.inputs_list]
        targets = data[self.outputs_list]

        features = features.to_numpy()
        targets = targets.to_numpy()


        # "features" is the array of inputs to the RNN, it consists of states of the CartPole and control input
        # "targets" is the array of CartPole states one time step ahead of "features" at the same index.
        # "targets[i]" is what we expect our network to predict given features[i]
        # features = np.hstack((np.array(states), np.array([u_effs]).T))
        features = torch.from_numpy(features[:-1, :]).float()

        # targets = np.array(states)
        targets = torch.from_numpy(targets[1:, :]).float()

        return features, targets


def plot_results(net, args, MyCart):
    """
    This function accepts RNN instance, arguments and CartPole instance.
    It runs one random experiment with CartPole,
    inputs the data into RNN and check how well RNN predicts CartPole state one time step ahead of time
    """
    # Reset the internal state of RNN cells, clear the output memory, etc.
    net.reset()

    # Generates ab CartPole  experiment and save its data
    test_set = Dataset(MyCart, args, train=False)  # Only experiment length is different for train=True/False

    # Format the experiment data
    # test_set[0] means that we take one random experiment, first on the list
    # The data will be however anyway generated on the fly and is in general not reproducible
    # TODO: Make data reproducable: set seed or find another solution
    features, targets = test_set[0]

    # Add empty dimension to fit the requirements of RNN input shape
    # (in fact we add dimension for batches - for testing we use only one batch)
    features = features.unsqueeze(0)

    # Convert Pytorch tensors to numpy matrices to inspect them - just for debugging purpose.
    # Variable explorers of IDEs are often not compatible with Pytorch format
    # features_np = features.detach().numpy()
    # targets_np = targets.detach().numpy()

    # Further modifying the input and output form to fit RNN requirements
    # If GPU available we send features to GPU
    if torch.cuda.is_available():
        features = features.float().cuda().transpose(0, 1)
        targets = targets.float()
    else:
        features = features.float().transpose(0, 1)
        targets = targets.float()

    # From features we extract control input and save it as a separate vector on the cpu
    u_effs = features[:, :, -1].cpu()
    # We shift it by one time step and double the last entry to keep the size unchanged
    u_effs = u_effs[1:]
    u_effs = np.append(u_effs, u_effs[-1])

    # Set the RNN in evaluation mode
    net = net.eval()
    # During several first time steps we let hidden layers adapt to the input data
    # train=False says that all the input should be used for initialization
    # -> we predict always only one time step ahead of time based on ground truth data
    predictions = net(features)

    # reformat the output of RNN to a form suitable for plotting the results
    # y_pred are prediction from RNN
    y_pred = predictions.squeeze().cpu().detach().numpy()
    # y_target are expected prediction from RNN, ground truth
    y_target = targets.squeeze().cpu().detach().numpy()

    # Get the time axes
    t = np.arange(0, y_target.shape[0]) * args.dt

    # Get position over time
    xp = y_pred[args.warm_up_len:, 0]
    xt = y_target[:, 0]

    # Get velocity over time
    vp = y_pred[args.warm_up_len:, 1]
    vt = y_target[:, 1]

    # get angle theta of the Pole
    tp = y_pred[args.warm_up_len:, 2] * 180.0 / np.pi  # t like theta
    tt = y_target[:, 2] * 180.0 / np.pi

    # Get angular velocity omega of the Pole
    op = y_pred[args.warm_up_len:, 3] * 180.0 / np.pi  # o like omega
    ot = y_target[:, 3] * 180.0 / np.pi

    # Create a figure instance
    fig, axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)  # share x axis so zoom zooms all plots

    # %matplotlib inline
    # Plot position
    axs[0].set_ylabel("Position (m)", fontsize=18)
    axs[0].plot(t, xt, 'k:', markersize=12, label='Ground Truth')
    axs[0].plot(t[args.warm_up_len:], xp, 'b', markersize=12, label='Predicted position')
    axs[0].tick_params(axis='both', which='major', labelsize=16)

    # Plot velocity
    axs[1].set_ylabel("Velocity (m/s)", fontsize=18)
    axs[1].plot(t, vt, 'k:', markersize=12, label='Ground Truth')
    axs[1].plot(t[args.warm_up_len:], vp, 'g', markersize=12, label='Predicted velocity')
    axs[1].tick_params(axis='both', which='major', labelsize=16)

    # Plot angle
    axs[2].set_ylabel("Angle (deg)", fontsize=18)
    axs[2].plot(t, tt, 'k:', markersize=12, label='Ground Truth')
    axs[2].plot(t[args.warm_up_len:], tp, 'c', markersize=12, label='Predicted angle')
    axs[2].tick_params(axis='both', which='major', labelsize=16)

    # Plot angular velocity
    axs[3].set_ylabel("Angular velocity (deg/s)", fontsize=18)
    axs[3].plot(t, ot, 'k:', markersize=12, label='Ground Truth')
    axs[3].plot(t[args.warm_up_len:], op, 'm', markersize=12, label='Predicted velocity')
    axs[3].tick_params(axis='both', which='major', labelsize=16)

    # Plot motor input command
    axs[4].set_ylabel("motor (N)", fontsize=18)
    axs[4].plot(t, u_effs, 'r', markersize=12, label='motor')
    axs[4].tick_params(axis='both', which='major', labelsize=16)

    # # Plot target position
    # axs[5].set_ylabel("position target", fontsize=18)
    # axs[5].plot(self.MyCart.dict_history['time'], self.MyCart.dict_history['target_position'], 'k')
    # axs[5].tick_params(axis='both', which='major', labelsize=16)

    axs[4].set_xlabel('Time (s)', fontsize=18)

    plt.show()
    # Save figure to png
    fig.savefig('my_figure.png')
    Image('my_figure.png')
