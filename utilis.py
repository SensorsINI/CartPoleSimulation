# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:28:34 2020

@author: Marcin
"""

from CartClass import Cart
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from IPython.display import Image

import matplotlib.pyplot as plt


def get_device():
    """
    Small function to correctly send data to GPU or CPU depending what is available
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


class Sequence(nn.Module):
    """"
    Our RNN class.
    """

    def __init__(self, args):
        super(Sequence, self).__init__()
        """Initialization of an RNN instance"""

        # Check if GPU is available. If yes device='cuda:0' if not device='cpu'
        self.device = get_device()
        # Save args (default of from terminal line)
        self.args = args
        # Initialize RNN layers
        self.gru1 = nn.GRUCell(5, args.h1_size)  # RNN accepts 5 inputs: CartPole state (4) and control input at time t
        self.gru2 = nn.GRUCell(args.h1_size, args.h2_size)
        self.linear = nn.Linear(args.h2_size, 4)  # RNN out
        # Count data samples (=time steps)
        self.sample_counter = 0
        # Declaration of the variables keeping internal state of GRU hidden layers
        self.h_t = None
        self.h_t2 = None
        # Variable keeping the most recent output of RNN
        self.output = None
        # List storing the history of RNN outputs
        self.outputs = []

        # Send the whole RNN to GPU if available, otherwise send it to CPU
        self.to(self.device)

    def forward(self, predict_len: int, input, terminate=False):
        """
        Predicts future CartPole states IN "CLOSED LOOP"
        (at every time step prediction for the next time step is done based on CartPole state
        resulting from the previous prediction; only control input is provided from the ground truth at every step)
        """
        # From input to RNN (CartPole state + control input) get control input
        u_effs = input[:, :, -1]
        # For number of time steps given in predict_len predict the state of the CartPole
        # At every time step RNN get as its input the ground truth value of control input
        # BUT instead of the ground truth value of CartPole state
        # it gets the result of the prediction for the last time step
        for i in range(predict_len):
            # Concatenate the previous prediction and current control input to the input to RNN for a new time step
            input_t = torch.cat((self.output, u_effs[self.sample_counter, :].unsqueeze(1)), 1)
            # Propagate input through RNN layers
            self.h_t = self.gru1(input_t, self.h_t)
            self.h_t2 = self.gru2(self.h_t, self.h_t2)
            self.output = self.linear(self.h_t2)
            # Append the output to the outputs history list
            self.outputs += [self.output]
            # Count number of samples
            self.sample_counter = self.sample_counter + 1

        # if terminate=True transform outputs history list to a Pytorch tensor and return it
        # Otherwise store the outputs internally as a list in the RNN instance
        if terminate:
            self.outputs = torch.stack(self.outputs, 1)
            return self.outputs

    def reset(self):
        """
        Reset the network (not the weights!)
        """
        self.sample_counter = 0
        self.h_t = None
        self.h_t2 = None
        self.output = None
        self.outputs = []

    def initialize_sequence(self, input, train=True):

        """
        Predicts future CartPole states IN "OPEN LOOP"
        (at every time step prediction for the next time step is done based on the true CartPole state)
        """

        # If in training mode we will only run this function during the first several (args.warm_up_len) data samples
        # Otherwise we run it for the whole input
        if train:
            starting_input = input[:self.args.warm_up_len, :, :]
        else:
            starting_input = input

        # Initialize hidden layers
        self.h_t = torch.zeros(starting_input.size(1), self.args.h1_size, dtype=torch.float).to(self.device)
        self.h_t2 = torch.zeros(starting_input.size(1), self.args.h2_size, dtype=torch.float).to(self.device)

        # The for loop takes the consecutive time steps from input plugs them into RNN and save the outputs into a list
        # THE NETWORK GETS ALWAYS THE GROUND TRUTH, THE REAL STATE OF THE CARTPOLE, AS ITS INPUT
        # IT PREDICTS THE STATE OF THE CARTPOLE ONE TIME STEP AHEAD BASED ON TRUE STATE NOW
        for i, input_t in enumerate(starting_input.chunk(starting_input.size(0), dim=0)):
            self.h_t = self.gru1(input_t.squeeze(0), self.h_t)
            self.h_t2 = self.gru2(self.h_t, self.h_t2)
            self.output = self.linear(self.h_t2)
            self.outputs += [self.output]
            self.sample_counter = self.sample_counter + 1

        # In the train mode we want to continue appending the outputs by calling forward function
        # The outputs will be saved internally in the network instance as a list
        # Otherwise we want to transform outputs list to a tensor and return it
        if not train:
            self.outputs = torch.stack(self.outputs, 1)
            return self.outputs


def Generate_Experiment(MyCart, exp_len=64 + 640 + 1, dt=0.002):
    """
    This function runs a random CartPole experiment
    and returns the history of CartPole states, control inputs and desired cart position
    :param MyCart: instance of CartPole containing CartPole dynamics
    :param exp_len: How many time steps should the experiment have
                (default: 64+640+1 this format is used as it can )
    """

    # Set CartPole in the right (automatic control) mode
    MyCart.mode = 1  # 1 - you are controlling with LQR

    # Initialize Variables
    states = []
    u_effs = []
    target_positions = []

    # Generate new random function returning desired target position of the cart
    MyCart.dt = dt
    MyCart.random_length = exp_len
    MyCart.N = 10  # Complexity of generated target position track
    MyCart.Generate_Random_Trace_Function()

    # Randomly set the initial state

    MyCart.time_total = 0.0

    MyCart.CartPosition = np.random.uniform(low=-MyCart.HalfLength / 2.0,
                                            high=MyCart.HalfLength / 2.0)
    MyCart.CartPositionD = np.random.uniform(low=-10.0,
                                             high=10.0)
    MyCart.angle = np.random.uniform(low=-17.5 * (np.pi / 180.0),
                                     high=17.5 * (np.pi / 180.0))
    MyCart.angleD = np.random.uniform(low=-15.5 * (np.pi / 180.0),
                                      high=15.5 * (np.pi / 180.0))

    MyCart.ueff = np.random.uniform(low=-0.9 * MyCart.umax,
                                    high=0.9 * MyCart.umax)

    # Target position at time 0 (should be always 0)
    MyCart.PositionTarget = MyCart.random_track_f(MyCart.time_total)  # = 0
    # Constrain target position
    if MyCart.PositionTarget > 0.8 * MyCart.HalfLength:
        MyCart.PositionTarget = 0.8 * MyCart.HalfLength
    elif MyCart.PositionTarget < -0.8 * MyCart.HalfLength:
        MyCart.PositionTarget = -0.8 * MyCart.HalfLength

    # Run the CartPole experiment for number of time
    for i in range(int(exp_len)):

        # Start by incrementing total time
        MyCart.time_total = i * MyCart.dt
        # Print an error message if it runs already to long (should stop before)
        if MyCart.time_total > MyCart.t_max_pre:
            MyCart.time_total = MyCart.t_max_pre
            print('ERROR: It seems the experiment is running too long...')

        # Calculate acceleration, angular acceleration, velocity, angular velocity, position and angle
        # for this new time_step
        MyCart.Equations_of_motion()

        # Determine the dimensionales [-1,1] value of the motor power Q
        MyCart.Update_Q()

        # Calculate the force created by the motor
        MyCart.Update_ueff()

        # Get the new value of desired cart position and constrain it
        MyCart.PositionTarget = MyCart.random_track_f(MyCart.time_total)
        if MyCart.PositionTarget > 0.8 * MyCart.HalfLength:
            MyCart.PositionTarget = 0.8 * MyCart.HalfLength
        elif MyCart.PositionTarget < -0.8 * MyCart.HalfLength:
            MyCart.PositionTarget = -0.8 * MyCart.HalfLength

        # Save all the new cart state variables you just updated
        state = (MyCart.CartPosition,
                 MyCart.CartPositionD,
                 MyCart.angle,
                 MyCart.angleD)

        # Save current cart state, control input and target position to a corresponding list
        states.append(state)
        u_effs.append(MyCart.ueff)
        target_positions.append(MyCart.PositionTarget)

    # After generating experiment finished, return states, control input and target_positions history
    return states, u_effs, target_positions


class Dataset(data.Dataset):
    """
    This is a Dataset class providing a proper data format for Pytorch applications
    It inherits from the standard Pytorch dataset class
    """

    def __init__(self, MyCart, args, train=True):
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
        states, u_effs, target_positions = Generate_Experiment(MyCart=self.MyCart,  # MyCart contain CartPole dynamics
                                                               exp_len=self.exp_len,
                                                               # How many data points should be generated
                                                               dt=self.dt)  # Simulation time step size

        # "features" is the array of inputs to the RNN, it consists of states of the CartPole and control input
        # "targets" is the array of CartPole states one time step ahead of "features" at the same index.
        # "targets[i]" is what we expect our network to predict given features[i]
        features = np.hstack((np.array(states), np.array([u_effs]).T))
        features = torch.from_numpy(features[:-1, :]).float()

        targets = np.array(states)
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
    predictions = net.initialize_sequence(features, train=False)

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
    # axs[5].plot(self.MyCart.dict_history['time'], self.MyCart.dict_history['PositionTarget'], 'k')
    # axs[5].tick_params(axis='both', which='major', labelsize=16)

    axs[4].set_xlabel('Time (s)', fontsize=18)

    plt.show()
    # Save figure to png
    fig.savefig('my_figure.png')
    Image('my_figure.png')
