# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 06:21:32 2020

@author: Marcin

The file generates an CartPole experiments and loads pretrained RNN network. It feeds
"""

import torch
import torch.utils.data.dataloader

import collections

from utilis import get_device, Sequence, plot_results
import ParseArgs
from CartClass import Cart

# Check if GPU is available. If yes device='cuda:0' if not device='cpu'
device = get_device()

# Get arguments as default or from terminal line
args = ParseArgs.args()
# Print the arguments
print(args.__dict__)


def test_network():

    """
    This function create RNN instance based on parameters saved on disc and also creates the CartPole instance.
    The actual work of evaluation prediction results is done in plot_results function
    """

    # Create CartPole instance (keeps the dynamical equations describing CartPole and simulation methods)
    MyCart = Cart()
    # Create instance of RNN
    net = Sequence(args)

    # If a pretrained model exists load the parameters from disc
    pre_trained_model = torch.load(args.savepathPre, map_location=torch.device('cpu'))
    print("Loading Model: ", args.savepathPre)

    # Load the parameters into created RNN instance
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
    net.load_state_dict(new_state_dict)

    # Generate a random experiment with CartPole and compare the the true CartPole state with prediction of RNN
    plot_results(net, args, MyCart)


if __name__ == '__main__':
    test_network()
