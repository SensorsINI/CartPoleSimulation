# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 06:21:32 2020

@author: Marcin

The file generates an CartPole experiments and loads pretrained RNN network. It feeds
"""

import torch
import torch.utils.data.dataloader

import collections

from modeling.rnn_tf.utilis_rnn import *
# Parameters of RNN
from modeling.rnn_tf.ParseArgs import args as my_args

filepath = './data/data_rnn_big.csv'
# filepath = './data/dt_variable_test-2.csv'

# Get arguments as default or from terminal line
args = my_args()
# Print the arguments
print(args.__dict__)

exp_len = 5000//args.downsampling

MULTIPLE_PICTURES = False


def test_network():

    """
    This function create RNN instance based on parameters saved on disc and also creates the CartPole instance.
    The actual work of evaluation prediction results is done in plot_results function
    """

    # Network architecture:
    rnn_name = args.rnn_name
    inputs_list = args.inputs_list
    outputs_list = args.outputs_list

    load_rnn = args.load_rnn  # If specified this is the name of pretrained RNN which should be loaded
    path_save = args.path_save

    # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
    net, rnn_name, inputs_list, outputs_list \
        = create_rnn_instance(args=args, rnn_name=rnn_name,
                              inputs_list=inputs_list, outputs_list=outputs_list,
                              load_rnn=load_rnn, path_save=path_save)
    title = 'Testing RNN: {}'.format(rnn_name)
    if MULTIPLE_PICTURES:
        for i in range(exp_len//20):
            close_loop_idx = (exp_len//4)+i*10
            plot_results(net=net, args=args, dataset=None, filepath=filepath, exp_len=exp_len,
                         comment=title,
                         inputs_list=inputs_list, outputs_list=outputs_list, save=True,
                         closed_loop_enabled=True, close_loop_idx=close_loop_idx)
    else:
        plot_results(net=net, args=args, dataset=None, filepath=filepath, exp_len=exp_len,
                     comment=title,
                     inputs_list=inputs_list, outputs_list=outputs_list, save=True,
                     closed_loop_enabled=True, close_loop_idx=exp_len//2)


if __name__ == '__main__':
    test_network()
