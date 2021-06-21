# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""



import argparse
import glob

path_save = './save/'
TRAIN_file_name = glob.glob('./data/train/' + '*.csv')
VAL_file_name = glob.glob('./data/validate/' + '*.csv')


'''
FIXME: To tailor input list, output list and closed loop list according to cartpole
angleD_next, positionD_next = cartpole_ode(s, Q2u(Q))
'''
RNN_name = 'GRU-32H1-32H2'
# inputs_list = ['position', 'positionD', 'angle', 'angleD', 'u']
inputs_list = ['position', 'angle', 'u']
# outputs_list = ['angle', 'position', 'positionD', 'angleD']
outputs_list = ['angle', 'position']
# closed_loop_list = ['position', 'angle']
closed_loop_list = outputs_list

# inputs_list = ['position', 'positionD', 'angle', 'angleD', 'target_position']
# outputs_list = ['Q']
# closed_loop_list = []

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Defining the model
    parser.add_argument('--rnn_name', nargs='?', const=RNN_name, default=None, type=str,
                        help='Name defining the RNN.'
                             'It has to have the form:'
                             '(RNN type [GRU/LSTM])-(size first hidden layer)H1-(size second hidden layer)H2-...'
                             'e.g. GRU-64H1-64H2-32H3')

    parser.add_argument('--train_file_name', default=TRAIN_file_name, type=str,
                        help='File name of the recording to be used for training the RNN'
                             'e.g. oval_easy.csv ')
    parser.add_argument('--val_file_name', default=VAL_file_name, type=str,
                        help='File name of the recording to be used for validating the RNN'
                             'e.g. oval_easy_test.csv ')

    parser.add_argument('--inputs_list', nargs="?", default=None, const=inputs_list,
                        help='List of inputs to RNN')
    parser.add_argument('--outputs_list', nargs="?", default=None, const=outputs_list,
                        help='List of outputs from RNN')
    parser.add_argument('--close_loop_for', nargs='?', default=None, const=closed_loop_list,
                        help='In RNN forward function this features will be fed beck from output to input')

    parser.add_argument('--load_rnn', nargs='?', default=None, const='last', type=str,
                        help='Full name defining the RNN which should be loaded without .csv nor .pt extension'
                             'e.g. GRU-8IN-64H1-64H2-3OUT-1')

    parser.add_argument("--cheat_dt", action='store_true',
                        help="Give RNN during training a true (future) dt.")

    parser.add_argument('--wash_out_len', default=10, type=int, help='Number of timesteps for a warm-up sequence')
    parser.add_argument('--seq_len', default=50, type=int, help='Number of timesteps in a sequence')
    parser.add_argument('--downsampling', default=1, type=int,
                        help='Take every n-th point of callected dataset to make dt bigger')


    # Training parameters
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of epochs of training')
    parser.add_argument('--batch_size', default=64, type=int, help='Size of a batch')
    parser.add_argument('--seed', default=1873, type=int, help='Set seed for reproducibility')
    parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers to produce data from data loaders')
    parser.add_argument('--PATH_TO_EXPERIMENT_RECORDINGS', default=path_save, type=str,
                        help='Path where to save/ from where to load models')

    parser.add_argument('--normalize', default=True, type=bool, help='Make all data between 0 and 1')

    args = parser.parse_args()
    if args.inputs_list is not None:
        args.inputs_list = sorted(args.inputs_list)
    if args.outputs_list is not None:
        args.outputs_list = sorted(args.outputs_list)
    return args

