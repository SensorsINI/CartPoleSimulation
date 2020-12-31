# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""



import argparse

path_save = './save_tf/'
TRAIN_file_name = [
                   './data/data_around_zero.csv',
                   './data/data_rnn_big-2.csv',
                   './data/data_rnn_big-1.csv',
                   './data/data_rnn_very_big.csv',
                   './data/data_rnn_very_big-1.csv',
                   './data/data_rnn-1.csv',
                   './data/data_rnn-2.csv',
                   ]
VAL_file_name = './data/data_rnn_big.csv'

# For training with variable dt
# TRAIN_file_name = ['./data/dt_variable-0.csv',
#                    './data/dt_variable-1.csv',
#                    './data/dt_variable-2.csv',
#                    './data/dt_variable-3.csv',
#                    './data/dt_variable-4.csv',
#                    './data/dt_variable-5.csv',
#                    './data/dt_variable-6.csv',
#                    './data/dt_variable-7.csv',
#                    './data/dt_variable-8.csv',
#                    './data/dt_variable-9.csv',
#                    ]
#
# VAL_file_name = './data/dt_variable_test-0.csv'

'''
FIXME: To tailor input list, output list and closed loop list according to cartpole
angleD_next, positionD_next = cartpole_ode(p, s, Q2u(Q,p))
'''
RNN_name = 'GRU-8H1-8H2'
# inputs_list = ['s.position', 's.positionD', 's.angle', 's.angleD', 'u']
# inputs_list = ['s.position', 's.positionD', 's.angle', 's.angleD', 'u', 'target_position']
# inputs_list = ['s.position', 's.angle', 'u']
inputs_list = ['s.position', 's.angle', 'u', 'target_position']
# outputs_list = ['s.angle', 's.position', 's.positionD', 's.angleD']
outputs_list = ['s.angle', 's.position']
closed_loop_list = ['s.position', 's.angle']
# closed_loop_list = outputs_list

# inputs_list = ['s.position', 's.positionD', 's.angle', 's.angleD', 'target_position']
# outputs_list = ['Q']
# closed_loop_list = []

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.')

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

    # Exp len>warm_up_len!f
    parser.add_argument('--exp_len', default=1024, type=int, help='Return after cosuming this number of points')
    parser.add_argument('--warm_up_len', default=10, type=int, help='Number of timesteps for a warm-up sequence')
    parser.add_argument('--downsampling', default=10, type=int, help='Take every n-th point of callected dataset to make dt bigger')

    # Training parameters
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of epochs of training')
    parser.add_argument('--batch_size', default=64, type=int, help='Size of a batch')
    parser.add_argument('--seed', default=1873, type=int, help='Set seed for reproducibility')
    parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers to produce data from data loaders')
    parser.add_argument('--path_save', default=path_save, type=str,
                        help='Path where to save/ from where to load models')

    parser.add_argument('--normalize', default=True, type=bool, help='Make all data between 0 and 1')

    args = parser.parse_args()
    if args.inputs_list is not None:
        args.inputs_list = sorted(args.inputs_list)
    if args.outputs_list is not None:
        args.outputs_list = sorted(args.outputs_list)
    return args

