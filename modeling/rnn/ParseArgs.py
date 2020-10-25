# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""



import argparse
import numpy as np


savepath = './save/' + 'MyNet' + '.pt'
savepathPre = './save/' + 'MyNetPre' + '.pt'

'''
FIXME: To tailor input list, output list and closed loop list according to cartpole
angleD_next, positionD_next = cartpole_ode(p, s, Q2u(Q,p))
'''
RNN_name = 'GRU-256H1-256H2'
inputs_list = ['target_position','s.position', 's.positionD', 's.positionDD', 's.angle', 's.angleD', 's.angleDD'] #FIXME : This is definately incorrect
outputs_list = ['angleD_next', 'positionD_next']
closed_loop_list = ['angleD', 'positionD']

def args():
    parser = argparse.ArgumentParser(description='Train a RNN for cartpole model prediction.')


    parser.add_argument('--dt',             default=10.0,        type=float,  help='Time interval of a time step in ms')
    parser.add_argument('--warm_up_len',    default=32,         type=int,    help='Number of timesteps for a warm-up sequence')
    parser.add_argument('--exp_len_train',  default=32+32+1,   type=int,    help='Number of timesteps for in a full experiment (warm-up+interative phase+1)')
    parser.add_argument('--exp_len_test',   default=2e3,        type=int,    help='Number of timesteps for in a full experiment test phase')
    parser.add_argument('--epoch_len',      default=2e3,        type=int,    help='How many sine waves are fed in NN during one epoch of training')
    parser.add_argument('--num_epochs',     default=3,         type=int,    help='Number of epochs of training')
    parser.add_argument('--batch_size',     default=128,         type=int,    help='Size of a batch')
    parser.add_argument('--epochs_per_win', default=15,         type=int,    help='Size of a batch')
    
    parser.add_argument('--lr',             default=1.0e-4,     type=float,  help='Learning rate')

    '''
    FIXME : We don't want h1_size, etc as we have this information in RNN name
    '''
    parser.add_argument('--h1_size',        default=128,        type=int,    help='First hidden layer size')
    parser.add_argument('--h2_size',        default=128,        type=int,    help='Second hindden layer size')
    
    parser.add_argument('--num_workers',    default=1,          type=int,    help='Number of workers to produce data from data loaders')
    
    parser.add_argument('--savepath',       default=savepath,   type=str,    help='Path where to save currently trained model')
    parser.add_argument('--savepathPre',    default=savepathPre,type=str,    help='Path from where to load a pretrained model')
    parser.add_argument('--seed',           default=1873,       type=int,    help='Set seed for reproducibility')

    parser.add_argument('--rnn_name', nargs='?', const=RNN_name, default=None, type=str,
                        help='Name defining the RNN.'
                             'It has to have the form:'
                             '(RNN type [GRU/LSTM])-(size first hidden layer)H1-(size second hidden layer)H2-...'
                             'e.g. GRU-64H1-64H2-32H3')
    parser.add_argument('--load_rnn', nargs='?', default=None, const='last', type=str,
                        help='Full name defining the RNN which should be loaded without .csv nor .pt extension'
                             'e.g. GRU-8IN-64H1-64H2-3OUT-1')

    parser.add_argument('--inputs_list', nargs="?", default=None, const=inputs_list,
                        help='List of inputs to RNN')
    parser.add_argument('--outputs_list', nargs="?", default=None, const=outputs_list,
                        help='List of outputs from RNN')
    parser.add_argument('--close_loop_for', nargs='?', default=None, const=closed_loop_list,
                        help='In RNN forward function this features will be fed beck from output to input')
    parser.add_argument('--load_rnn', nargs='?', default=None, const='last', type=str,
                        help='Full name defining the RNN which should be loaded without .csv nor .pt extension'
                             'e.g. GRU-8IN-64H1-64H2-3OUT-1')


    args = parser.parse_args()
    return args  