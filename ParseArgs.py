# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""



import argparse
import numpy as np


savepath = './save/' + 'MyNet' + '.pt'
savepathPre = './save/' + 'MyNetPre' + '.pt'

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.')


    parser.add_argument('--dt',             default=2.0,        type=float,  help='Time interval of a time step in ms')
    parser.add_argument('--warm_up_len',    default=256,         type=int,    help='Number of timesteps for a warm-up sequence')
    parser.add_argument('--exp_len_train',  default=256+256+1,   type=int,    help='Number of timesteps for in a full experiment (warm-up+interative phase+1)')
    parser.add_argument('--exp_len_test',   default=2e3,        type=int,    help='Number of timesteps for in a full experiment test phase')
    parser.add_argument('--epoch_len',      default=2e4,        type=int,    help='How many sine waves are fed in NN during one epoch of training')
    parser.add_argument('--num_epochs',     default=10,         type=int,    help='Number of epochs of training')
    parser.add_argument('--batch_size',     default=128,         type=int,    help='Size of a batch')
    parser.add_argument('--epochs_per_win', default=15,         type=int,    help='Size of a batch')
    
    parser.add_argument('--lr',             default=1.0e-4,     type=float,  help='Learning rate')
    parser.add_argument('--h1_size',        default=128,        type=int,    help='First hidden layer size')
    parser.add_argument('--h2_size',        default=128,        type=int,    help='Second hindden layer size')
    
    parser.add_argument('--num_workers',    default=1,          type=int,    help='Number of workers to produce data from data loaders')
    
    parser.add_argument('--savepath',       default=savepath,   type=str,    help='Path where to save currently trained model')
    parser.add_argument('--savepathPre',    default=savepathPre,type=str,    help='Path from where to load a pretrained model')
    parser.add_argument('--seed',           default=1873,       type=int,    help='Set seed for reproducibility')

    args = parser.parse_args()
    return args  