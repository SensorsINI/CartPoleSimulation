# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""



import argparse
import glob

net_name = 'Dense-16H1-16H2'

# Path to trained models and their logs
PATH_TO_MODELS = './SI_Toolkit/TF/Models/'

PATH_TO_NORMALIZATION_INFO = './SI_Toolkit/NormalizationInfo/' + 'Dataset-1-norm.csv'

# The following paths to dictionaries may be replaced by the list of paths to data files.
TRAINING_FILES = './ExperimentRecordings/Dataset-1/Train/'
VALIDATION_FILES = './ExperimentRecordings/Dataset-1/Validate/'
TEST_FILES = './ExperimentRecordings/Dataset-1/Test/'





# region Set inputs and outputs

# For training closed loop dynamics model
inputs = ['Q', 'angle_sin', 'angle_cos', 'angleD', 'position', 'positionD']
outputs = ['angle_sin', 'angle_cos', 'angleD', 'position', 'positionD']

# For training open loop dynamics model
# inputs = ['position', 'positionD', 'angle_sin', 'angle_cos', 'angleD']
# outputs = inputs_list

# For training of RNN imitating MPC
# inputs = ['position', 'positionD', 'angle', 'angleD', 'target_position']
# outputs = ['Q']

# endregion

# region Set which features are fed in closed loop (testing only)
closed_loop_list = outputs
# closed_loop_list = []
# endregion

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.')

    # Defining the model
    parser.add_argument('--net_name', default=net_name, type=str,
                        help='Name defining the network.'
                             'It has to have the form:'
                             '(RNN type [GRU/LSTM]/Dense)-(size first hidden layer)H1-(size second hidden layer)H2-...'
                             'e.g. GRU-64H1-64H2-32H3')

    parser.add_argument('--training_files', default=TRAINING_FILES, type=str,
                        help='File name of the recording to be used for training the RNN'
                             'e.g. oval_easy.csv ')
    parser.add_argument('--validation_files', default=VALIDATION_FILES, type=str,
                        help='File name of the recording to be used for validating the RNN'
                             'e.g. oval_easy_test.csv ')
    parser.add_argument('--test_files', default=TEST_FILES, type=str,
                        help='File name of the recording to be used for validating the RNN'
                             'e.g. oval_easy_test.csv ')

    parser.add_argument('--inputs', default=inputs,
                        help='List of inputs to RNN')
    parser.add_argument('--outputs', default=outputs,
                        help='List of outputs from RNN')
    parser.add_argument('--close_loop_for', default=closed_loop_list,
                        help='In RNN forward function this features will be fed beck from output to input')

    # Only valid for graphical testing:
    parser.add_argument('--test_len', default=50, type=int,
                        help='For graphical testing only test_len samples from first test file is taken.')
    parser.add_argument('--test_start_idx', default=100, type=int, help='Indicates from which point data from test file should be taken.')
    parser.add_argument('--test_max_horizon', default=5, type=int,
                        help='Indicates prediction horizon for testing.')

    # Training only:
    parser.add_argument('--wash_out_len', default=10, type=int, help='Number of timesteps for a wash-out sequence')
    parser.add_argument('--post_wash_out_len', default=50, type=int,
                        help='Number of timesteps after wash-out sequence (this is used to calculate loss)')

    # Training parameters
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs of training')
    parser.add_argument('--batch_size', default=256, type=int, help='Size of a batch')
    parser.add_argument('--seed', default=1873, type=int, help='Set seed for reproducibility')
    parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')

    parser.add_argument('--path_to_models', default=PATH_TO_MODELS, type=str,
                        help='Path where to save/ from where to load models')

    parser.add_argument('--path_to_normalization_info', default=PATH_TO_NORMALIZATION_INFO, type=str,
                        help='Path where the cartpole data is saved')

    parser.add_argument('--on_fly_data_generation', default=False, type=bool,
                        help='Generate data for training during training, instead of loading previously saved data')
    parser.add_argument('--normalize', default=True, type=bool, help='Make all data between 0 and 1')

    args = parser.parse_args()

    # Make sure that the provided lists of inputs and outputs are in alphabetical order
    if args.inputs is not None:
        args.inputs = sorted(args.inputs)
    if args.outputs is not None:
        args.outputs = sorted(args.outputs)
    return args

