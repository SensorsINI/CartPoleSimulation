# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Marcin
"""



import argparse
import glob

# Path to trained models and their logs
PATH_TO_MODELS = './Modeling/TF/Models/'

PATH_TO_NORMALIZATION_INFO = './Modeling/NormalizationInfo/' + 'NI_2021-03-04_17-29-53.csv'

# The following paths to dictionaries may be replaced by the list of paths to data files.
TRAINING_FILES = './ExperimentRecordings/Train/'
VALIDATION_FILES = './ExperimentRecordings/Validate/'
TEST_FILES = './ExperimentRecordings/Test/'

# endregion

net_name = 'Dense-64H1-64H2'

# region Set inputs and outputs

# For training closed loop dynamics model
inputs = ['Q', 's.angle.sin', 's.angle.cos', 's.angleD', 's.position', 's.positionD']
outputs = ['s.angle.sin', 's.angle.cos', 's.angleD', 's.position', 's.positionD']

# For training open loop dynamics model
# inputs = ['s.position', 's.positionD', 's.angle.sin', 's.angle.cos', 's.angleD']
# outputs = inputs_list

# For training of RNN imitating MPC
# inputs = ['s.position', 's.positionD', 's.angle', 's.angleD', 'target_position']
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

    # Experiment length - number of time steps to take for testing
    parser.add_argument('--test_len', default=11, type=int, help='Number of time steps to take for testing')

    # Training only:
    parser.add_argument('--wash_out_len', default=10, type=int, help='Number of timesteps for a wash-out sequence')
    parser.add_argument('--post_wash_out_len', default=10, type=int,
                        help='Number of timesteps after wash-out sequence (this is used to calculate loss)')

    # Training parameters
    parser.add_argument('--num_epochs', default=15, type=int, help='Number of epochs of training')
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

