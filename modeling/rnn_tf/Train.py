# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 06:21:32 2020

@author: Marcin
"""


import timeit

# Various

# Custom functions
from modeling.rnn_tf.utilis_rnn import *
# Parameters of RNN
from modeling.rnn_tf.ParseArgs import args

import tensorflow as tf
import tensorflow.keras as keras

print('')

args = args()
print(args.__dict__)


# Uncomment the @profile(precision=4) to get the report on memory usage after the training
# Warning! It may affect performance. I would discourage you to use it for long training tasks
# @profile(precision=4)
def train_network():
    print('')
    print('')
    # Start measuring time - to evaluate performance of the training function
    start = timeit.default_timer()

    # Set seeds
    set_seed(args)

    # Make folders if not yet exist
    try:
        os.makedirs('save_tf')
    except FileExistsError:
        pass

    # Save relevant arguments from args and set hardcoded arguments
    lr = args.lr  # learning rate
    batch_size = args.batch_size  # Mini-batch size
    num_epochs = args.num_epochs  # Number of epochs to train the network
    exp_len = args.exp_len

    # Network architecture:
    rnn_name = args.rnn_name
    inputs_list = args.inputs_list
    outputs_list = args.outputs_list

    load_rnn = args.load_rnn  # If specified this is the name of pretrained RNN which should be loaded
    path_save = args.path_save

    # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
    net, rnn_name, inputs_list, outputs_list \
        = create_rnn_instance(args, rnn_name, inputs_list, outputs_list, load_rnn, path_save)
    columns_list = list(set(inputs_list).union(set(outputs_list)))

    # Create log for this RNN and determine its full name
    rnn_full_name = create_log_file(rnn_name, inputs_list, outputs_list, path_save)
    net.rnn_full_name = rnn_full_name

    ########################################################
    # Create Dataset
    ########################################################

    train_dfs, _ = load_data(args, args.train_file_name, columns_list=columns_list)

    normalization_info =  calculate_normalization_info(train_dfs, args.path_save, rnn_full_name)

    test_dfs, time_axes_test = load_data(args, args.val_file_name, columns_list=columns_list)

    # Take just one data set
    train_dfs_norm = normalize_df(train_dfs, normalization_info)
    test_dfs_norm = normalize_df(test_dfs, normalization_info)

    train_set = Dataset(train_dfs_norm, args, inputs_list=inputs_list, outputs_list=outputs_list)
    test_set = Dataset(test_dfs_norm, args, time_axes=time_axes_test, shuffle=False, inputs_list=inputs_list, outputs_list=outputs_list)
    print('Number of samples in training set: {}'.format(train_set.number_of_samples))
    print('The training sets sizes are: {}'.format(train_set.df_lengths))
    print('Number of samples in validation set: {}'.format(test_set.number_of_samples))
    print('')

    del train_dfs, test_dfs, train_dfs_norm, test_dfs_norm

    net.summary()
    net.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(0.001)
    )

    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            plot_string = 'This is the network after {} training epoch(s)'.format(epoch +1)
            plot_results(net=net, args=args, dataset=test_set,
                         comment=plot_string,
                         save=True,
                         closed_loop_enabled=True,
                         exp_len = 500//args.downsampling)

    history = net.fit(
        train_set,
        epochs=args.num_epochs,
        verbose=1,
        shuffle=False,
        validation_data=test_set,
        callbacks=[CustomCallback()],
    )

    net.save_weights(args.path_save + rnn_full_name + '.ckpt')

    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

    # Calculate the total time it took to run the function
    stop = timeit.default_timer()
    total_time = stop - start

    # Return the total time it took to run the function
    return total_time


if __name__ == '__main__':

    time_to_accomplish = train_network()
    print('Total time of training the network: ' + str(time_to_accomplish))
