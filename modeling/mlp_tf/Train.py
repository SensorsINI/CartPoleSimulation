# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 06:21:32 2020
@author: Marcin
"""
# Various
# Custom functions
#Test string
from modeling.mlp_tf.utilis_mlp import *
# Parameters of NN
from modeling.mlp_tf.ParseArgs import args

import tensorflow.keras as keras

import timeit

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

    # Network architecture:
    nn_name = args.nn_name
    inputs_list = args.inputs_list
    outputs_list = args.outputs_list

    load_nn = args.load_nn  # If specified this is the name of pretrained NN which should be loaded
    path_save = args.path_save

    # Create nn instance and update lists of input, outputs and its name (if pretraind net loaded)
    net, nn_name, inputs_list, outputs_list, normalization_info \
        = create_nn_instance(args, nn_name, inputs_list, outputs_list, load_nn, path_save)
    columns_list = list(set(inputs_list).union(set(outputs_list)))

    # Create log for this NN and determine its full name
    if net.nn_full_name is None:
        nn_full_name = create_log_file(nn_name, inputs_list, outputs_list, path_save)
        net.nn_full_name = nn_full_name
    else:
        nn_full_name = net.nn_full_name

    ########################################################
    # Create Dataset
    ########################################################

    train_dfs, _ = load_data(args, args.train_file_name, columns_list=columns_list)

    if normalization_info is None:
        normalization_info =  calculate_normalization_info(train_dfs, args.path_save, nn_full_name)

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
            # if epoch==4:
            if epoch >= 0:
                plot_string = 'This is the network after {} training epoch(s)'.format(epoch +1)
                plot_results(net=net, args=args, dataset=test_set,
                             comment=plot_string,
                             save=True,
                             closed_loop_enabled=True,
                             exp_len=120//args.downsampling)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=args.path_save + nn_full_name + '.ckpt',
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=False)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=1,
        min_lr=0.0001,
        verbose=2
    )

    history = net.fit(
        train_set,
        epochs=args.num_epochs,
        verbose=1,
        shuffle=False,
        validation_data=test_set,
        callbacks=[CustomCallback(), model_checkpoint_callback, reduce_lr],
    )

    net.save_weights(args.path_save + nn_full_name + '.ckpt')

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
    import os.path, time
    file = os.path.realpath(__file__)
    print("Training script last modified: %s" % time.ctime(os.path.getmtime(file)))
    # warm_up_lens = [30, 5, 20, 5, 30, 5, 40, 5, 20, 5]

    # warm_up_lens = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 55, 50, 45, 40, 45, 40, 35, 30, 25, 20, 15, 10, 5]
    # for warm_up_len_idx in range(len(warm_up_lens)):
    #     print('We are at iteration: {}'.format(warm_up_len_idx))
    #     args.warm_up_len = warm_up_lens[warm_up_len_idx]
    #     args.exp_len = warm_up_lens[warm_up_len_idx]+5
    #     time_to_accomplish = train_network()
    #     print('Total time of training the network: ' + str(time_to_accomplish))

    time_to_accomplish = train_network()
    print('Total time of training the network: ' + str(time_to_accomplish))
