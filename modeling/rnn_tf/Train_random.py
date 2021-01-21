# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 06:21:32 2020

@author: Marcin
"""


# Various

# Custom functions
from modeling.rnn_tf.utilis_rnn import *
# Parameters of RNN
from modeling.rnn_tf.ParseArgs import args

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
    net, rnn_name, inputs_list, outputs_list, normalization_info \
        = create_rnn_instance(args, rnn_name, inputs_list, outputs_list, load_rnn, path_save)
    columns_list = list(set(inputs_list).union(set(outputs_list)))

    # Create log for this RNN and determine its full name
    if net.rnn_full_name is None:
        rnn_full_name = create_log_file(rnn_name, inputs_list, outputs_list, path_save)
        net.rnn_full_name = rnn_full_name
    else:
        rnn_full_name = net.rnn_full_name

    ########################################################
    # Create Dataset
    ########################################################


    train_set = DatasetRandom(args, inputs_list=inputs_list, outputs_list=outputs_list, number_of_batches=1000)
    test_set = DatasetRandom(args, inputs_list=inputs_list, outputs_list=outputs_list, number_of_batches=10)

    net.summary()
    net.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(0.001)
    )

    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # if epoch==4:
            if epoch >= 0:
                plot_string = 'This is the network after {} training epoch(s), warm_up={}'.format(epoch +1, args.warm_up_len)
                plot_results(net=net, args=args,
                             comment=plot_string,
                             save=True,
                             closed_loop_enabled=True,
                             exp_len=120//args.downsampling)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=args.path_save + rnn_full_name + '.ckpt',
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
    import os.path, time
    file = os.path.realpath(__file__)
    print("Training script last modified: %s" % time.ctime(os.path.getmtime(file)))
    # warm_up_lens = [30, 5, 20, 5, 30, 5, 40, 5, 20, 5]

    warm_up_lens = [20, 50, 20, 30, 20]

    # warm_up_lens = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 55, 50, 45, 40, 45, 40, 35, 30, 25, 20, 15, 10, 5]
    for warm_up_len_idx in range(len(warm_up_lens)):
        print('We are at iteration: {}'.format(warm_up_len_idx))
        args.warm_up_len = warm_up_lens[warm_up_len_idx]
        args.exp_len = warm_up_lens[warm_up_len_idx]+5
        time_to_accomplish = train_network()
        print('Total time of training the network: ' + str(time_to_accomplish))

    # time_to_accomplish = train_network()
    # print('Total time of training the network: ' + str(time_to_accomplish))
