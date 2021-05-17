# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 06:21:32 2020

@author: Marcin
"""

try:
    import nni
except ModuleNotFoundError:
    print('Module nni not found - only needed to run training with NNI Framework')

import matplotlib.pyplot as plt

from tensorflow import keras

import os

import timeit
from warnings import warn as warning

# "Command line" parameters
from SI_Toolkit.TF.Parameters import args

# Custom functions
from SI_Toolkit.TF.TF_Functions.Initialization import set_seed, create_full_name, create_log_file, get_net_and_norm_info
from SI_Toolkit.TF.TF_Functions.Loss import loss_msr_sequence_customizable
from SI_Toolkit.TF.TF_Functions.Dataset import Dataset, DatasetRandom
from SI_Toolkit.load_and_normalize import load_data, normalize_df, \
    get_sampling_interval_from_datafile, get_paths_to_datafiles

# region Import and print "command line" arguments
print('')
a = args()  # 'a' like arguments
print(a.__dict__)
print('')
# endregion

# Uncomment the @profile(precision=4) to get the report on memory usage after the training
# Warning! It may affect performance. I would discourage you to use it for long training tasks
# @profile(precision=4)
def train_network(nni_parameters=None):
    # region Start measuring time - to evaluate performance of the training function
    start = timeit.default_timer()
    # endregion

    # region If NNI enabled load new parameters
    if nni_parameters is not None:
        a.net_name = 'GRU-' + str(nni_parameters['h1']) + 'H1-' + str(nni_parameters['h2']) + 'H2'
        a.wash_out_len = int(nni_parameters['wash_out_len'])
    # endregion

    # region Set seeds to make experiment reproducible
    set_seed(a)
    # endregion

    # region Make folder to keep trained models and their logs if not yet exist
    try:
        os.makedirs(a.path_to_models[:-1])
    except FileExistsError:
        pass
    # endregion

    net, net_info, normalization_info = get_net_and_norm_info(a)

    # Create a copy of the network suitable for inference (stateful and with sequence length one)
    net_for_inference, net_for_inference_info, normalization_info = \
        get_net_and_norm_info(a, time_series_length=a.test_len,
                              batch_size=1, stateful=True)

    # Create new full name for the pretrained net
    create_full_name(net_info, a.path_to_models)

    # а is an "argument"
    # It must contain:
    # path to models
    # information about paths for:
    #                              - training
    #                              - validation
    #                              - testing
    create_log_file(net_info, a)

    # region Load data and prepare datasets

    if a.on_fly_data_generation:
        # TODO DatasetRandom should have normalization info too...
        # TODO It should be possible only to provide here the dt as in norm info
        train_set = DatasetRandom(a, inputs_list=net_info.inputs, outputs_list=net_info.outputs, number_of_batches=1000)
        validation_set = DatasetRandom(a, inputs_list=net_info.inputs, outputs_list=net_info.outputs,
                                       number_of_batches=10)

    else:

        paths_to_datafiles_training = get_paths_to_datafiles(a.training_files)
        paths_to_datafiles_validation = get_paths_to_datafiles(a.validation_files)

        for path in paths_to_datafiles_training + paths_to_datafiles_validation:
            dt_sampling = get_sampling_interval_from_datafile(path)
            if abs(net_info.sampling_interval - dt_sampling) > 1.0e-5:
                warning('A difference between network sampling interval and save interval of data file {} detected'
                        .format(path))

        training_dfs = load_data(paths_to_datafiles_training)
        validation_dfs = load_data(paths_to_datafiles_validation)

        training_dfs_norm = normalize_df(training_dfs, normalization_info)
        training_dataset = Dataset(training_dfs_norm, a, shuffle=True, inputs=net_info.inputs, outputs=net_info.outputs)

        validation_dfs_norm = normalize_df(validation_dfs, normalization_info)
        validation_dataset = Dataset(validation_dfs_norm, a, shuffle=True, inputs=net_info.inputs,
                                     outputs=net_info.outputs)

        # test_dfs is not deleted as we need it further for plotting
        del training_dfs, validation_dfs, paths_to_datafiles_validation, paths_to_datafiles_training

    # region In either case testing is done on a data collected offline
    paths_to_datafiles_test = get_paths_to_datafiles(a.test_files)
    test_dfs = load_data(paths_to_datafiles_test)
    test_dfs_norm = normalize_df(test_dfs, normalization_info)
    test_set = Dataset(test_dfs_norm, a, shuffle=False, inputs=net_info.inputs, outputs=net_info.outputs)
    # Check the sampling interval for test file
    for path in paths_to_datafiles_test:
        dt_sampling = get_sampling_interval_from_datafile(path)
        if abs(net_info.sampling_interval - dt_sampling) > 1.0e-5:
            warning('A difference between network sampling interval and save interval of data file {} detected'
                    .format(path))
    # endregion

    # endregion

    net.compile(
        loss=loss_msr_sequence_customizable(wash_out_len=a.wash_out_len,
                                            post_wash_out_len=a.post_wash_out_len,
                                            discount_factor=1.0),
        optimizer=keras.optimizers.Adam(0.001)
    )
    net.summary()
    # endregion

    # region Define callbacks to be used in training

    callbacks_for_training = []

    class PlotPredictionsCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            ...
            # net_for_inference.set_weights(net.get_weights())
            # plot_string = 'This is the network after {} training epoch(s), warm_up={}'.format(epoch + 1, a.wash_out_len)
            # ground_truth, net_outputs, time_axis = \
            #     get_predictions_TF(net_for_inference, net_for_inference_info,
            #                        test_set, normalization_info,
            #                        experiment_length=a.test_len)
            # brunton_widget(net_for_inference_info.inputs, net_for_inference_info.outputs,
            #                ground_truth, net_outputs, time_axis,
            #                )

    plot_predictions_callback = PlotPredictionsCallback()

    callbacks_for_training.append(plot_predictions_callback)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=net_info.path_to_net + net_info.net_full_name + '.ckpt',
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=False)

    callbacks_for_training.append(model_checkpoint_callback)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=1,
        min_lr=0.0001,
        verbose=2
    )

    callbacks_for_training.append(reduce_lr)

    csv_logger = keras.callbacks.CSVLogger(net_info.path_to_net + 'log_training.csv', append=False, separator=';')
    callbacks_for_training.append(csv_logger)

    # endregion

    history = net.fit(
        training_dataset,
        epochs=a.num_epochs,
        verbose=True,
        shuffle=False,
        validation_data=validation_dataset,
        callbacks=callbacks_for_training,
    )

    # region Save final weights as checkpoint
    net.save_weights(net_info.path_to_net + net_info.net_full_name + '.ckpt')
    # endregion

    # region Plot loss change during training
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    # endregion

    # region If NNI enabled send final report
    if nni_parameters is not None:
        nni.report_final_result(history.history['val_loss'][-1])
    # endregion

    # region Calculate and print the total time it took to train the network

    stop = timeit.default_timer()
    total_time = stop - start

    # Print the total time it took to run the function
    print('Total time of training the network: ' + str(total_time))

    # endregion


if __name__ == '__main__':
    import os.path
    import time

    # The following lines help to determine if the file on Google Colab was updated
    file = os.path.realpath(__file__)
    print("Training script last modified: %s" % time.ctime(os.path.getmtime(file)))

    # Run the training function and measure time of execution
    train_network()

    # Use the call below instead of train_network() if you want to use NNI
    # nni_parameters = nni.get_next_parameter()
    # train_network(nni_parameters)
