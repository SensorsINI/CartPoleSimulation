import os
import sys
import numpy as np

from SI_Toolkit.Functions.TF.Network import plot_weights_distribution, get_activation_statistics

from tensorflow import keras
try:
    from SI_Toolkit.Functions.TF.Network import get_pruning_params, make_prunable
    from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
    from tensorflow_model_optimization.sparsity.keras import strip_pruning
except ModuleNotFoundError:
    print('tensorflow_model_optimization not found. Pruning will not be available.')

from SI_Toolkit.Functions.TF.Dataset import Dataset

from SI_Toolkit.Functions.TF.Loss import LossMSRSequence, LossMSRSequenceCustomizableRelative


# Uncomment the @profile(precision=4) to get the report on memory usage after the training
# Warning! It may affect performance. I would discourage you to use it for long training tasks
# @profile(precision=4)
def train_network_core(net, net_info, training_dfs, validation_dfs, test_dfs, a):

    training_dataset = Dataset(training_dfs, a, shuffle=True, inputs=net_info.inputs, outputs=net_info.outputs)

    validation_dataset = Dataset(validation_dfs, a, shuffle=False, inputs=net_info.inputs,
                                 outputs=net_info.outputs)

    del training_dfs, validation_dfs, test_dfs

    print('')
    print('Number of samples in training set: {}'.format(training_dataset.number_of_samples))
    print('The mean number of samples from each experiment used for training is {} with variance {}'.format(np.mean(training_dataset.df_lengths), np.std(training_dataset.df_lengths)))
    print('Number of samples in validation set: {}'.format(validation_dataset.number_of_samples))
    print('')

    # endregion

    # region Set basic training features: optimizer, loss, scheduler...

    if a.pruning_activated:
        net = make_prunable(net, a, training_dataset.number_of_batches)

    # Might be not the same as Pytorch - MSE, not checked
    # net.compile(
    #     loss="mse",
    #     optimizer=keras.optimizers.Adam(a.lr_initial)
    # )

    optimizer = keras.optimizers.Adam(a.lr_initial)

    # loss = LossMSRSequenceCustomizableRelative(
    loss = LossMSRSequence(
        wash_out_len=a.wash_out_len,
        post_wash_out_len=a.post_wash_out_len,
        discount_factor=1.0)

    net.compile(
        loss=loss,
        optimizer=optimizer,
    )
    net.optimizer = optimizer  # When loading a pretrained network, setting optimizer in compile does nothing.
    # region Define callbacks to be used in training

    callbacks_for_training = []

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(net_info.path_to_net, 'ckpt' + '.ckpt'),
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=False)

    callbacks_for_training.append(model_checkpoint_callback)

    if a.reduce_lr_on_plateau:
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=a.lr_decrease_factor,
            patience=a.lr_patience,
            min_lr=a.lr_minimal,
            min_delta=a.min_delta,
            verbose=2
        )

        callbacks_for_training.append(reduce_lr)

    if a.pruning_activated:
        callbacks_for_training.append(pruning_callbacks.UpdatePruningStep())

    post_epoch_training_loss = []
    class AdditionalValidation(keras.callbacks.Callback):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset
        def on_epoch_end(self, epoch, logs=None):
            post_epoch_training_loss.append(self.model.evaluate(self.dataset))

    callbacks_for_training.append(AdditionalValidation(dataset=training_dataset))


    class saving_Callback(keras.callbacks.Callback):
        def __init__(self, path_to_save):
            super().__init__()
            self.path_to_save = path_to_save

        def on_epoch_end(self, epoch, logs=None):
            self.model.save(self.path_to_save)

    callbacks_for_training.append(saving_Callback(os.path.join(net_info.path_to_net, net_info.net_full_name + '.keras')))

    csv_logger = keras.callbacks.CSVLogger(os.path.join(net_info.path_to_net, 'log_training.csv'), append=False, separator=';')
    callbacks_for_training.append(csv_logger)

    # endregion

    # endregion


    loss_eval = net.evaluate(validation_dataset)
    print('Validation loss before starting training is {}'.format(loss_eval))

    # region Print information about the network
    net.summary()
    # endregion

    # region Training loop
    history = net.fit(
        training_dataset,
        epochs=a.num_epochs,
        verbose=True,
        shuffle=False,
        validation_data=validation_dataset,
        callbacks=callbacks_for_training,
    )

    try:
        loss = history.history['loss']
    except KeyError:
        print('Could not find loss in history. Maybe you set the number of epochs to zero? Continuing with the loss set to [].')
        loss = []
    try:
        validation_loss = history.history['val_loss']
    except KeyError:
        print('Could not find validation loss in history. Maybe you set the number of epochs to zero? Continuing with the validation loss set to [].')
        validation_loss = []

    # endregion
    if a.pruning_activated:
        net = strip_pruning(net)
    # region Save final weights as checkpoint
    if net_info.quantization['ACTIVATED']:
        from qkeras.utils import model_save_quantized_weights
        model_save_quantized_weights(net, os.path.join(net_info.path_to_net, net_info.net_full_name + '.h5'))
        try:
            from SI_Toolkit.HLS4ML.convert_with_hls4ml import convert_model_with_hls4ml
            convert_model_with_hls4ml(net)
        except ImportError:
            print('HLS4ML not found. No corresponding info possible.')

    net.save(os.path.join(net_info.path_to_net, net_info.net_full_name + '.keras'))
    net.save_weights(os.path.join(net_info.path_to_net, 'ckpt' + '.ckpt'))
    # endregion

    if a.plot_weights_distribution:
        path_to_parameters_distribution_histograms = os.path.join(net_info.path_to_net, 'parameters_histograms')
        os.makedirs(path_to_parameters_distribution_histograms)

        plot_weights_distribution(net, show=False, path_to_save=path_to_parameters_distribution_histograms)
        # if a.activation_statistics:
        if a.pruning_activated:
            net = keras.models.clone_model(net)
        activation_statistics_datasets = [validation_dataset]
        # activation_statistics_datasets = [training_dataset, validation_dataset]
        get_activation_statistics(net, activation_statistics_datasets, path_to_save=path_to_parameters_distribution_histograms)

    return np.array(loss), validation_loss, post_epoch_training_loss
