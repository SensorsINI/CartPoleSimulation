# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 06:21:32 2020

@author: Marcin
"""

import timeit

# Various
import torch.optim as optim
import torch.utils.data.dataloader
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from SI_Toolkit.load_and_normalize import calculate_normalization_info
import torch.nn as nn
# from ax.plot.contour import plot_contour
# from ax.plot.trace import optimization_trace_single_method
# from ax.service.managed_loop import optimize
# from ax.utils.notebook.plotting import render, init_notebook_plotting
# from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN
# from memory_profiler import profile

import time

# Custom functions
from Modeling.Pytorch.utilis_rnn import *
# Parameters of RNN
from Modeling.Pytorch.ParseArgs import args

print('')

device = get_device()

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
        os.makedirs('save')
    except FileExistsError:
        pass

    # Save relevant arguments from a and set hardcoded arguments
    lr = args.lr  # learning rate
    batch_size = args.batch_size  # Mini-batch size
    num_epochs = args.num_epochs  # Number of epochs to train the network
    seq_len = args.seq_len

    # Network architecture:
    rnn_name = args.rnn_name
    inputs_list = args.inputs_list
    outputs_list = args.outputs_list

    load_rnn = args.load_rnn  # If specified this is the name of pretrained RNN which should be loaded
    path_save = args.path_save

    # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
    net, rnn_name, inputs_list, outputs_list \
        = create_rnn_instance(rnn_name, inputs_list, outputs_list, load_rnn, path_save, device)

    # Create log for this RNN and determine its full name
    rnn_full_name = create_log_file(rnn_name, inputs_list, outputs_list, path_save)
    net.rnn_full_name = rnn_full_name

    ########################################################
    # Create Dataset
    ########################################################

    train_dfs, _ = load_data(args, args.train_file_name)

    normalization_info =  calculate_normalization_info(train_dfs, args.path_save, rnn_full_name)

    test_dfs, time_axes_dev = load_data(args, args.val_file_name)

    train_dfs_norm = normalize_df(train_dfs, normalization_info)
    test_dfs_norm = normalize_df(test_dfs, normalization_info)

    del train_dfs, test_dfs

    train_set = Dataset(train_dfs_norm, args)
    dev_set = Dataset(test_dfs_norm, args, time_axes=time_axes_dev)
    print('Number of samples in training set: {}'.format(train_set.number_of_samples))
    print('The training sets sizes are: {}'.format(train_set.df_lengths))
    print('Number of samples in validation set: {}'.format(dev_set.number_of_samples))
    print('')


    plot_results(net=net, args=args, dataset=dev_set, seq_len=1024,
                 comment='This is the network at the beginning of the training',
                 inputs_list=inputs_list, outputs_list=outputs_list,
                 save=True,
                 closed_loop_enabled=True)

    # Create PyTorch dataloaders for train and dev set
    train_generator = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                                      num_workers=args.num_workers)
    dev_generator = data.DataLoader(dataset=dev_set, batch_size=512, shuffle=False, num_workers=args.num_workers)

    # Print parameter count
    print_parameter_count(net)  # Seems not to function well

    # Select Optimizer
    optimizer = optim.Adam(net.parameters(), amsgrad=True, lr=lr)

    # TODO: Verify if scheduler is working. Try tweaking parameters of below scheduler and try cyclic lr scheduler

    # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=0.1)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=1, verbose=True)

    # Select Loss Function
    criterion = nn.MSELoss()  # Mean square error loss function
    '''
    Init Tensorboard
    '''
    comment = f' batch_size={batch_size} lr={lr} seq_len={seq_len}'
    tb = SummaryWriter(comment=comment)
    ########################################################
    # Training
    ########################################################
    print("Starting training...")
    print('')
    time.sleep(0.001)

    # Create dictionary to store training history
    dict_history = {}
    dict_history['epoch'] = []
    dict_history['time'] = []
    dict_history['lr'] = []
    dict_history['train_loss'] = []
    dict_history['dev_loss'] = []
    dict_history['dev_gain'] = []
    dict_history['test_loss'] = []
    dev_gain = 1

    # The epoch_saved variable will indicate from which epoch is the last RNN model,
    # which was good enough to be saved
    epoch_saved = -1
    for epoch in range(num_epochs):

        ###########################################################################################################
        # Training - Iterate batches
        ###########################################################################################################
        # Set RNN in training mode
        net = net.train()
        # Define variables accumulating training loss and counting training batchs
        train_loss = 0
        train_batches = 0

        # Iterate training over available batches
        # tqdm() is just a function which displays the progress bar
        # Otherwise the line below is the same as "for batch, labels in train_generator:"
        for batch, labels in tqdm(train_generator):  # Iterate through batches

            # Reset the network (internal states of hidden layers and output history not the weights!)
            net.reset()

            # Further modifying the input and output form to fit RNN requirements
            # If GPU available we send tensors to GPU (cuda)
            if torch.cuda.is_available():
                batch = batch.float().cuda().transpose(0, 1)
                labels = labels.float().cuda()
            else:
                batch = batch.float().transpose(0, 1)
                labels = labels.float()

            # # Reset memory of gradients
            # optimizer.zero_grad()

            # Warm-up (open loop prediction) to settle the internal state of RNN hidden layers
            net(rnn_input=batch[:args.warm_up_len, :, :])

            # Reset memory of gradients
            optimizer.zero_grad()

            # Forward propagation - These are the results from which we calculate the update to RNN weights
            # GRU Input size must be (seq_len, batch, input_size)
            net(rnn_input=batch[args.warm_up_len:, :, :])
            out = net.return_outputs_history()

            # Get loss
            loss = criterion(out[:, args.warm_up_len:, :],
                             labels[:, args.warm_up_len:, :])

            # Backward propagation
            loss.backward()

            # Gradient clipping - prevent gradient from exploding
            torch.nn.utils.clip_grad_norm_(net.parameters(), 100)

            # Update parameters
            optimizer.step()
            # scheduler.step()
            # Update variables for loss calculation
            batch_loss = loss.detach()
            train_loss += batch_loss  # Accumulate loss
            train_batches += 1  # Accumulate count so we can calculate mean later

        ###########################################################################################################
        # Validation - Iterate batches
        ###########################################################################################################

        # Set the network in evaluation mode
        net = net.eval()

        # Define variables accumulating evaluation loss and counting evaluation batches
        dev_loss = 0
        dev_batches = 0

        for (batch, labels) in tqdm(dev_generator):

            # Reset the network (internal states of hidden layers and output history not the weights!)
            net.reset()

            # Further modifying the input and output form to fit RNN requirements
            # If GPU available we send tensors to GPU (cuda)
            if torch.cuda.is_available():
                batch = batch.float().cuda().transpose(0, 1)
                labels = labels.float().cuda()
            else:
                batch = batch.float().transpose(0, 1)
                labels = labels.float()

            # Warm-up (open loop prediction) to settle the internal state of RNN hidden layers
            net(rnn_input=batch)
            out = net.return_outputs_history()


            # Get loss
            # For evaluation we always calculate loss over the whole maximal prediction period
            # This allow us to compare RNN models from different epochs
            loss = criterion(out[:, args.warm_up_len: args.seq_len],
                             labels[:, args.warm_up_len: args.seq_len])

            # Update variables for loss calculation
            batch_loss = loss.detach()
            dev_loss += batch_loss  # Accumulate loss
            dev_batches += 1  # Accumulate count so we can calculate mean later

        # Reset the network (internal states of hidden layers and output history not the weights!)
        net.reset()
        # Get current learning rate
        # TODO(Fixed. It does changes now): I think now the learning rate do not change during traing, or it is not a right way to get this info.

        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

        scheduler.step(dev_loss)
        '''
        Add data for tensorboard
        TODO : Add network graph and I/O to tensorboard
        '''
        # tb.add_graph(net)
        tb.add_scalar('Train Loss', train_loss / train_batches, epoch)
        tb.add_scalar('Dev Loss', dev_loss / dev_batches, epoch)

        # Add the first sample of batch to tensorboard. Prediction is represented by Dotted line
        # TODO: Concatenate such graphs. But they are not continous
        # for i in range(labels.shape[2]):
        #     time_label = np.arange(0, labels.shape[1], 1)
        #     time_out = np.arange(0, out.shape[1], 1)
        #     true_data = labels[1, :, i]
        #     predicted_data = out[1, :, i]
        #     fig_tb = plt.figure(5)
        #     plt.plot(time_label, true_data.detach().cpu())
        #     plt.plot(time_out, predicted_data.detach().cpu(), linestyle='dashed')
        #     tb.add_figure(tag=str(a.outputs_list[i]), figure=fig_tb, global_step=epoch)

        for name, param in net.named_parameters():
            tb.add_histogram(name, param, epoch)
            tb.add_histogram(f'{name}.grad', param.grad, epoch)
        tb.close()

        # Write the summary information about the training for the just completed epoch to a dictionary

        dict_history['epoch'].append(epoch)
        dict_history['lr'].append(lr_curr)
        dict_history['train_loss'].append(
            train_loss.detach().cpu().numpy() / train_batches / (args.seq_len - args.warm_up_len))
        dict_history['dev_loss'].append(
            dev_loss.detach().cpu().numpy() / dev_batches / (args.seq_len - args.warm_up_len))

        # Get relative loss gain for network evaluation
        if epoch >= 1:
            dev_gain = (dict_history['dev_loss'][epoch - 1] - dict_history['dev_loss'][epoch]) / \
                       dict_history['dev_loss'][epoch - 1]
        dict_history['dev_gain'].append(dev_gain)

        # Print the summary information about the training for the just completed epoch
        print('\nEpoch: %3d of %3d | '
              'LR: %1.5f | '
              'Train-L: %6.4f | '
              'Val-L: %6.4f | '
              'Val-Gain: %3.2f |' % (dict_history['epoch'][epoch], num_epochs - 1,
                                     dict_history['lr'][epoch],
                                     dict_history['train_loss'][epoch],
                                     dict_history['dev_loss'][epoch],
                                     dict_history['dev_gain'][epoch] * 100))
        print('')

        # Save the best model with the lowest dev loss
        # Always save the model from epoch 0
        # TODO: this is a bug: you should only save the model from epoch 0 if there is no pretraind network
        if epoch == 0:
            min_dev_loss = dev_loss
        # If current loss smaller equal than minimal till now achieved loss,
        # save the current RNN model and save its loss as minimal ever achieved
        if dev_loss <= min_dev_loss:
            epoch_saved = epoch
            min_dev_loss = dev_loss
            torch.save(net.state_dict(), args.path_save + rnn_full_name + '.pt', _use_new_zipfile_serialization=False)
            print('>>> saving best model from epoch {}'.format(epoch))
            print('')

            plot_string = 'This is the network after {} training epoch'.format(epoch + 1)
            plot_results(net=net, args=args, dataset=dev_set, seq_len=1024,
                         comment=plot_string,
                         inputs_list=inputs_list, outputs_list=outputs_list, save=True,
                         closed_loop_enabled=True)
        else:
            print('>>> We keep model from epoch {}'.format(epoch_saved))
            print('')

        # Evaluate the performance of the current network
        # by checking its predictions on a randomly generated CartPole experiment
        # get_predictions(net, a, val_file)

    # When finished the training print the final message
    print("Training Completed...                                               ")
    print(" ")

    # Calculate the total time it took to run the function
    stop = timeit.default_timer()
    total_time = stop - start

    # Return the total time it took to run the function
    return total_time


if __name__ == '__main__':
    parameters = dict(
        lr=[.1, .01]
        , batch_size=[100, 200, 300, 400]
        , seq_len=[512 + 512 + 1]
    )
    param_values = [v for v in parameters.values()]

    # for lr, batch_size, seq_len in product(*param_values):
    #     a.lr = lr
    #     a.batch_size = batch_size
    #     a.seq_len = seq_len
    #     time_to_accomplish = train_network()
    #     print('Total time of training the network: ' + str(time_to_accomplish))

    # def train_evaluate(parameterization):
    #     time_to_accomplish, min_dev_loss = train_network()
    #     return min_dev_loss
    # best_parameters, values, experiment, model = optimize(
    #     parameters=[
    #         {"name": "a.lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
    #         {"name": "a.batch_size", "type": "range", "bounds": [100, 400]},
    #         {"name": "a.seq_len", "type": "range", "bounds": [500, 700]},
    #     ],
    #     evaluation_function=train_evaluate,
    #     objective_name='Validation Loss',
    #     minimize=True
    # )

    time_to_accomplish = train_network()
    print('Total time of training the network: ' + str(time_to_accomplish))
