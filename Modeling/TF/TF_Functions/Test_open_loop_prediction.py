
import copy
import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors

from Modeling.load_and_normalize import denormalize_df, load_data, load_normalization_info, normalize_df, denormalize_numpy_array
from Modeling.TF.TF_Functions.Network import get_internal_states, load_internal_states

from .Dataset import Dataset

cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

def open_loop_prediction_experiment(net_for_inference,
                                    net_for_inference_info,
                                    dataset=None,
                                    normalization_info=None,
                                    experiment_length=None
                                    ):

    # How far NN should predict
    max_horizon = 10

    # Reset dataset to change experiment length
    dataset.reset_exp_len(experiment_length+max_horizon)

    # Get features, target, and time axis
    # Format the experiment data
    features, targets, time_axis = dataset.get_experiment(0)  # Put number in brackets to get the same idx at every run
    time_axis = time_axis[:-1]

    # Make a prediction
    net_outputs = np.zeros(shape=(max_horizon, targets.shape[0], targets.shape[1]))
    normalized_net_output = np.zeros(shape=(max_horizon, targets.shape[0], targets.shape[1]))

    for timestep in range(features.shape[0]-max_horizon):

        # Make prediction based on true data
        net_input = features[timestep, :]
        net_input = net_input[np.newaxis, np.newaxis, :]
        # t2 = timeit.default_timer()
        normalized_net_output[0, timestep, :] = np.squeeze(net_for_inference.predict_on_batch(net_input))
        # t3 = timeit.default_timer()
        # print('t3 evaluation {} ms'.format((t3 - t2) * 1000.0))

        # Save internal state
        states = get_internal_states(net_for_inference)

        # Progress max_horison-1 steps in closed loop
        # save the data for every step in a third dimension of an array
        for i in range(1, max_horizon):
            # We assume control input is the first variable
            # All other variables are in closed loop
            net_input = features[np.newaxis, np.newaxis, timestep + i, :]
            net_input[..., 1:] = normalized_net_output[i-1, timestep, :]
            normalized_net_output[i, timestep, :] = \
                np.squeeze(net_for_inference.predict_on_batch(net_input))


        #  Reset state to where it was after the first step
        net_for_inference.reset_states()
        load_internal_states(net_for_inference, states)


    features_denormalized = denormalize_numpy_array(features, net_for_inference_info.inputs, normalization_info)
    targets_denormalized = denormalize_numpy_array(targets, net_for_inference_info.outputs, normalization_info)
    net_outputs_denormalized = denormalize_numpy_array(net_outputs, net_for_inference_info.outputs, normalization_info)

    # Data tailored for plotting
    ground_truth = features_denormalized
    net_outputs = net_outputs_denormalized

    # time_axis is a time axis for ground truth
    return ground_truth, net_outputs, time_axis

def brunton_widget(inputs, outputs, ground_truth, net_outputs, time_axis,
                   starting_point_at_timeaxis=None, max_horizon=10, plot_all=False):

    # Start at should be done bu widget (slider)
    if starting_point_at_timeaxis is None:
        starting_point_at_timeaxis = ground_truth.shape[0]//2
    feature = 's.position'
    
    # Todo: Find feature_idx for ground truth and target for
    feature_idx = inputs.index(feature)
    target_idx = outputs.index(feature)

    # Brunton Plot
    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    axs[0].plot(time_axis, ground_truth[:, feature_idx], 'k:', markersize=12, label='Ground Truth')
    if not plot_all:
        axs[0].plot(time_axis[starting_point_at_timeaxis], ground_truth[starting_point_at_timeaxis, feature_idx],
                    'g.', markersize=16, label='Start')
        prediction_distance = []
        for i in range(max_horizon):
            prediction_distance.append(net_outputs[i, starting_point_at_timeaxis, target_idx])
            axs[0].plot(time_axis[starting_point_at_timeaxis+i+1], prediction_distance[i],
                        c=cmap(float(i)/max_horizon),
                        marker='.')
        plt.show()
    





# 
# 
# 
# 
# 
# # There should be one function returning dataframe with performed experiment and another one plotting and saving
# def open_loop_prediction_experiment(net,
#                                     args,
#                                     dataset=None,
#                                     normalization_info=None,
#                                     time_axes=None,
#                                     testset_filepath=None,
#                                     inputs_list=None,
#                                     outputs_list=None,
#                                     closed_loop_list=None,
#                                     exp_len=None,
#                                     warm_up_len=None,
#                                     closed_loop_enabled=False,
#                                     comment='',
#                                     rnn_full_name=None,
#                                     save=False,
#                                     close_loop_idx=None,
#                                     path_save=None,
#                                     start_at = None):
#     """
#     This function accepts RNN instance, arguments and CartPole instance.
#     It runs one random experiment with CartPole,
#     inputs the data into RNN and check how well RNN predicts CartPole state one time step ahead of time
#     """
# 
#     rnn_full_name = net.rnn_full_name
#     rnn_name = net.rnn_name
#     inputs_list = net.inputs_list
#     outputs_list = net.outputs_list
# 
#     if path_save is None and args is not None:
#         path_save = args.PATH_TO_EXPERIMENT_RECORDINGS
# 
#     if testset_filepath is None:
#         testset_filepath = args.val_file_name
#         if type(testset_filepath) == list:
#             testset_filepath = testset_filepath[0]
# 
#     if warm_up_len is None:
#         warm_up_len = args.warm_up_len
# 
#     if exp_len is None:
#         exp_len = args.exp_len
# 
#     columns_list = list(set(inputs_list).union(set(outputs_list)))
# 
#     if closed_loop_enabled and (closed_loop_list is None):
#         closed_loop_list = args.close_loop_for
#         if closed_loop_list is None:
#             raise ValueError('RNN closed-loop-inputs not provided!')
# 
#     if close_loop_idx is None:
#         close_loop_idx = exp_len//2
# 
#     # Create new model which will return after every time step
# 
#     net_predict = myNN(rnn_name,
#                        inputs_list,
#                        outputs_list,
#                        warm_up_len=1,
#                        return_sequence=False,
#                        batchSize=1,
#                        stateful=True
#                        )
#     net_predict.rnn_full_name = rnn_full_name
# 
#     net_predict.set_weights(net.get_weights())
# 
#     # net_predict.summary()
#     # SAVEPATH = PATH_TO_EXPERIMENT_RECORDINGS+rnn_full_name+'/1/'
#     # print(SAVEPATH)
#     #
#     # net_predict = keras.models.load_model(SAVEPATH)
#     # net_predict.set_weights(net.get_weights())
# 
#     if normalization_info is None:
#         normalization_info = load_normalization_info(path_save, rnn_full_name)
# 
#     if dataset is None or dataset.time_axes is None:
#         test_dfs, time_axes = load_data(args, testset_filepath, columns_list=columns_list)
#         test_dfs_norm = normalize_df(test_dfs, normalization_info)
#         test_set = Dataset(test_dfs_norm, args,
#                            time_axes=time_axes, exp_len=exp_len,
#                            inputs_list=inputs_list, outputs_list=outputs_list)
#         del test_dfs
#     else:
#         test_set = copy.deepcopy(dataset)
#         test_set.reset_exp_len(exp_len=exp_len)
# 
#     # Format the experiment data
#     features, targets, time_axis = test_set.get_experiment(0)  # Put number in brackets to get the same idx at every run
#     if start_at is not None:
#         features = features[start_at:]
#         targets = targets[start_at:]
#         time_axis = time_axis[start_at:]
# 
#     features_pd = pd.DataFrame(data=features, columns=inputs_list, dtype=np.float32)
#     targets_pd = pd.DataFrame(data=targets, columns=outputs_list, dtype=np.float32)
#     rnn_outputs = pd.DataFrame(columns=outputs_list, dtype=np.float32)
# 
#     idx_cl = 0
#     close_the_loop = False
#     # print()
#     for index, row in features_pd.iterrows():
# 
#         states = get_internal_states(net_predict)
#         # print(states)
#         net_predict.reset_states()
#         load_internal_states(net_predict, states)
# 
#         if idx_cl == close_loop_idx:
#             close_the_loop = True
#             # print('RNN input:')
#             # print(rnn_input)
#             # print()
#             # print('Rnn internal states')
#             # for state in states:
#             #     print(state)
#             #     print()
#             print('p: {}'.format(normalized_rnn_output))
# 
# 
#         # states = get_internal_states(net_predict)
#         rnn_input = pd.DataFrame(copy.deepcopy(row)).transpose().reset_index(drop=True)
# 
#         if closed_loop_enabled and close_the_loop and (normalized_rnn_output is not None):
#             rnn_input[closed_loop_list] = normalized_rnn_output[closed_loop_list]
#         rnn_input = np.squeeze(rnn_input.to_numpy())
#         rnn_input = rnn_input[np.newaxis, np.newaxis, :]
#         # t2 = timeit.default_timer()
#         normalized_rnn_output = net_predict.predict_on_batch(rnn_input)
#         # t3 = timeit.default_timer()
#         # print('t3 evaluation {} ms'.format((t3 - t2) * 1000.0))
#         normalized_rnn_output = np.squeeze(normalized_rnn_output).tolist()
#         normalized_rnn_output = copy.deepcopy(pd.DataFrame(data=[normalized_rnn_output], columns=outputs_list))
# 
#         rnn_outputs = rnn_outputs.append(copy.deepcopy(normalized_rnn_output), ignore_index=True)
#         idx_cl += 1
# 
#     features_pd_denorm = denormalize_df(features_pd, normalization_info)
#     targets_pd_denorm = denormalize_df(targets_pd, normalization_info)
#     rnn_outputs_denorm = denormalize_df(rnn_outputs, normalization_info)
#     figs = plot_open_loop_prediction_experiment(targets_pd_denorm, rnn_outputs_denorm, features_pd_denorm, time_axis, comment, closed_loop_enabled,
#                                                 close_loop_idx)
# 
#     plt.show()
# 
#     if save:
#         # Make folders if not yet exist
#         try:
#             os.makedirs('save_plots_tf')
#         except FileExistsError:
#             pass
#         dateTimeObj = datetime.now()
#         timestampStr = dateTimeObj.strftime("-%d%b%Y_%H%M%S")
# 
#         for i in range(len(figs)):
#             fig = figs[i]
#             figNrStr = '-'+str(i)+''
#             if rnn_full_name is not None:
#                 fig.savefig('./save_plots_tf/' + rnn_full_name + figNrStr +timestampStr + '.png')
#             else:
#                 fig.savefig('./save_plots_tf/' + figNrStr + timestampStr + '.png')
# 
# 
# def plot_open_loop_prediction_experiment(targets_pd, rnn_outputs, features_pd_denorm, time_axis, comment, closed_loop_enabled, close_loop_idx):
#     start_idx = 10
#     end_idx = -1
#     if time_axis == []:
#         time_axis = np.arange(0, targets_pd.shape[0])
#         time_axis_string = 'Sample number'
#     else:
#         time_axis = time_axis[1:]
#         time_axis = time_axis-min(time_axis) # Start at 0, comment out if you want to relate to a true experiment
#         time_axis_string = 'Time [s]'
# 
#     figs = []
#     angle_target = None
#     angle_output = None
# 
#     if ('s.angle.sin' in rnn_outputs) and ('s.angle.cos' in rnn_outputs):
#         sin_target = targets_pd['s.angle.sin'].to_numpy()
#         sin_output = rnn_outputs['s.angle.sin'].to_numpy()
#         cos_target = targets_pd['s.angle.cos'].to_numpy()
#         cos_output = rnn_outputs['s.angle.cos'].to_numpy()
# 
#         # Create a figure instance
#         fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
#         plt.subplots_adjust(hspace=0.4)
#         axs[0].set_title(comment, fontsize=20)
# 
#         axs[0].set_ylabel("Cos (-)", fontsize=18)
#         axs[0].plot(time_axis[start_idx:end_idx], cos_target[start_idx:end_idx], 'k:', markersize=12, label='Ground Truth')
#         axs[0].plot(time_axis[start_idx:end_idx], cos_output[start_idx:end_idx], 'b', markersize=12, label='Predicted Cos')
# 
#         axs[0].plot(time_axis[start_idx], cos_target[start_idx], 'g.', markersize=16, label='Start')
#         axs[0].plot(time_axis[start_idx], cos_output[start_idx], 'g.', markersize=16)
#         axs[0].plot(time_axis[-1], cos_target[-1], 'r.', markersize=16, label='End')
#         axs[0].plot(time_axis[-1], cos_output[-1], 'r.', markersize=16)
#         if closed_loop_enabled:
#             axs[0].plot(time_axis[close_loop_idx], cos_target[close_loop_idx], '.', color='darkorange', markersize=16, label='connect output->input')
#             axs[0].plot(time_axis[close_loop_idx], cos_output[close_loop_idx], '.', color='darkorange', markersize=16)
# 
#         axs[0].legend()
# 
#         axs[1].set_ylabel("Sin (-)", fontsize=18)
#         axs[1].plot(time_axis[start_idx:end_idx], sin_target[start_idx:end_idx], 'k:', markersize=12, label='Ground Truth')
#         axs[1].plot(time_axis[start_idx:end_idx], sin_output[start_idx:end_idx], 'b', markersize=12,
#                     label='Predicted angle')
# 
#         axs[1].plot(time_axis[start_idx], sin_target[start_idx], 'g.', markersize=16,
#                     label='Start')
#         axs[1].plot(time_axis[start_idx], sin_output[start_idx], 'g.', markersize=16)
#         axs[1].plot(time_axis[-1], sin_target[-1], 'r.', markersize=16, label='End')
#         axs[1].plot(time_axis[-1], sin_output[-1], 'r.', markersize=16)
#         if closed_loop_enabled:
#             axs[1].plot(time_axis[close_loop_idx], sin_target[close_loop_idx], '.',
#                         color='darkorange', markersize=16, label='connect output->input')
#             axs[1].plot(time_axis[close_loop_idx], sin_output[close_loop_idx], '.',
#                         color='darkorange', markersize=16)
#         axs[1].legend()
# 
# 
#     if ('s.position' in targets_pd) and ('s.position' in rnn_outputs):
#         if ('s.angle' in targets_pd) and ('s.angle' in rnn_outputs):
#             angle_target = np.rad2deg(targets_pd['s.angle'].to_numpy())
#             angle_output = np.rad2deg(rnn_outputs['s.angle'].to_numpy())
#         elif ('s.angle.sin' in rnn_outputs) and ('s.angle.cos' in rnn_outputs):
#             angle_target = np.rad2deg(np.arctan2(targets_pd['s.angle.sin'].to_numpy(), targets_pd['s.angle.cos'].to_numpy()))
#             angle_output = np.rad2deg(np.arctan2(rnn_outputs['s.angle.sin'].to_numpy(), rnn_outputs['s.angle.cos'].to_numpy()))
# 
# 
#         position_target = targets_pd['s.position'].to_numpy()
#         position_output = rnn_outputs['s.position'].to_numpy()
# 
#         number_of_plots = 1
#         if angle_output is not None:
#             number_of_plots += 1
#             if 'Q' in features_pd_denorm.columns:
#                 number_of_plots += 1
# 
# 
# 
# 
#         # Create a figure instance
#         fig, axs = plt.subplots(number_of_plots, 1, figsize=(18, 10), sharex=True)
#         plt.subplots_adjust(hspace=0.4)
#         axs[0].set_title(comment, fontsize=20)
# 
#         axs[0].set_ylabel("Position (m)", fontsize=18)
#         axs[0].plot(time_axis[start_idx:end_idx], position_target[start_idx:end_idx], 'k:', markersize=12, label='Ground Truth')
#         axs[0].plot(time_axis[start_idx:end_idx], position_output[start_idx:end_idx], 'b', markersize=12, label='Predicted position')
# 
#         axs[0].plot(time_axis[start_idx], position_target[start_idx], 'g.', markersize=16, label='Start')
#         axs[0].plot(time_axis[start_idx], position_output[start_idx], 'g.', markersize=16)
#         axs[0].plot(time_axis[-1], position_target[-1], 'r.', markersize=16, label='End')
#         axs[0].plot(time_axis[-1], position_output[-1], 'r.', markersize=16)
#         if closed_loop_enabled:
#             axs[0].plot(time_axis[close_loop_idx], position_target[close_loop_idx], '.', color='darkorange', markersize=16, label='connect output->input')
#             axs[0].plot(time_axis[close_loop_idx], position_output[close_loop_idx], '.', color='darkorange', markersize=16)
# 
#         axs[0].legend()
# 
#         if number_of_plots>1:
#             axs[1].set_ylabel("Angle (deg)", fontsize=18)
#             axs[1].plot(time_axis[start_idx:end_idx], angle_target[start_idx:end_idx], 'k:', markersize=12, label='Ground Truth')
#             axs[1].plot(time_axis[start_idx:end_idx], angle_output[start_idx:end_idx], 'b', markersize=12,
#                         label='Predicted angle')
# 
#             axs[1].plot(time_axis[start_idx], angle_target[start_idx], 'g.', markersize=16,
#                         label='Start')
#             axs[1].plot(time_axis[start_idx], angle_output[start_idx], 'g.', markersize=16)
#             axs[1].plot(time_axis[-1], angle_target[-1], 'r.', markersize=16, label='End')
#             axs[1].plot(time_axis[-1], angle_output[-1], 'r.', markersize=16)
#             if closed_loop_enabled:
#                 axs[1].plot(time_axis[close_loop_idx], angle_target[close_loop_idx], '.',
#                             color='darkorange', markersize=16, label='connect output->input')
#                 axs[1].plot(time_axis[close_loop_idx], angle_output[close_loop_idx], '.',
#                             color='darkorange', markersize=16)
#             axs[1].legend()
# 
#         if number_of_plots>2:
#             Q = features_pd_denorm['Q'].to_numpy()
#             axs[2].set_ylabel("Motor (-)", fontsize=18)
#             axs[2].plot(time_axis[start_idx:end_idx], Q[start_idx:end_idx], 'r', markersize=12, label='Ground Truth')
# 
#             axs[2].plot(time_axis[start_idx], Q[start_idx], 'g.', markersize=16,
#                         label='Start')
#             axs[2].plot(time_axis[-1], Q[-1], 'r.', markersize=16, label='End')
#             if closed_loop_enabled:
#                 axs[2].plot(time_axis[close_loop_idx], Q[close_loop_idx], '.',
#                             color='darkorange', markersize=16, label='connect output->input')
# 
#             axs[2].tick_params(axis='both', which='major', labelsize=16)
# 
#             axs[2].set_xlabel(time_axis_string, fontsize=18)
#             axs[2].legend()
# 
#         figs.append(fig)
# 
#     if ('s.positionD' in targets_pd) and ('s.positionD' in rnn_outputs) and \
#             ('s.angleD' in targets_pd) and ('s.angleD' in rnn_outputs):
# 
#         number_of_plots = 2
#         if 'Q' in features_pd_denorm.columns:
#             number_of_plots += 1
# 
#         positionD_target = targets_pd['s.positionD'].to_numpy()
#         positionD_output = rnn_outputs['s.positionD'].to_numpy()
# 
#         angleD_target = targets_pd['s.angleD'].to_numpy()
#         angleD_output = rnn_outputs['s.angleD'].to_numpy()
# 
#         # Create a figure instance
#         fig, axs = plt.subplots(number_of_plots, 1, figsize=(18, 10), sharex=True)
#         plt.subplots_adjust(hspace=0.4)
#         axs[0].set_title(comment, fontsize=20)
# 
#         axs[0].set_ylabel("PositionD (m/s)", fontsize=18)
#         axs[0].plot(time_axis[start_idx:end_idx], positionD_target[start_idx:end_idx], 'k:', markersize=12, label='Ground Truth')
#         axs[0].plot(time_axis[start_idx:end_idx], positionD_output[start_idx:end_idx], 'b', markersize=12, label='Predicted position')
# 
#         axs[0].plot(time_axis[start_idx], positionD_target[start_idx], 'g.', markersize=16, label='Start')
#         axs[0].plot(time_axis[start_idx], positionD_output[start_idx], 'g.', markersize=16)
#         axs[0].plot(time_axis[-1], positionD_target[-1], 'r.', markersize=16, label='End')
#         axs[0].plot(time_axis[-1], positionD_output[-1], 'r.', markersize=16)
#         if closed_loop_enabled:
#             axs[0].plot(time_axis[close_loop_idx], positionD_target[close_loop_idx], '.', color='darkorange', markersize=16, label='connect output->input')
#             axs[0].plot(time_axis[close_loop_idx], positionD_output[close_loop_idx], '.', color='darkorange', markersize=16)
# 
#         axs[0].legend()
# 
# 
#         axs[1].set_ylabel("AngleD (deg/s)", fontsize=18)
#         axs[1].plot(time_axis[start_idx:end_idx], angleD_target[start_idx:end_idx], 'k:', markersize=12, label='Ground Truth')
#         axs[1].plot(time_axis[start_idx:end_idx], angleD_output[start_idx:end_idx], 'b', markersize=12,
#                     label='Predicted angle')
# 
#         axs[1].plot(time_axis[start_idx], angleD_target[start_idx], 'g.', markersize=16,
#                     label='Start')
#         axs[1].plot(time_axis[start_idx], angleD_output[start_idx], 'g.', markersize=16)
#         axs[1].plot(time_axis[-1], angleD_target[-1], 'r.', markersize=16, label='End')
#         axs[1].plot(time_axis[-1], angleD_output[-1], 'r.', markersize=16)
#         if closed_loop_enabled:
#             axs[1].plot(time_axis[close_loop_idx], angleD_target[close_loop_idx], '.',
#                         color='darkorange', markersize=16, label='connect output->input')
#             axs[1].plot(time_axis[close_loop_idx], angleD_output[close_loop_idx], '.',
#                         color='darkorange', markersize=16)
#         axs[1].legend()
# 
#         if number_of_plots>2:
#             Q = features_pd_denorm['Q'].to_numpy()
#             axs[2].set_ylabel("Motor (-)", fontsize=18)
#             axs[2].plot(time_axis[start_idx:end_idx], Q[start_idx:end_idx], 'r', markersize=12, label='Ground Truth')
# 
#             axs[2].plot(time_axis[start_idx], Q[start_idx], 'g.', markersize=16,
#                         label='Start')
#             axs[2].plot(time_axis[-1], Q[-1], 'r.', markersize=16, label='End')
#             if closed_loop_enabled:
#                 axs[2].plot(time_axis[close_loop_idx], Q[close_loop_idx], '.',
#                             color='darkorange', markersize=16, label='connect output->input')
# 
#             axs[2].tick_params(axis='both', which='major', labelsize=16)
# 
#             axs[2].set_xlabel(time_axis_string, fontsize=18)
#             axs[2].legend()
# 
#         figs.append(fig)
# 
# 
#     if 'Q' in targets_pd:
#         motor_power_target = targets_pd['Q'].to_numpy()
#         motor_power_output = targets_pd['Q'].to_numpy()
# 
#         # Create a figure instance
#         fig, axs = plt.subplots(1, 1, figsize=(18, 10))
#         plt.subplots_adjust(hspace=0.4)
#         start_idx = 0
#         axs.set_title(comment, fontsize=20)
# 
#         axs.set_ylabel("Motor power (-)", fontsize=18)
#         axs.plot(time_axis, motor_power_target, 'k:', markersize=12, label='Ground Truth')
#         axs.plot(time_axis, motor_power_output, 'b', markersize=12, label='Predicted position')
# 
#         axs.plot(time_axis[start_idx], motor_power_target[start_idx], 'g.', markersize=16, label='Start')
#         axs.plot(time_axis[start_idx], motor_power_output[start_idx], 'g.', markersize=16)
#         axs.plot(time_axis[-1], motor_power_target[-1], 'r.', markersize=16, label='End')
#         axs.plot(time_axis[-1], motor_power_output[-1], 'r.', markersize=16)
# 
#         axs.legend()
# 
#         figs.append(fig)
# 
# 
#     return figs