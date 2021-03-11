# -*- coding: utf-8 -*-
"""
Testing network predictions for CartPole
"""

# "Command line" parameters
from Modeling.TF.Parameters import args

# Custom functions
from Modeling.TF.TF_Functions.Initialization import set_seed, create_full_name, create_log_file, get_net_and_norm_info
from Modeling.TF.TF_Functions.Loss import loss_msr_sequence_customizable
from Modeling.TF.TF_Functions.Dataset import Dataset, DatasetRandom
from Modeling.load_and_normalize import load_data, normalize_df, \
    get_sampling_interval_from_datafile, get_paths_to_datafiles
from Modeling.TF.TF_Functions.Test_predictions import open_loop_prediction_experiment, run_test_gui


# region Import and print "command line" arguments
print('')
a = args()  # 'a' like arguments
print(a.__dict__)
print('')
# endregion



def test_network():

    """
    This function create RNN instance based on parameters saved on disc and also creates the CartPole instance.
    The actual work of evaluation prediction results is done in open_loop_prediction_experiment function
    """

    # Create a copy of the network suitable for inference (stateful and with sequence length one)
    net_for_inference, net_for_inference_info, normalization_info = \
        get_net_and_norm_info(a, time_series_length=1,
                              batch_size=1, stateful=True)


    # region In either case testing is done on a data collected offline
    paths_to_datafiles_test = get_paths_to_datafiles(a.test_files)
    test_dfs = load_data(paths_to_datafiles_test)
    test_dfs_norm = normalize_df(test_dfs, normalization_info)
    test_set = Dataset(test_dfs_norm, a, shuffle=False,
                       inputs=net_for_inference_info.inputs, outputs=net_for_inference_info.outputs)

    ground_truth, net_outputs, time_axis = \
        open_loop_prediction_experiment(net_for_inference, net_for_inference_info,
                                        test_set, normalization_info,
                                        experiment_length=a.test_len)




    run_test_gui(net_for_inference_info.inputs, net_for_inference_info.outputs,
                   ground_truth, net_outputs, time_axis
                   )

if __name__ == '__main__':
    test_network()
