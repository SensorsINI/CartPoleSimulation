from SI_Toolkit.load_and_normalize import \
    load_data, get_sampling_interval_from_datafile



def preprocess_for_brunton(a):
    # Get dataset:
    test_dfs = load_data(a.test_file)
    if a.test_len == 'max':
        a.test_len = len(test_dfs[
                             0]) - a.test_max_horizon - a.test_start_idx  # You could have +1; then, for last prediction you don not have ground truth to compare with, but you can still calculate it.
    dataset = test_dfs[0].iloc[a.test_start_idx:a.test_start_idx + a.test_len + a.test_max_horizon, :]
    dataset.reset_index(drop=True, inplace=True)

    # Get sampling interval
    dataset_sampling_dt = get_sampling_interval_from_datafile(a.test_file[0])
    if dataset_sampling_dt is None:
        raise ValueError ('No information about sampling interval found')

    time_axis = dataset['time'].to_numpy()[:a.test_len]
    ground_truth_features = a.features+['Q']
    ground_truth = dataset[ground_truth_features].to_numpy()[:a.test_len, :]

    return dataset, time_axis, dataset_sampling_dt, ground_truth
