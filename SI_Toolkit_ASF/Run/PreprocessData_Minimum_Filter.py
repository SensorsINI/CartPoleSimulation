from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = 'SI_Toolkit_ASF/Experiments/CPP_MPCswingups_before_23_04_2025/Recordings'
save_files_to = 'SI_Toolkit_ASF/Experiments/CPP_MPCswingups_before_23_04_2025_min/Recordings'

window = 5
features = ['err_angleDD', 'err_positionDD']
thresholds = [50.0, 25.0]

transform_dataset(get_files_from, save_files_to, transformation='minimum_filter', window=window,
                    features=features, thresholds=thresholds)