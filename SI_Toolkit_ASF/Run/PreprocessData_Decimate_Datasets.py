from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = 'Experiment_Recordings/TestingAdaptiveNet'
save_files_to = 'Experiment_Recordings/TestingAdaptiveNet-u'
keep_every_nth_row = 4

transform_dataset(get_files_from, save_files_to, transformation='decimate_datasets',
                    keep_every_nth_row=keep_every_nth_row)
