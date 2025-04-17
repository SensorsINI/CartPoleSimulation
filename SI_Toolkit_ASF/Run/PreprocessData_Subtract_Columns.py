from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = './Experiment_Recordings/'
save_files_to = './Experiment_Recordings/'
variables_to_subtract = [  # ['a', 'b', 'c'] -> a-b = c
    ['positionDD', 'positionD_D', 'err_positionDD']
]
indices_by_which_to_shift = [-1]

transform_dataset(get_files_from, save_files_to, transformation='subtract_columns',
                    variables_to_shift=variables_to_shift, indices_by_which_to_shift=indices_by_which_to_shift)
