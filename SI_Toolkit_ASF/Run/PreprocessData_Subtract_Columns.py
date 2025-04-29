from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = './SI_Toolkit_ASF/Experiments/CPP_MPCswingups_before_23_04_2025/Recordings/Train/'
save_files_to = './SI_Toolkit_ASF/Experiments/CPP_MPCswingups_before_23_04_2025/Recordings/Train/'
variables_to_subtract = [  # ['a', 'b', 'c'] -> a-b = c
    ['D_angleD', 'angleDD', 'err_angleDD'],
    ['D_positionD', 'positionDD', 'err_positionDD'],
]

transform_dataset(get_files_from, save_files_to,
                  transformation='subtract_columns',
                  variables_to_subtract=variables_to_subtract,
                  )
