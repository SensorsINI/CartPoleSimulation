from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = 'SI_Toolkit_ASF/Experiments/Pretrained-RNN-1-Derivative/Recordings/Train'
save_files_to = get_files_from
variables_for_derivative = ['angle_sin', 'angle_cos', 'angleD', 'position', 'positionD']
derivative_algorithm = "single_difference"

transform_dataset(get_files_from, save_files_to, transformation='append_derivatives',
                  variables_for_derivative=variables_for_derivative, derivative_algorithm=derivative_algorithm)
