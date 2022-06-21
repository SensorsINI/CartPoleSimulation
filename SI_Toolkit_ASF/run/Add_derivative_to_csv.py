from SI_Toolkit.load_and_normalize import add_derivatives_to_csv_files

get_files_from = 'SI_Toolkit_ASF/Experiments/Pretrained-RNN-1-Derivative/Recordings/Train'
save_files_to = get_files_from
variables_for_derivative = ['angle_sin', 'angle_cos', 'angleD', 'position', 'positionD']
derivative_algorithm = "single_difference"

add_derivatives_to_csv_files(get_files_from, save_files_to, variables_for_derivative, derivative_algorithm)