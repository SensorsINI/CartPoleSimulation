from SI_Toolkit.data_preprocessing import transform_dataset

# # A = 'Test/Test-'
# # A = 'Validate/Validate-'
# A = 'Train/Train-'
# # B = '1s500ms'
# B = '27s'
# folder = A+B
# get_files_from = 'SI_Toolkit_ASF/Experiments/DG-27s-and-1s500ms-noisy/Recordings/'+folder
# save_files_to = 'SI_Toolkit_ASF/Experiments/DG-27s-and-1s500ms-noisy-u/Recordings/'+folder

get_files_from = 'Experiment_Recordings/Experiment_16_11_2024_pole_L_and_m'
save_files_to = 'Experiment_Recordings/Experiment_16_11_2024_pole_L_and_m'
variables_to_shift = ['Q_applied']
indices_by_which_to_shift = [-1]

transform_dataset(get_files_from, save_files_to, transformation='add_shifted_columns',
                    variables_to_shift=variables_to_shift, indices_by_which_to_shift=indices_by_which_to_shift)
