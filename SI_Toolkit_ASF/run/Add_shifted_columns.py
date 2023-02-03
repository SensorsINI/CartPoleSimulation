from SI_Toolkit.load_and_normalize import add_shifted_columns

# # A = 'Test/Test-'
# # A = 'Validate/Validate-'
# A = 'Train/Train-'
# # B = '1s500ms'
# B = '27s'
# folder = A+B
# get_files_from = 'SI_Toolkit_ASF/Experiments/DG-27s-and-1s500ms-noisy/Recordings/'+folder
# save_files_to = 'SI_Toolkit_ASF/Experiments/DG-27s-and-1s500ms-noisy-u/Recordings/'+folder

get_files_from = 'Experiment_Recordings/TestingAdaptiveNet'
save_files_to = 'Experiment_Recordings/TestingAdaptiveNet-u'
variables_to_shift = ['u']
indices_by_which_to_shift = [-1]

if __name__ == '__main__':
    add_shifted_columns(get_files_from, save_files_to, variables_to_shift, indices_by_which_to_shift)
