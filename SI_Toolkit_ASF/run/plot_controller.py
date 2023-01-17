from SI_Toolkit.load_and_normalize import load_data, normalize_df, get_paths_to_datafiles
import matplotlib.pyplot as plt
from others.globals_and_utils import load_config

config = load_config('config_data_gen.yml')
dt = config['dt']['saving']

# paths_to_datafiles_test = get_paths_to_datafiles('SI_Toolkit_ASF/Experiments/Experiment_compare_neural_imitator_mppi/Recordings/Validate')
# exp_low = load_data(paths_to_datafiles_test)
# Q_low = exp_low[0].Q.array
# pl_low = exp_low[0].pole_length.array

# paths_to_datafiles_test = get_paths_to_datafiles('SI_Toolkit_ASF/Experiments/Mppi_high/Recordings/Validate')
# exp_high = load_data(paths_to_datafiles_test)
# Q_high = exp_high[0].Q.array
# pl_high = exp_high[0].pole_length.array
#
# paths_to_datafiles_test = get_paths_to_datafiles('SI_Toolkit_ASF/Experiments/Mppi_low_to_high_fixed/Recordings/Validate')
# exp_low_to_high = load_data(paths_to_datafiles_test)
# Q_low_to_high = exp_low_to_high[0].Q.array
# pl_low_to_high = exp_low_to_high[0].pole_length.array
#
# paths_to_datafiles_test = get_paths_to_datafiles('SI_Toolkit_ASF/Experiments/Mppi_high_to_low/Recordings/Validate')
# exp_high_to_low = load_data(paths_to_datafiles_test)
# Q_high_to_low = exp_high_to_low[0].Q.array
# pl_high_to_low = exp_high_to_low[0].pole_length.array

# fig, axs = plt.subplots(2)

# axs[0].plot([j*dt for j in range(len(Q_low))], pl_low, 'r')
# axs[0].plot([j*dt for j in range(len(Q_high))], pl_high, 'y')
# axs[0].plot([j*dt for j in range(len(Q_low_to_high))], pl_low_to_high, 'g')
# axs[0].plot([j*dt for j in range(len(Q_high_to_low))], pl_high_to_low, 'b')

# axs[1].plot([j*dt for j in range(len(Q_low))], Q_low, 'r', label='Pole length = 0.2')
# axs[1].plot([j*dt for j in range(len(Q_high))], Q_high, 'y', label='Pole length = 0.6')
# axs[1].plot([j*dt for j in range(len(Q_low_to_high))], Q_low_to_high, 'g', label='Pole length = 0.2 first half, 0.6 second half')
# axs[1].plot([j*dt for j in range(len(Q_high_to_low))], Q_high_to_low, 'b', label='Pole length = 0.6 first half, 0.2 second half')

# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(handles, labels, loc='upper right')

""" Plot chosen controller """

paths_to_datafiles_test = get_paths_to_datafiles('SI_Toolkit_ASF/Experiments/Experiment-21/Recordings/Validate')
exp = load_data(paths_to_datafiles_test)
Q_neural_im = exp[0].Q.array
Q_mppi = exp[0].mppi_Q_pred_from_neural_imitator.array

fig, axs = plt.subplots(2)

axs[0].plot([j*dt for j in range(len(Q_neural_im))], Q_neural_im, 'r', label='Online Neural imitator Q')
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels, loc='upper right')

axs[1].plot([j*dt for j in range(len(Q_mppi))], Q_mppi, 'b', label='Mppi Q')
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles, labels, loc='upper right')



plt.show()
