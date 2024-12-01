
from SI_Toolkit.Predictors.neural_network_evaluator import neural_network_evaluator
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

get_files_from = ['./PCA/Experiment/']
paths_to_recordings = get_paths_to_datafiles(get_files_from)
dfs = load_data(list_of_paths_to_datafiles=paths_to_recordings, verbose=False)

net_name = 'Dense-128IN-128H1-128H2-2OUT-0'
path_to_models = './Experiments/AGRU_L_m_pole_up/Models'


net_evaluator = neural_network_evaluator(
    net_name=net_name,
    path_to_models=path_to_models,
    batch_size=1,
    input_precision='float',
    hls4ml=False)

network_inputs = ['GRU_H1_00', 'GRU_H1_01', 'GRU_H1_02', 'GRU_H1_03', 'GRU_H1_04', 'GRU_H1_05', 'GRU_H1_06', 'GRU_H1_07', 'GRU_H1_08', 'GRU_H1_09', 'GRU_H1_10', 'GRU_H1_11', 'GRU_H1_12', 'GRU_H1_13', 'GRU_H1_14', 'GRU_H1_15', 'GRU_H1_16', 'GRU_H1_17', 'GRU_H1_18', 'GRU_H1_19', 'GRU_H1_20', 'GRU_H1_21', 'GRU_H1_22', 'GRU_H1_23', 'GRU_H1_24', 'GRU_H1_25', 'GRU_H1_26', 'GRU_H1_27', 'GRU_H1_28', 'GRU_H1_29', 'GRU_H1_30', 'GRU_H1_31', 'GRU_H1_32', 'GRU_H1_33', 'GRU_H1_34', 'GRU_H1_35', 'GRU_H1_36', 'GRU_H1_37', 'GRU_H1_38', 'GRU_H1_39', 'GRU_H1_40', 'GRU_H1_41', 'GRU_H1_42', 'GRU_H1_43', 'GRU_H1_44', 'GRU_H1_45', 'GRU_H1_46', 'GRU_H1_47', 'GRU_H1_48', 'GRU_H1_49', 'GRU_H1_50', 'GRU_H1_51', 'GRU_H1_52', 'GRU_H1_53', 'GRU_H1_54', 'GRU_H1_55', 'GRU_H1_56', 'GRU_H1_57', 'GRU_H1_58', 'GRU_H1_59', 'GRU_H1_60', 'GRU_H1_61', 'GRU_H1_62', 'GRU_H1_63', 'GRU_H2_00', 'GRU_H2_01', 'GRU_H2_02', 'GRU_H2_03', 'GRU_H2_04', 'GRU_H2_05', 'GRU_H2_06', 'GRU_H2_07', 'GRU_H2_08', 'GRU_H2_09', 'GRU_H2_10', 'GRU_H2_11', 'GRU_H2_12', 'GRU_H2_13', 'GRU_H2_14', 'GRU_H2_15', 'GRU_H2_16', 'GRU_H2_17', 'GRU_H2_18', 'GRU_H2_19', 'GRU_H2_20', 'GRU_H2_21', 'GRU_H2_22', 'GRU_H2_23', 'GRU_H2_24', 'GRU_H2_25', 'GRU_H2_26', 'GRU_H2_27', 'GRU_H2_28', 'GRU_H2_29', 'GRU_H2_30', 'GRU_H2_31', 'GRU_H2_32', 'GRU_H2_33', 'GRU_H2_34', 'GRU_H2_35', 'GRU_H2_36', 'GRU_H2_37', 'GRU_H2_38', 'GRU_H2_39', 'GRU_H2_40', 'GRU_H2_41', 'GRU_H2_42', 'GRU_H2_43', 'GRU_H2_44', 'GRU_H2_45', 'GRU_H2_46', 'GRU_H2_47', 'GRU_H2_48', 'GRU_H2_49', 'GRU_H2_50', 'GRU_H2_51', 'GRU_H2_52', 'GRU_H2_53', 'GRU_H2_54', 'GRU_H2_55', 'GRU_H2_56', 'GRU_H2_57', 'GRU_H2_58', 'GRU_H2_59', 'GRU_H2_60', 'GRU_H2_61', 'GRU_H2_62', 'GRU_H2_63']
network_outputs = ['L', 'm_pole']

# Extract test features and targets
time = df.loc[:, 'time'].to_numpy()
x_tst = df.loc[:, network_inputs].to_numpy()
y_tst = df.loc[:, network_outputs].to_numpy()

y_predicted = []
for i in trange(0, len(y_tst)):
    y_predicted.append(np.squeeze(net_evaluator.step(x_tst[i])))

y_predicted = np.array(y_predicted)


plt.figure()
plt.title('Pole Length vs. Time')
plt.scatter(time, 100.0*y_predicted[:, 0], label='Predicted Values', color='green')
plt.plot(time, 100.0*y_tst[:, 0], label='True Values', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('Pole Length [cm]')
plt.legend()
plt.show()


plt.figure()
plt.title('m_pole')
plt.plot(time, y_predicted[:, 1], label='Predicted Values')
plt.plot(time, y_tst[:, 1], label='True Values')
plt.legend()
plt.show()

