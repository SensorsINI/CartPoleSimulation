import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import trange
from matplotlib.gridspec import GridSpec

from SI_Toolkit.load_and_normalize import load_data
from SI_Toolkit.Predictors.neural_network_evaluator import neural_network_evaluator

testfile = './MLP_readout_v1.csv'
net_name = 'Dense-128IN-128H1-128H2-2OUT-0'
path_to_models = './'
network_inputs = ['GRU_H1_00', 'GRU_H1_01', 'GRU_H1_02', 'GRU_H1_03', 'GRU_H1_04', 'GRU_H1_05', 'GRU_H1_06', 'GRU_H1_07', 'GRU_H1_08', 'GRU_H1_09', 'GRU_H1_10', 'GRU_H1_11', 'GRU_H1_12', 'GRU_H1_13', 'GRU_H1_14', 'GRU_H1_15', 'GRU_H1_16', 'GRU_H1_17', 'GRU_H1_18', 'GRU_H1_19', 'GRU_H1_20', 'GRU_H1_21', 'GRU_H1_22', 'GRU_H1_23', 'GRU_H1_24', 'GRU_H1_25', 'GRU_H1_26', 'GRU_H1_27', 'GRU_H1_28', 'GRU_H1_29', 'GRU_H1_30', 'GRU_H1_31', 'GRU_H1_32', 'GRU_H1_33', 'GRU_H1_34', 'GRU_H1_35', 'GRU_H1_36', 'GRU_H1_37', 'GRU_H1_38', 'GRU_H1_39', 'GRU_H1_40', 'GRU_H1_41', 'GRU_H1_42', 'GRU_H1_43', 'GRU_H1_44', 'GRU_H1_45', 'GRU_H1_46', 'GRU_H1_47', 'GRU_H1_48', 'GRU_H1_49', 'GRU_H1_50', 'GRU_H1_51', 'GRU_H1_52', 'GRU_H1_53', 'GRU_H1_54', 'GRU_H1_55', 'GRU_H1_56', 'GRU_H1_57', 'GRU_H1_58', 'GRU_H1_59', 'GRU_H1_60', 'GRU_H1_61', 'GRU_H1_62', 'GRU_H1_63', 'GRU_H2_00', 'GRU_H2_01', 'GRU_H2_02', 'GRU_H2_03', 'GRU_H2_04', 'GRU_H2_05', 'GRU_H2_06', 'GRU_H2_07', 'GRU_H2_08', 'GRU_H2_09', 'GRU_H2_10', 'GRU_H2_11', 'GRU_H2_12', 'GRU_H2_13', 'GRU_H2_14', 'GRU_H2_15', 'GRU_H2_16', 'GRU_H2_17', 'GRU_H2_18', 'GRU_H2_19', 'GRU_H2_20', 'GRU_H2_21', 'GRU_H2_22', 'GRU_H2_23', 'GRU_H2_24', 'GRU_H2_25', 'GRU_H2_26', 'GRU_H2_27', 'GRU_H2_28', 'GRU_H2_29', 'GRU_H2_30', 'GRU_H2_31', 'GRU_H2_32', 'GRU_H2_33', 'GRU_H2_34', 'GRU_H2_35', 'GRU_H2_36', 'GRU_H2_37', 'GRU_H2_38', 'GRU_H2_39', 'GRU_H2_40', 'GRU_H2_41', 'GRU_H2_42', 'GRU_H2_43', 'GRU_H2_44', 'GRU_H2_45', 'GRU_H2_46', 'GRU_H2_47', 'GRU_H2_48', 'GRU_H2_49', 'GRU_H2_50', 'GRU_H2_51', 'GRU_H2_52', 'GRU_H2_53', 'GRU_H2_54', 'GRU_H2_55', 'GRU_H2_56', 'GRU_H2_57', 'GRU_H2_58', 'GRU_H2_59', 'GRU_H2_60', 'GRU_H2_61', 'GRU_H2_62', 'GRU_H2_63']
network_outputs = ['L', 'm_pole']

df = load_data(list_of_paths_to_datafiles=[testfile], verbose=False)[0]

def network_evaluator():

    net_evaluator = neural_network_evaluator(
        net_name=net_name,
        path_to_models=path_to_models,
        batch_size=1,
        input_precision='float',
        hls4ml=False)

    # Extract test features and targets
    x_tst = df.loc[:, network_inputs].to_numpy()
    y_tst = df.loc[:, network_outputs].to_numpy()

    y_predicted = []
    for i in trange(0, len(y_tst)):
        y_predicted.append(np.squeeze(net_evaluator.step(x_tst[i])))

    y_predicted = np.array(y_predicted)

    df_out = pd.DataFrame(y_predicted, columns=network_outputs)

    return df_out


df_network_output = network_evaluator()

# 1. Set font sizes (same as the first plot)
plt.rcParams.update({'font.size': 14})

# 2. Define time range for plotting
time_start = 5  # Start time in seconds
time_end = 35  # End time in seconds

# Optionally, you can also limit to time_end:
time_range_mask = (df['time'] >= time_start) & (df['time'] <= time_end)

# Filter data based on the time range
time = df['time'][time_range_mask]
predicted_L = 100.0 * df_network_output['L'][time_range_mask]
true_L = 100.0 * df['L'][time_range_mask]
angle_degrees = np.degrees(df['angle'][time_range_mask])
y_target_position = df['target_position'][time_range_mask] * 100  # Convert meters to centimeters
y_position = df['position'][time_range_mask] * 100  # Convert meters to centimeters
y_target_equilibrium = df['target_equilibrium'][time_range_mask]

# 3. Create the plots with adjusted subplot heights
fig = plt.figure(figsize=(16, 8))  # Increased height for better spacing
gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.1)  # Increased hspace for clarity

# 4. First subplot: scatter and line plot for 'L'
ax0 = fig.add_subplot(gs[0])

# Scatter and line plots with adjusted z-order
ax0.plot(time, true_L, label='True Values', color='orange', linewidth=3, zorder=1)
ax0.scatter(time, predicted_L, label='Predicted Values', color='green', s=10.0, zorder=2)

ax0.set_ylabel('Pole Length (cm)')
ax0.legend()

# **Hide x-axis ticks and labels for ax0**
ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# 5. Second subplot: Target Position and Equilibrium
ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax1a = ax1.twinx()

position_color = 'deepskyblue'
ax1.plot(time, y_target_position, color='blue', label='Target Position')
ax1.plot(time, y_position, color=position_color, label='Target Position')
ax1.text(-0.065, 0.5, 'Target Position', color='blue', fontsize=14,
         transform=ax1.transAxes, rotation=90, va='center', ha='right')
# Add blue "Position (cm)" text
ax1.text(-0.06, 0.075, '&', color='black', fontsize=14,
         transform=ax1.transAxes, rotation=90, va='center', ha='left')
ax1.text(-0.06, 0.41, 'Position', color=position_color, fontsize=14,
         transform=ax1.transAxes, rotation=90, va='center', ha='left')
ax1.text(-0.06, 0.85, '(cm)', color='black', fontsize=14,
         transform=ax1.transAxes, rotation=90, va='center', ha='left')

ax1.tick_params(axis='y', labelcolor='blue')

ax1a.plot(time, y_target_equilibrium, color='red', label='Target Equilibrium')
ax1a.set_ylabel('Target \nEquilibrium', color='red')
ax1a.tick_params(axis='y', labelcolor='red')
ax1a.set_yticks([-1, 1])
ax1a.set_ylim(-1.3, 1.3)

# **Hide x-axis ticks and labels for ax1**
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Optional: Add legends for ax1 and ax1a
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1a.get_legend_handles_labels()

# 6. Third subplot: Angle in degrees
ax2 = fig.add_subplot(gs[2], sharex=ax0)
ax2.plot(time, angle_degrees, color='green', label='Angle')
ax2.set_ylabel('Angle \n(degrees)')
ax2.set_xlabel('Time (s)')

# 7. Tight layout and save
# plt.tight_layout()
plt.savefig('MLP_readout.pdf', format='pdf', bbox_inches='tight')
plt.show()
