import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from SI_Toolkit_ASF.L4DC_Plots.plots_helpers import break_line_on_jump


# 1. Set font sizes (same as the first plot)
plt.rcParams.update({'font.size': 14})
legend_fontsize = 14

# 1. Load 'cardinal_test_1.csv' and extract features
df_main = pd.read_csv('./cardinal_test_1.csv', comment='#')

# List of features from 'cardinal_test_1.csv' to plot
main_features = [
    'Q_calculated',
    'Q_calculated_gru_adaptive_fixed_Q',
    'Q_calculated_gru_memoryless'
]

# Clip Q values to [-1, 1] in df_main
for feature in main_features:
    df_main[feature] = df_main[feature].clip(-1, 1)

# 2. Load 'Q_calculated_integrated' from 'cardinal_test_1_1.csv' to 'cardinal_test_1_8.csv'
integrated_files = [f'./cardinal_test_1_{i}.csv' for i in range(1, 9)]
df_integrated_list = [pd.read_csv(f, comment='#') for f in integrated_files]

# Extract 'time' and 'Q_calculated_integrated' from each file
time_integrated = df_integrated_list[0]['time']  # Assuming all have the same time column
Q_integrated_all = np.array([df['Q_calculated_integrated'].values for df in df_integrated_list])

# Clip Q_integrated_all to [-1, 1]
Q_integrated_all = np.clip(Q_integrated_all, -1, 1)

# Compute average and standard deviation
Q_integrated_mean = Q_integrated_all.mean(axis=0)
Q_integrated_std = Q_integrated_all.std(axis=0)

# 3. Define custom labels, colors, and plotting order for features
plotting_configs = {
    'Q_calculated': {
        'label': 'Informed MPC, pole: 5cm',
        'color': 'blue',
        'order': 2,
        'feature': 'Q_calculated',
        'on_subplots': [1, 2],
        'SSD': False,
    },
    'Q_calculated_gru_adaptive_fixed_Q': {
        'label': 'Adaptive NC',
        'color': 'green',
        'order': 3,
        'feature': 'Q_calculated_gru_adaptive_fixed_Q',
        'on_subplots': [1],
        'SSD': True,
    },
    'Q_calculated_gru_memoryless': {
        'label': 'Experience Deprived NC',
        'color': 'red',
        'order': 1,
        'feature': 'Q_calculated_gru_memoryless',
        'on_subplots': [1, 2],
        'SSD': True,
    },
    'Q_calculated_integrated_mean': {
        'label': 'Average MPC: pole: 5-80cm',
        'color': 'darkorange',
        'order': 3,
        'feature': 'Q_calculated_integrated_mean',
        'on_subplots': [2],
        'SSD': True,
    },
}

# 4. Set time range up to 6 seconds
t_max = 6.0

# Filter df_main and time_integrated based on t_max
df_main_filtered = df_main[df_main['time'] <= t_max]
time_integrated_filtered = time_integrated[time_integrated <= t_max]

# For Q_integrated_all and Q_integrated_mean and Q_integrated_std, we need to filter accordingly
mask_integrated = time_integrated <= t_max
Q_integrated_all_filtered = Q_integrated_all[:, mask_integrated]
Q_integrated_mean_filtered = Q_integrated_mean[mask_integrated]
Q_integrated_std_filtered = Q_integrated_std[mask_integrated]
time_integrated_filtered = time_integrated[mask_integrated]

# 6. Create the plots with adjusted subplot heights
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(4, 1, height_ratios=[2, 2, 1, 1], hspace=0.1)

# Collect features for first subplot
first_subplot_features = [config for config in plotting_configs.values() if 1 in config['on_subplots']]
first_subplot_features.sort(key=lambda x: x['order'])

# Collect features for second subplot
second_subplot_features = [config for config in plotting_configs.values() if 2 in config['on_subplots']]
second_subplot_features.sort(key=lambda x: x['order'])

# Define the legend order
legend_order = ['Q_calculated', 'Q_calculated_gru_adaptive_fixed_Q', 'Q_calculated_gru_memoryless', 'Q_calculated_integrated_mean']

# First plot
ax0 = fig.add_subplot(gs[0])
x = df_main_filtered['time']
y_Q_calculated = df_main_filtered['Q_calculated']

# Plot features on first subplot
for config in first_subplot_features:
    feature_name = config['feature']
    if feature_name == 'Q_calculated_integrated_mean':
        continue  # Should not be plotted on the first subplot
    y_feature = df_main_filtered[feature_name]
    if config['feature'] == 'Q_calculated_gru_adaptive_fixed_Q':
        linewidth = 2.5
    elif config['feature'] == 'Q_calculated_gru_memoryless':
        linewidth = 2.0
    else:
        linewidth = 1.5  # Default line width
    line, = ax0.plot(x, y_feature, color=config['color'], linewidth=linewidth)

ax0.set_ylabel('Control Signal')
ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Second plot
ax1 = fig.add_subplot(gs[1], sharex=ax0)

# Plot features on second subplot
for config in second_subplot_features:
    feature_name = config['feature']
    if feature_name == 'Q_calculated_integrated_mean':
        # Interpolate Q_calculated_integrated_mean onto x
        y_feature = np.interp(x, time_integrated_filtered, Q_integrated_mean_filtered)
    else:
        y_feature = df_main_filtered[feature_name]
    if config['feature'] == 'Q_calculated_gru_memoryless':
        linewidth = 2.0
    else:
        linewidth = 1.5  # Default line width
    line, = ax1.plot(x, y_feature, color=config['color'], linewidth=linewidth)

ax1.set_ylabel('Control Signal')
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Collect legend handles and labels manually
from matplotlib.lines import Line2D

legend_handles = []
legend_labels = []

for feature in legend_order:
    config = plotting_configs[feature]
    color = config['color']
    label = config['label']

    # Compute SSD if needed
    if config['SSD']:
        # Need to get y_feature and y_Q_calculated
        if feature == 'Q_calculated_integrated_mean':
            y_feature = np.interp(df_main_filtered['time'], time_integrated_filtered, Q_integrated_mean_filtered)
        else:
            y_feature = df_main_filtered[feature]
        y_Q_calculated = df_main_filtered['Q_calculated']
        SSD = np.sum((y_feature - y_Q_calculated) ** 2)
        label_with_SSD = f"{label} (SSD={SSD:.2f})"
    else:
        label_with_SSD = label

    legend_labels.append(label_with_SSD)

    # Create a Line2D object
    line = Line2D([0], [0], color=color, linewidth=2)
    legend_handles.append(line)

# Create legend at the top of the figure
legend = fig.legend(
    legend_handles,
    legend_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.97),  # Adjust the second value to move the legend up or down
    ncol=2,
    fontsize=legend_fontsize
)
legend.get_frame().set_edgecolor('white')  # Remove border edge color
legend.get_frame().set_alpha(0)  # Make frame transparent

# Third plot: 'target_position'
ax2 = fig.add_subplot(gs[2], sharex=ax0)

y_target_position = df_main_filtered['target_position'].to_numpy() * 100  # Convert meters to centimeters
y_position = df_main_filtered['position'].to_numpy() * 100  # Convert meters to centimeters
time_target = x

position_color = 'blue'
target_position_color = 'black'

ax2.plot(time_target, y_target_position, color=target_position_color, label='Target', linestyle='--')
ax2.plot(time_target, y_position, color=position_color, label='Actual', linewidth=2.5)

ax2.set_ylabel('Cart Position\n(cm)', labelpad=25)

legend2 = ax2.legend(fontsize=legend_fontsize)
legend2.get_frame().set_edgecolor('white')  # Remove border edge color
legend2.get_frame().set_alpha(0)  # Make frame transparent

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Fourth plot: 'angle' in degrees
ax3 = fig.add_subplot(gs[3], sharex=ax0)
angle_degrees = np.degrees(df_main_filtered['angle'].to_numpy())

# Break the line on jumps in the angle
time_modified, angle_degrees_modified = break_line_on_jump(x.to_numpy(), angle_degrees, threshold=90.0)

y_target_equilibrium = df_main_filtered['target_equilibrium'].to_numpy()
target_angle_up = np.where(y_target_equilibrium == 1, 0, 180)
target_angle_down = np.where(y_target_equilibrium == 1, 0, -180)


ax3.plot(x, target_angle_up, color='black', linestyle='--', label='Target')
ax3.plot(x, target_angle_down, color='black', linestyle='--')
ax3.plot(time_modified, angle_degrees_modified, color='blue', label='Actual', linewidth=2.5)
ax3.set_ylabel('Pole Angle \n(degrees)')
ax3.set_xlabel('Time (s)')

legend3 = ax3.legend(fontsize=legend_fontsize)
legend3.get_frame().set_edgecolor('white')  # Remove border edge color
legend3.get_frame().set_alpha(0)  # Make frame transparent

plt.tight_layout()

plt.savefig('ComparisonsControllers.pdf', format='pdf', bbox_inches='tight')

plt.show()
