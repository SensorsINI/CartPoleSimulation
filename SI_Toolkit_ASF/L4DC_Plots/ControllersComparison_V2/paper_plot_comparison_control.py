import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

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
        'label': 'Baseline MPC, pole: 5cm',
        'color': 'blue',
        'order': 2,
        'feature': 'Q_calculated',
        'on_subplots': [1, 2],
        'SSD': False,
    },
    'Q_calculated_gru_adaptive_fixed_Q': {
        'label': 'Adaptive GRU',
        'color': 'green',
        'order': 3,
        'feature': 'Q_calculated_gru_adaptive_fixed_Q',
        'on_subplots': [1],
        'SSD': True,
    },
    'Q_calculated_gru_memoryless': {
        'label': 'Experience Deprived GRU',
        'color': 'red',
        'order': 1,
        'feature': 'Q_calculated_gru_memoryless',
        'on_subplots': [1, 2],
        'SSD': True,
    },
    'Q_calculated_integrated_mean': {
        'label': 'Average MPC: pole: 5-80cm',
        'color': 'orange',
        'order': 3,
        'feature': 'interp_Q_calculated_integrated_mean',
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

# 5. Set font sizes
plt.rcParams.update({'font.size': 14})

# 6. Create the plots with adjusted subplot heights
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(4, 1, height_ratios=[2, 2, 1, 1], hspace=0.1)

# Collect features for first subplot
first_subplot_features = [config for config in plotting_configs.values() if 1 in config['on_subplots']]
first_subplot_features.sort(key=lambda x: x['order'])

# Collect features for second subplot
second_subplot_features = [config for config in plotting_configs.values() if 2 in config['on_subplots']]
second_subplot_features.sort(key=lambda x: x['order'])

# For legends, collect handles and labels in the specified order
legend_order = ['Q_calculated', 'Q_calculated_gru_adaptive_fixed_Q', 'Q_calculated_gru_memoryless', 'Q_calculated_integrated_mean']
legend_handles = []
legend_labels = []

legend_handles_dict = {}
legend_labels_dict = {}

# First plot
ax0 = fig.add_subplot(gs[0])
x = df_main_filtered['time']
y_Q_calculated = df_main_filtered['Q_calculated']

# Plot features on first subplot
for config in first_subplot_features:
    feature_name = config['feature']
    if feature_name == 'interp_Q_calculated_integrated_mean':
        continue  # Should not be plotted on the first subplot
    y_feature = df_main_filtered[feature_name]
    if config['feature'] == 'Q_calculated_gru_adaptive_fixed_Q':
        linewidth = 2.5
    elif config['feature'] == 'Q_calculated_gru_memoryless':
        linewidth = 2.0
    else:
        linewidth = 1.5  # Default line width
    line, = ax0.plot(x, y_feature, color=config['color'], linewidth=linewidth)

    # Compute SSD if needed
    if config['SSD']:
        SSD = np.sum((y_feature - y_Q_calculated) ** 2)
        label_with_SSD = f"{config['label']} (SSD={SSD:.2f})"
    else:
        label_with_SSD = config['label']

    legend_handles_dict[config['feature']] = line
    legend_labels_dict[config['feature']] = label_with_SSD

ax0.set_ylabel('Control Signal')
ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Second plot
ax1 = fig.add_subplot(gs[1], sharex=ax0)

# Plot features on second subplot
for config in second_subplot_features:
    feature_name = config['feature']
    if feature_name == 'interp_Q_calculated_integrated_mean':
        # Interpolate Q_calculated_integrated_mean onto x
        y_feature = np.interp(x, time_integrated_filtered, Q_integrated_mean_filtered)
    else:
        y_feature = df_main_filtered[feature_name]
    if config['feature'] == 'Q_calculated_gru_memoryless':
        linewidth = 2.0
    else:
        linewidth = 1.5  # Default line width
    line, = ax1.plot(x, y_feature, color=config['color'], linewidth=linewidth)

    # Compute SSD if needed
    if config['SSD']:
        SSD = np.sum((y_feature - y_Q_calculated) ** 2)
        label_with_SSD = f"{config['label']} (SSD={SSD:.2f})"
    else:
        label_with_SSD = config['label']

    legend_handles_dict[config['feature']] = line
    legend_labels_dict[config['feature']] = label_with_SSD

ax1.set_ylabel('Control Signal')
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Collect legend handles and labels in specified order
legend_handles = [legend_handles_dict[feature] for feature in legend_order if feature in legend_handles_dict]
legend_labels = [legend_labels_dict[feature] for feature in legend_order if feature in legend_labels_dict]

# Create legend at the top of the figure
fig.legend(legend_handles, legend_labels, loc='upper center', ncol=4, fontsize=12)

# Third plot: 'target_position'
ax2 = fig.add_subplot(gs[2], sharex=ax0)

y_target_position = df_main_filtered['target_position'].to_numpy() * 100  # Convert meters to centimeters
x_target = x

ax2.plot(x_target, y_target_position, color='blue')
ax2.set_ylabel('Target \nPosition (cm)', color='blue', labelpad=20)
ax2.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Fourth plot: 'angle' in degrees
ax3 = fig.add_subplot(gs[3], sharex=ax0)
angle_degrees = np.degrees(df_main_filtered['angle'])

ax3.plot(x, angle_degrees, color='green')
ax3.set_ylabel('Angle \n(degrees)')
ax3.set_xlabel('Time (s)')

plt.tight_layout()

plt.savefig('ComparisonsControllers.pdf', format='pdf', bbox_inches='tight')

plt.show()
