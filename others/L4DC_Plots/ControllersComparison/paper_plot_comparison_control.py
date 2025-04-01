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
    'Q_calculated_large_parameters',
    'Q_calculated_gru_adaptive_2',
    'Q_calculated_gru_adaptive_fixed_Q',
    'Q_calculated_dense',
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
        'order': 1,
        'feature': 'Q_calculated',
        'on_subplots': [1, 2],
        'SSD': False,
    },
    'Q_calculated_gru_memoryless': {
        'label': 'Experience-Deprived GRU\n      ',
        'color': 'red',
        'order': 2,
        'feature': 'Q_calculated_gru_memoryless',
        'on_subplots': [1, 2],
        'SSD': True,
    },
    'Q_calculated_gru_adaptive_fixed_Q': {
        'label': 'Adaptive GRU\n',
        'color': 'green',
        'order': 3,
        'feature': 'Q_calculated_gru_adaptive_fixed_Q',
        'on_subplots': [1],
        'SSD': True,
    },
    'Q_calculated_integrated_mean': {
        'label': 'Average MPC\npole: 5-80cm\n',
        'color': 'orange',
        'order': 4,
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

# First plot
ax0 = fig.add_subplot(gs[0])
x = df_main_filtered['time']
y_Q_calculated = df_main_filtered['Q_calculated']

# For legends
legend_handles_upper_right = []
legend_labels_upper_right = []
legend_handles_lower_right = []
legend_labels_lower_right = []

# Plot features on first subplot
for config in first_subplot_features:
    feature_name = config['feature']
    if feature_name == 'interp_Q_calculated_integrated_mean':
        continue  # Should not be plotted on the first subplot
    y_feature = df_main_filtered[feature_name]
    if "Adaptive" in plotting_configs[feature_name]['label']:
        linewidth = 2.5
    else:
        linewidth = None
    line, = ax0.plot(x, y_feature, color=config['color'], linewidth=linewidth)

    # Manage legends
    # Compute SSD and add to lower right legend with SSD
    if config['SSD']:
        SSD = np.sum((y_feature - y_Q_calculated) ** 2)
        label_with_SSD = f"{config['label']}(SSD={SSD:.2f})"
    else:
        label_with_SSD = config['label']
    if config['on_subplots'] == [1, 2]:
        # Add to upper right legend without SSD
        legend_handles_upper_right.append(line)
        legend_labels_upper_right.append(label_with_SSD)
    elif config['on_subplots'] == [1]:
        legend_handles_lower_right.append(line)
        legend_labels_lower_right.append(label_with_SSD)

ax0.set_ylabel('Control Signal')

class CustomLegendHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        if isinstance(orig_handle, Line2D):
            # Create a line with the same properties
            line = Line2D([x0, x0 + width], [y0 + 1.5*height, y0 + 1.5*height],
                          color=orig_handle.get_color(),
                          linewidth=orig_handle.get_linewidth(),
                          linestyle=orig_handle.get_linestyle(),
                          label=orig_handle.get_label())
            line.set_transform(trans)
            return [line]
        else:
            return super().create_artists(legend, orig_handle, x0, y0, width, height, fontsize, trans)

# Create a handler_map where only the second handle uses CustomLegendHandler
handler_map = {
    legend_handles_upper_right[1]: CustomLegendHandler()
}

# Add legends to first subplot
legend_upper = ax0.legend(
    legend_handles_upper_right,
    legend_labels_upper_right,
    loc='upper right',
    fontsize=12,
    handler_map=handler_map
)

legend_upper.get_frame().set_edgecolor('white')  # Remove border edge color
legend_upper.get_frame().set_alpha(0)        # Make frame transparent
ax0.add_artist(legend_upper)

if legend_handles_lower_right:
    legend_lower = ax0.legend(legend_handles_lower_right, legend_labels_lower_right, loc='lower right', fontsize=12)
    legend_lower.get_frame().set_edgecolor('white')  # Remove border edge color
    legend_lower.get_frame().set_alpha(0)  # Make frame transparent

ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Second plot
ax1 = fig.add_subplot(gs[1], sharex=ax0)

# For legends
legend_handles_second_lower_right = []
legend_labels_second_lower_right = []

# Plot features on second subplot
for config in second_subplot_features:
    feature_name = config['feature']
    if feature_name == 'interp_Q_calculated_integrated_mean':
        # Interpolate Q_calculated_integrated_mean onto x
        y_feature = np.interp(x, time_integrated_filtered, Q_integrated_mean_filtered)
    else:
        y_feature = df_main_filtered[feature_name]
    line, = ax1.plot(x, y_feature, color=config['color'])

    # Manage legends
    if config['on_subplots'] == [1, 2]:
        # Do not add legend here; it's already added in the first subplot
        pass
    elif config['on_subplots'] == [2]:
        # Compute SSD and add to lower right legend with SSD
        if config['SSD']:
            SSD = np.sum((y_feature - y_Q_calculated) ** 2)
            label_with_SSD = f"{config['label']}(SSD={SSD:.2f})"
        else:
            label_with_SSD = config['label']
        legend_handles_second_lower_right.append(line)
        legend_labels_second_lower_right.append(label_with_SSD)

ax1.set_ylabel('Control Signal')
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


# Add legend to second subplot
if legend_handles_second_lower_right:
    legend_second_lower = ax1.legend(legend_handles_second_lower_right, legend_labels_second_lower_right,
                                     loc='lower right', fontsize=12)
    legend_second_lower.get_frame().set_edgecolor('white')  # Remove border edge color
    legend_second_lower.get_frame().set_alpha(0)        # Make frame transparent

# Third plot: 'target_position' and 'target_equilibrium' with secondary y-axis
ax2 = fig.add_subplot(gs[2])
ax2a = ax2.twinx()

# Update ax2 limits dynamically when ax0 limits change
width_factor = 0.93  # Adjust this value between 0 and 1 to control the width
# Flag to avoid recursion
updating = False

# Callback for updating ax2 when ax0 changes
def update_ax2_limits(event):
    global updating
    if not updating:
        updating = True
        limits_a0 = ax0.get_xlim()
        new_x_lim = (limits_a0[0], limits_a0[0] + width_factor * (limits_a0[1] - limits_a0[0]))
        ax2.set_xlim(new_x_lim)
        updating = False

# Callback for updating ax0 when ax2 changes
def update_ax0_limits(event):
    global updating
    if not updating:
        updating = True
        ax0.set_xlim(ax2.get_xlim())
        updating = False

# Connect callbacks
ax0.callbacks.connect('xlim_changed', update_ax2_limits)
ax2.callbacks.connect('xlim_changed', update_ax0_limits)

update_ax2_limits(None)

# # Adjust the width of ax
pos = ax2.get_position()
new_width = pos.width * width_factor
new_pos = [pos.x0, pos.y0, new_width, pos.height]
ax2.set_position(new_pos)
ax2a.set_position(new_pos)  # Ensure the twin axis matches ax2

subset_length = int(width_factor * len(x))
y_target_position = df_main_filtered['target_position'].to_numpy() * 100  # Convert meters to centimeters
y_target_equilibrium = df_main_filtered['target_equilibrium'].to_numpy()

y_target_position = y_target_position[:subset_length]
y_target_equilibrium = y_target_equilibrium[:subset_length]
x_target = x[:subset_length]


ax2.plot(x_target, y_target_position, color='blue')
ax2.set_ylabel('Target \nPosition (cm)', color='blue', labelpad=20)
ax2.tick_params(axis='y', labelcolor='blue')

# Adjust target equilibrium to prevent overlap
ax2a.plot(x_target, y_target_equilibrium, color='red')
ax2a.set_ylabel('Target \nEquilibrium', color='red')
ax2a.tick_params(axis='y', labelcolor='red')
ax2a.set_yticks([-1, 1])
ax2a.set_ylim(-1.3, 1.3)  # Slightly expand y-limits
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
