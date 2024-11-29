import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# 3. Define custom labels for plotting
line_labels = {
    'Q_calculated': 'Calculated Q',
    'Q_calculated_gru_memoryless': 'GRU Memoryless',
    'Q_calculated_gru_adaptive_fixed_Q': 'GRU Adaptive Fixed Q',
    'Q_calculated_integrated_mean': 'Integrated Mean Q',
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

# 6. Create the plots
fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

# First plot
x = df_main_filtered['time']
y_Q_calculated = df_main_filtered['Q_calculated']

axs[0].plot(x, y_Q_calculated, label=line_labels['Q_calculated'])

for feature in ['Q_calculated_gru_memoryless', 'Q_calculated_gru_adaptive_fixed_Q']:
    y_feature = df_main_filtered[feature]
    # Compute SSD with respect to Q_calculated
    SSD = np.sum((y_feature - y_Q_calculated) ** 2)
    # Display SSD in legend with custom label
    label = f"{line_labels.get(feature, feature)} (SSD={SSD:.2f})"
    axs[0].plot(x, y_feature, label=label)

axs[0].set_ylabel('Control Signal')
axs[0].legend(loc='upper right')

# Second plot
# Interpolate Q_calculated_integrated_mean onto x
interp_Q_calculated_integrated_mean = np.interp(x, time_integrated_filtered, Q_integrated_mean_filtered)

y_Q_calculated_gru_memoryless = df_main_filtered['Q_calculated_gru_memoryless']
SSD_gru_memoryless = np.sum((y_Q_calculated_gru_memoryless - y_Q_calculated) ** 2)
label_gru_memoryless = f"{line_labels.get('Q_calculated_gru_memoryless', 'Q_calculated_gru_memoryless')} (SSD={SSD_gru_memoryless:.2f})"

SSD_integrated_mean = np.sum((interp_Q_calculated_integrated_mean - y_Q_calculated) ** 2)
label_integrated_mean = f"{line_labels.get('Q_calculated_integrated_mean', 'Q_calculated_integrated_mean')} (SSD={SSD_integrated_mean:.2f})"

axs[1].plot(x, y_Q_calculated, label=line_labels['Q_calculated'])
axs[1].plot(x, y_Q_calculated_gru_memoryless, label=label_gru_memoryless)
axs[1].plot(x, interp_Q_calculated_integrated_mean, label=label_integrated_mean)

axs[1].set_ylabel('Control Signal')
axs[1].legend(loc='upper right')

# Third plot: 'target_position' and 'target_equilibrium' with secondary y-axis
ax3 = axs[2]
ax3a = ax3.twinx()

y_target_position = df_main_filtered['target_position'] * 100  # Convert meters to centimeters
y_target_equilibrium = df_main_filtered['target_equilibrium']

ax3.plot(x, y_target_position, color='blue')
ax3.set_ylabel('Target Position (cm)', color='blue')
ax3.tick_params(axis='y', labelcolor='blue')

ax3a.plot(x, y_target_equilibrium, color='red')
ax3a.set_ylabel('Target Equilibrium', color='red')
ax3a.tick_params(axis='y', labelcolor='red')
ax3a.set_yticks([-1, 1])

# No legend for third plot as colors and axis labels are sufficient

# Fourth plot: 'angle' in degrees
angle_degrees = np.degrees(df_main_filtered['angle'])

axs[3].plot(x, angle_degrees, color='green')
axs[3].set_ylabel('Angle (degrees)')
axs[3].set_xlabel('Time (s)')

# No legend for fourth plot as only one feature is plotted

plt.tight_layout()
plt.show()
