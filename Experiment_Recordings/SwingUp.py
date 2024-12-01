import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from SI_Toolkit_ASF.L4DC_Plots.plots_helpers import label_target_position_and_position

label_mpc = 'Baseline MPC, pole: 15cm'
label_informed = 'GRU - Adaptive'
label_uninformed = 'GRU - Memoryless'

informed_df = pd.read_csv('./informed.csv', comment='#')
uninformed_df = pd.read_csv('./uninformed.csv', comment='#')
mpc_df = pd.read_csv('./mpc.csv', comment='#')

# 1. Set font sizes
fontsize = 14
plt.rcParams.update({'font.size': fontsize})

# 2. Define time range for plotting
time_start = 0  # Start time in seconds
time_end = 30  # End time in seconds
time = informed_df['time']
time_range_mask = (time >= time_start) & (time <= time_end)
time = time[time_range_mask]
time = time - time.iloc[0]
time = time.to_numpy()

# 3. Create the plots with adjusted subplot heights
fig = plt.figure(figsize=(16, 8))  # Increased height for better spacing
gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.1)

# 4. First subplot: Angle

angle_informed = np.degrees(informed_df['angle'])[time_range_mask]
angle_uninformed = np.degrees(uninformed_df['angle'])[time_range_mask]
angle_mpc = np.degrees(mpc_df['angle'])[time_range_mask]

t_threshold = 2.0
smale_angle_regime = 15.0

# Primary axis
ax0 = fig.add_subplot(gs[0])

# Mask for data before t_threshold
mask_before = time < t_threshold

# Plot data up to t_threshold on ax0
ax0.plot(time[mask_before], angle_mpc[mask_before], label=label_mpc, color='orange', zorder=1)
ax0.plot(time[mask_before], angle_informed[mask_before], label=label_informed, color='green', zorder=1)
ax0.plot(time[mask_before], angle_uninformed[mask_before], label=label_uninformed, color='red', zorder=1)

# Set the limits for the primary axis
ax0.set_xlim(time.min(), time.max())
ax0.set_ylim(min(angle_mpc[mask_before].min(), angle_informed[mask_before].min(), angle_uninformed[mask_before].min()),
            max(angle_mpc[mask_before].max(), angle_informed[mask_before].max(), angle_uninformed[mask_before].max()))

# Create a secondary y-axis
ax0a = ax0.twinx()

# Mask for data from t_threshold onwards
mask_after = time >= t_threshold

# Plot data from t_threshold onwards on ax0a
ax0a.plot(time[mask_after], angle_mpc[mask_after], color='orange', zorder=2)
ax0a.plot(time[mask_after], angle_informed[mask_after], color='green', zorder=2)
ax0a.plot(time[mask_after], angle_uninformed[mask_after], color='red', zorder=2)

# Set the limits for the secondary axis (zoomed-in range)
ax0a.set_ylim(-smale_angle_regime, smale_angle_regime)

# Optionally, adjust the x-axis limits if you want to emphasize the zoom region
# For example, keep the full x-axis range
ax0.set_xlim(time.min(), time.max())

# Combine legends from both axes
lines_0, labels_0 = ax0.get_legend_handles_labels()
lines_0a, labels_0a = ax0a.get_legend_handles_labels()
# Since the labels are the same, avoid duplication by only taking from ax0
legend = ax0.legend(lines_0, labels_0, loc='center', bbox_to_anchor=(0.8, 1.15), ncol=3)
legend.get_frame().set_edgecolor('white')  # Remove border edge color
legend.get_frame().set_alpha(0)  # Make frame transparent

# Differentiate the two y-axes
ax0.set_ylabel('Angle (Full Range)', color='black')
ax0a.set_ylabel('Angle (Zoomed)', color='black')

# Optional: Add vertical line at t_threshold to indicate the change
ax0.axvline(x=t_threshold, color='gray', linestyle='--', linewidth=1)

# Optional: Add horizontal lines to indicate zoom limits on ax0a
ax0a.axhline(y=smale_angle_regime, color='gray', linestyle=':', linewidth=0.5)
ax0a.axhline(y=-smale_angle_regime, color='gray', linestyle=':', linewidth=0.5)


# 5. Second subplot: Position
ax1 = fig.add_subplot(gs[1], sharex=ax0)
label_target_position_and_position(ax1, 'black', 'blue', fontsize)

position_informed = 100.0 * informed_df['position'][time_range_mask]
position_uninformed = 100.0 * uninformed_df['position'][time_range_mask]
position_mpc = 100.0 * mpc_df['position'][time_range_mask]

target_position_informed = 100.0 * informed_df['target_position'][time_range_mask]
target_position_uninformed = 100.0 * uninformed_df['target_position'][time_range_mask]
target_position_mpc = 100.0 * mpc_df['target_position'][time_range_mask]

ax1.plot(time, target_position_mpc, color='blue', label='Target Position')

ax1.plot(time, position_mpc, color='orange', label=label_mpc)
ax1.plot(time, position_informed, color='green', label=label_informed)
ax1.plot(time, position_uninformed, color='red', label=label_uninformed)

ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# 6. Third subplot: Control
ax2 = fig.add_subplot(gs[2], sharex=ax0)

control_mpc = mpc_df['Q_calculated'][time_range_mask]
control_informed = informed_df['Q_calculated'][time_range_mask]
control_uninformed = uninformed_df['Q_calculated'][time_range_mask]

ax2.plot(time, control_mpc, color='orange', label=label_mpc)
ax2.plot(time, control_informed, color='green', label=label_informed)
ax2.plot(time, control_uninformed, color='red', label=label_uninformed)

ax2.set_ylabel('Control Signal', labelpad=15)
ax2.set_xlabel('Time (s)')

plt.show()