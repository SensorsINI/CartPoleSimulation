import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from others.L4DC_Plots.plots_helpers import label_target_position_and_position, break_line_on_jump

POSTER = True  # Set to True for poster style, False for paper style

PATH_TO_DATA = './L4DC_Plots_with_data/SwingUp'

label_mpc = 'Informed MPC, pole: 15cm'
label_informed = 'Adaptive NC'
label_uninformed = 'Experience-Deprived NC'

informed_path = os.path.join(PATH_TO_DATA, 'informed.csv')
uninformed_path = os.path.join(PATH_TO_DATA, 'uninformed.csv')
mpc_path = os.path.join(PATH_TO_DATA, 'mpc.csv')

informed_df = pd.read_csv(informed_path, comment='#')
uninformed_df = pd.read_csv(uninformed_path, comment='#')
mpc_df = pd.read_csv(mpc_path, comment='#')

# 1. Set font sizes

font_size = 21 if POSTER else 14
legend_fontsize = 21 if POSTER else 14


plt.rcParams.update({'font.size': font_size})

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

angle_informed = np.degrees(informed_df['angle'])[time_range_mask].to_numpy()
angle_uninformed = np.degrees(uninformed_df['angle'])[time_range_mask].to_numpy()
angle_mpc = np.degrees(mpc_df['angle'])[time_range_mask].to_numpy()



time_angle = time

t_threshold = 2.0
smale_angle_regime = 15.0

# Primary axis
ax0 = fig.add_subplot(gs[0])

# Mask for data before t_threshold
mask_before = time_angle < t_threshold

time_before_mpc, angle_before_mpc = break_line_on_jump(time_angle[mask_before], angle_mpc[mask_before])
time_before_informed, angle_before_informed = break_line_on_jump(time_angle[mask_before], angle_informed[mask_before])
time_before_uninformed, angle_before_uninformed = break_line_on_jump(time_angle[mask_before], angle_uninformed[mask_before])

# Plot data up to t_threshold on ax0
ax0.plot(time_before_mpc, angle_before_mpc, label=label_mpc, color='blue', zorder=3, linewidth=2.5)
ax0.plot(time_before_informed, angle_before_informed, label=label_informed, color='green', zorder=2, linewidth=5.0)
ax0.plot(time_before_uninformed, angle_before_uninformed, label=label_uninformed, color='red', zorder=1, linewidth=2.5)

# Create a secondary y-axis
ax0a = ax0.twinx()

# Mask for data from t_threshold onwards
mask_after = time_angle >= t_threshold

# Plot data from t_threshold onwards on ax0a
time_after_mpc, angle_after_mpc = break_line_on_jump(time_angle[mask_after], angle_mpc[mask_after])
time_after_informed, angle_after_informed = break_line_on_jump(time_angle[mask_after], angle_informed[mask_after])
time_after_uninformed, angle_after_uninformed = break_line_on_jump(time_angle[mask_after], angle_uninformed[mask_after])

ax0a.plot(time_after_mpc, angle_after_mpc, color='blue', zorder=3, linewidth=2.5)
ax0a.plot(time_after_informed, angle_after_informed, color='green', zorder=2, linewidth=5.0)
ax0a.plot(time_after_uninformed, angle_after_uninformed, color='red', zorder=1, linewidth=2.5)

# Set the limits for the secondary axis (zoomed-in range)
ax0a.set_ylim(-smale_angle_regime, smale_angle_regime)

# Optionally, adjust the x-axis limits if you want to emphasize the zoom region
# For example, keep the full x-axis range
ax0.set_xlim(time_angle.min(), time_angle.max())

# Combine legends from both axes
lines_0, labels_0 = ax0.get_legend_handles_labels()
lines_0a, labels_0a = ax0a.get_legend_handles_labels()
# Since the labels are the same, avoid duplication by only taking from ax0
bbox_to_anchor = (1.02, 1.2) if POSTER else None  # Adjust the second value to move the legend up or down
handletextpad = 0.3 if POSTER else 0.8  # Adjust column spacing for poster or paper style
columnspacing = 1.5 if POSTER else 1.0  # Adjust column spacing for poster or paper style
legend = ax0.legend(lines_0, labels_0,
                    loc='upper right', bbox_to_anchor=bbox_to_anchor,
                    ncol=3, handletextpad=handletextpad, columnspacing=columnspacing,
                    fontsize=legend_fontsize,
                    )
legend.get_frame().set_edgecolor('white')  # Remove border edge color
legend.get_frame().set_alpha(0)  # Make frame transparent

# **1. Make "Adaptive NC" label bold in the legend**
for text in legend.get_texts():
    if text.get_text() == label_informed:
        text.set_fontweight('bold')

# Differentiate the two y-axes
ax0.set_ylabel('Pole Angle\nFull Range (deg)', color='black')
ax0a.set_ylabel('Pole Angle\nZoomed (deg)', color='black')

# Optional: Add vertical line at t_threshold to indicate the change
ax0.axvline(x=t_threshold, color='gray', linestyle='--', linewidth=3)

# Optional: Add horizontal lines to indicate zoom limits on ax0a
ax0a.axhline(y=smale_angle_regime, color='gray', linestyle=':', linewidth=0.5)
ax0a.axhline(y=-smale_angle_regime, color='gray', linestyle=':', linewidth=0.5)
ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# 5. Second subplot: Position
ax1 = fig.add_subplot(gs[1], sharex=ax0)

ax1.set_ylabel('Cart Position\n(cm)')

position_informed = 100.0 * informed_df['position'][time_range_mask]
position_uninformed = 100.0 * uninformed_df['position'][time_range_mask]
position_mpc = 100.0 * mpc_df['position'][time_range_mask]

target_position_informed = 100.0 * informed_df['target_position'][time_range_mask]
target_position_uninformed = 100.0 * uninformed_df['target_position'][time_range_mask]
target_position_mpc = 100.0 * mpc_df['target_position'][time_range_mask]

# **2. Plot lines in the same order as the first subplot with increased linewidth**
# Plot Target first (assuming it's a reference line)
ax1.plot(time, target_position_mpc, color='black', linestyle='--', label='Target', linewidth=3.0)

# Plot mpc, informed, and uninformed in the specified order
ax1.plot(time, position_mpc, color='blue', linewidth=2.0, zorder=3)
ax1.plot(time, position_informed, color='green', linewidth=3.5, zorder=2)
ax1.plot(time, position_uninformed, color='red', linewidth=2.0, zorder=1)

# Create legend including 'Target'
legend1 = ax1.legend(fontsize=legend_fontsize, loc='upper right')
legend1.get_frame().set_edgecolor('white')  # Optional: Remove border edge color
legend1.get_frame().set_alpha(0)  # Optional: Make frame transparent

ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# 6. Third subplot: Control
ax2 = fig.add_subplot(gs[2], sharex=ax0)

control_mpc = mpc_df['Q_calculated'][time_range_mask]
control_informed = informed_df['Q_calculated'][time_range_mask]
control_uninformed = uninformed_df['Q_calculated'][time_range_mask]

# **3. Plot lines in the same order as the first subplot with increased linewidth**
ax2.plot(time, control_mpc, color='blue', label=label_mpc, linewidth=2.0, zorder=3)
ax2.plot(time, control_informed, color='green', label=label_informed, linewidth=3.0, zorder=2)

time_swing_up, control_uninformed_swing_up = time[mask_before], control_uninformed[mask_before]
ax2.axvline(x=t_threshold, color='gray', linestyle='--', linewidth=3)


# time_swing_up, control_uninformed_swing_up = time, control_uninformed
ax2.plot(time_swing_up, control_uninformed_swing_up, color='red', label=label_uninformed, linewidth=2.0, zorder=1)

y_label = 'Control\nSignal' if POSTER else 'Control Signal'
ax2.set_ylabel(y_label, labelpad=15)
ax2.set_xlabel('Time (s)')

plt.savefig('SwingUp.pdf', format='pdf', bbox_inches='tight')


plt.show()