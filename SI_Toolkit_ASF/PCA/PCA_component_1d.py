import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec



df = pd.read_csv('PCA_components.csv')
# df = df[(df.m_pole == 0.15) | (df.m_pole == 0.015)]
df = df.iloc[:-1500, :]
L = df['L']
# m_pole = df['m_pole']
# PCA on hidden states
PCA1 = df['PCA1']
PCA2 = df['PCA2']
time = df['time']
angle = df['angle']
Q = df['Q_calculated']

# Create a figure with a specific size
fig = plt.figure(figsize=(12, 9))

# Define GridSpec with 3 rows and 2 columns
# The second column is reserved for colorbars with a fixed width
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1],
                       wspace=0.3, hspace=0.4)

# -----------------------------
# First Subplot: PCA1 vs L
ax0 = fig.add_subplot(gs[0, 0])
scatter1 = ax0.scatter(L, PCA1, c=time, cmap='coolwarm', alpha=0.7)
ax0.set_title('PCA1 vs L')
ax0.set_xlabel('L')
ax0.set_ylabel('PCA1')
ax0.grid(True)

# Add colorbar for the first scatter plot
cax0 = fig.add_subplot(gs[0, 1])
cbar1 = fig.colorbar(scatter1, cax=cax0)
cbar1.set_label('time')

# -----------------------------
# Second Subplot: PCA1 vs time
ax1 = fig.add_subplot(gs[1, 0])
scatter2 = ax1.scatter(time, PCA1, c=L, cmap='coolwarm', alpha=0.7)
ax1.set_title('PCA1 vs time')
ax1.set_xlabel('time')
ax1.set_ylabel('PCA1')
ax1.grid(True)

# Add colorbar for the second scatter plot
cax1 = fig.add_subplot(gs[1, 1])
cbar2 = fig.colorbar(scatter2, cax=cax1)
cbar2.set_label('L')

# -----------------------------
# Third Subplot: angle vs time (No colorbar)
ax2 = fig.add_subplot(gs[2, 0])
ax2.plot(time, angle, color='green', alpha=0.7)
ax2.set_title('angle vs time')
ax2.set_xlabel('time')
ax2.set_ylabel('angle')
ax2.grid(True)


# ax2 = fig.add_subplot(gs[2, 0])
# ax2.plot(time, Q, color='green', alpha=0.7)
# ax2.set_title('Q vs time')
# ax2.set_xlabel('time')
# ax2.set_ylabel('Q')
# ax2.grid(True)
# If you want to align the third subplot's width with the others, add an empty subplot
# in the GridSpec's second column for the third row
ax3 = fig.add_subplot(gs[2, 1])
ax3.axis('off')  # Hide the empty subplot

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()