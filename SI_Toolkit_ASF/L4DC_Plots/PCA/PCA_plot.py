import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

# 1. Set font sizes (same as the first plot)
plt.rcParams.update({'font.size': 14})

df = pd.read_csv('PCA_components.csv')

df = df.iloc[:-1500, :]
L = df['L']*100

PCA1 = df['PCA1']
PCA2 = df['PCA2']
time = df['time']
time = time - time.min()

fig = plt.figure(figsize=(16, 8))

# Define GridSpec with 3 rows and 2 columns
# The second column is reserved for colorbars with a fixed width
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.05, 1, 0.05], height_ratios=[1],
                       wspace=0.3, hspace=0.4)

# -----------------------------
# First Subplot: PCA1 vs L
ax0 = fig.add_subplot(gs[0, 0])
scatter1 = ax0.scatter(L, PCA1, c=time, cmap='coolwarm', alpha=0.7)
ax0.set_xlabel('Pole Length (cm)')
ax0.set_ylabel('PCA1')
ax0.grid(True)

# Add colorbar for the first scatter plot
cax0 = fig.add_subplot(gs[0, 1])
cbar1 = fig.colorbar(scatter1, cax=cax0)
cbar1.set_label('time')

# -----------------------------
# Second Subplot: PCA1 vs time
ax1 = fig.add_subplot(gs[0, 2])
scatter2 = ax1.scatter(time, PCA1, c=L, cmap='coolwarm', alpha=0.7)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('PCA1')
ax1.grid(True)

# Add colorbar for the second scatter plot
cax1 = fig.add_subplot(gs[0, 3])
cbar2 = fig.colorbar(scatter2, cax=cax1)
cbar2.set_label('Pole Length (cm)')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()