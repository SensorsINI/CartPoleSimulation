import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

PATH_TO_DATA = '../L4DC_Plots_with_data/PCA/'

# 1. Set font sizes (same as the first plot)
plt.rcParams.update({'font.size': 14})

# Load the data
path_to_csv = os.path.join(PATH_TO_DATA, 'PCA_components.csv')
df = pd.read_csv(path_to_csv, comment='#')
df = df.iloc[:-1500, :]  # Slice the data
L = df['L'] * 100
PCA1 = df['PCA1']
PCA2 = df['PCA2']
time = df['time']
time = time - time.min()

fig = plt.figure(figsize=(16, 4))

# Define GridSpec with 2 rows and 2 columns
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.05], hspace=0.7, wspace=0.2)

# -----------------------------
# First Subplot: PCA1 vs L
ax0 = fig.add_subplot(gs[0, 0])
scatter1 = ax0.scatter(L, PCA1, c=time, cmap='coolwarm', alpha=0.7)
ax0.set_xlabel('Pole Length (cm)')
ax0.set_ylabel('PCA1')
ax0.grid(True)

# Add colorbar for the first scatter plot below the plot
cax0 = fig.add_subplot(gs[1, 0])
cbar1 = fig.colorbar(scatter1, cax=cax0, orientation='horizontal')
cbar1.set_label('Time (s)')

# -----------------------------
# Second Subplot: PCA1 vs time
ax1 = fig.add_subplot(gs[0, 1])
scatter2 = ax1.scatter(time, PCA1, c=L, cmap='coolwarm', alpha=0.7)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('PCA1')
ax1.grid(True)

# Add colorbar for the second scatter plot below the plot
cax1 = fig.add_subplot(gs[1, 1])
cbar2 = fig.colorbar(scatter2, cax=cax1, orientation='horizontal')
cbar2.set_label('Pole Length (cm)')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('PCA1.pdf', format='pdf', bbox_inches='tight', bbox_extra_artists=[cax1])

# Display the plots
plt.show()
