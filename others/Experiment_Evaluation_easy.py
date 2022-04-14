import csv
import re
import numpy as np
import tensorflow as tf
import os
import glob

import matplotlib.pyplot as plt
import matplotlib
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
matplotlib.use('Qt5Agg')


from CartPole.cartpole_model import TrackHalfLength
from CartPole.state_utilities import (
    ANGLE_COS_IDX,
    ANGLE_IDX,
    ANGLED_IDX,
    ANGLE_SIN_IDX,
    POSITION_IDX,
    POSITIOND_IDX,
    STATE_VARIABLES,
    STATE_INDICES,
    create_cartpole_state,
)

exp_length_idx = 0
sim_dt_idx = 1
cont_dt_idx = 2
saving_dt_idx = 3
controller_idx = 4

m_param_idx = 5
M_param_idx = 6
L_param_idx = 7
u_max_param_idx = 8
M_fric_parm_idx = 9
J_fric_param_idx = 10
v_max_idx = 11
TrackHalfLength_idx = 12
contDist_idx = 13
contBias_idx = 14
g_param_idx = 15
k_param_idx = 16


def data_idx(list):
    ds = re.compile('# Data:')
    for i in range(len(list)):
        if ds.match(list[i][0]):
            return i
    return -1


# %%

lines = []
with open('Experiment_Recordings/CP_mppi-tf_2022-04-14_17-43-20.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        if line:
            lines.append(line)

# %%

beginning = re.compile('#.*:\s*')
second = re.compile('\s*s\Z')
exp_info = []
with_s_idx = [3, 5, 6, 7, 8]
for idx in with_s_idx:
    truncated = beginning.sub('', second.sub('', lines[idx][0]))
    if not truncated:
        exp_info.append(None)
    else:
        exp_info.append(float(truncated))
controller = beginning.sub('', second.sub('', lines[10][0]))
param_idx = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
for idx in param_idx:
    exp_info.append(float(beginning.sub('', lines[idx][0])))

ds = data_idx(lines) + 1

data_legend = lines[ds]
data = np.float32(lines[ds + 1:])
time_idx = data_legend.index('time')
position_idx = data_legend.index('position')
angle_idx = data_legend.index('angle')
u_idx = data_legend.index('u')
angle_cos_idx = data_legend.index('angle_cos')
angle_sin_idx = data_legend.index('angle_sin')


fig, ax1 = plt.subplots(3,1,num='yoyoyo')
ax1 = plt.subplot(3, 1, 1)
plt.plot(data[:, time_idx], data[:, angle_idx])
plt.ylim(-np.pi,np.pi)


ax1 = plt.subplot(3, 1, 2)
plt.plot(data[:, time_idx], data[:, position_idx])
plt.ylim(-exp_info[TrackHalfLength_idx], exp_info[TrackHalfLength_idx])

ax1 = plt.subplot(3, 1, 3)
plt.plot(data[:, time_idx], data[:, u_idx])
plt.ylim(-exp_info[u_max_param_idx], exp_info[u_max_param_idx])
plt.show()

