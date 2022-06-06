import csv
import re
import numpy as np
import tensorflow as tf
import os
import glob

import matplotlib.pyplot as plt
import matplotlib
import sys
from others.cost_functions.quadratic_boundary import q as stage_cost #use correct stage cost here, probably need to slightly adjust cost in controller

# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
matplotlib.use('Qt5Agg')


def runnig_avg(costs, horizon):
    filt = tf.ones([horizon, 1, 1], dtype=tf.float32)
    return tf.squeeze(tf.nn.conv1d(costs[:, :, tf.newaxis], filt, 1, 'SAME', data_format="NWC")) / filt.shape[0]


def swingup_time_calc(S, target_pos, TrackHalfLength):
    invalid = tf.cast((tf.abs(S[..., ANGLE_IDX]) > 0.34) \
                      | (tf.abs(S[..., POSITION_IDX] - target_pos) > 1.0 * TrackHalfLength), tf.float32)
    swingup_test = tf.cumsum(invalid, axis=1, reverse=True).numpy()
    violation = np.argwhere(np.flip(swingup_test, axis=1) == 1)
    _, last_ind = np.unique(violation[:, 0], return_index=True)
    last_crossings = violation[last_ind, 1]
    upright_time = last_crossings * exp_info[cont_dt_idx]
    return exp_info[exp_length_idx] - upright_time

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


# %% extract all data from all experiments
Expname = 'Exp-mppi-optimize-swingup-A'
isSwingup = True


path = 'Experiment_Recordings/'+Expname+'*.csv'
savepath = 'Experiment_Setups/'+Expname+'/'
os.makedirs(savepath, exist_ok = True)


files = glob.glob(path)
print("{} experiments total".format(len(files)))
files = files[0:20]


all_data = []
exp_tick_length = 0
for file in files:
    lines = []
    with open(file, mode='r') as file_read:
        csvFile = csv.reader(file_read)
        for line in csvFile:
            if line:
                lines.append(line)
    ds = data_idx(lines) + 1
    all_data.append(lines[ds + 1:])
    if len(lines[ds + 1:]) > exp_tick_length:
        exp_tick_length = len(lines[ds + 1:])
#%%
for experiment in all_data:
    if len(experiment) < exp_tick_length:
        all_data.remove(experiment)
#%%
beginning = re.compile('#.*:\s*')
second = re.compile('\s*s\Z')
exp_info = []
with_s_idx = [3, 6, 7, 8]
for idx in with_s_idx:
    truncated = beginning.sub('', second.sub('', lines[idx][0]))
    if not truncated:
        exp_info.append(None)
    else:
        exp_info.append(float(truncated))
controller = beginning.sub('', second.sub('', lines[10][0]))
exp_info.append(controller)
param_idx = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
for idx in param_idx:
    exp_info.append(float(beginning.sub('', lines[idx][0])))

data_legend = lines[ds]
time_idx = data_legend.index('time')
position_idx = data_legend.index('position')
angle_idx = data_legend.index('angle')
u_idx = data_legend.index('u')
Q_idx = data_legend.index('Q')
angle_cos_idx = data_legend.index('angle_cos')
angle_sin_idx = data_legend.index('angle_sin')
angleD_idx = data_legend.index('angleD')
positionD_idx = data_legend.index('positionD')
target_pos_idx = data_legend.index('target_position')
TrackHalfLength = exp_info[TrackHalfLength_idx]

all_data = np.float32(all_data)

# %% Reformat to correct shape and calculate stage cost: Important use costfunction from controller file
S = np.empty(shape=(all_data.shape[0], all_data.shape[1], 6))
S[..., ANGLE_COS_IDX] = all_data[..., angle_cos_idx]
S[..., ANGLE_SIN_IDX] = all_data[..., angle_sin_idx]
S[..., POSITION_IDX] = all_data[..., position_idx]
S[..., ANGLE_IDX] = all_data[..., angle_idx]
S[..., POSITIOND_IDX] = all_data[..., positionD_idx]
S[..., ANGLED_IDX] = all_data[..., angleD_idx]

target_pos = tf.constant(all_data[..., target_pos_idx], dtype=tf.float32)

Q = all_data[..., Q_idx]

Q = tf.constant(Q, dtype=tf.float32)
S = tf.constant(S, dtype=tf.float32)
num_rol = all_data.shape[0]

#%%
costs = stage_cost(S, Q, target_pos, Q[0, 0])

#%%evaluation
""" ***********************************
    Evaluate different measures here!!!
    ***********************************"""


ravg = runnig_avg(costs, 20)
avg_cost = tf.math.reduce_mean(costs)
avg_cost_per = tf.math.reduce_mean(costs, axis = 1)
print(avg_cost.numpy())

#%%
ravg_mean = tf.math.reduce_mean(ravg, axis = 0)

if isSwingup:
    swingup_time = swingup_time_calc(S, target_pos, TrackHalfLength)
    successful_swingups = swingup_time < 5.0
    swcount = np.sum(successful_swingups)
    print("{} of {} swingups successful".format(swcount, S.shape[0]))
    swingup_time = swingup_time[successful_swingups]
    swingup_mean = np.mean(swingup_time)
    swingup_std = np.std(swingup_time)

    figHist, axHist = plt.subplots(1,1,figsize = (16,12))
    n, bins, edges = axHist.hist(swingup_time, bins= 10, ec='black')
    plt.axvline(x=swingup_mean, color='r')
    plt.xticks(bins)
    plt.savefig(savepath + 'swingup_times.png', bbox_inches='tight', dpi=200)

#%%
figHist1, axHist1 = plt.subplots(1,1, num='running cost', figsize = (16,12))
n, bins, edges = axHist1.hist(avg_cost_per.numpy(), bins = 10, ec='black')
plt.xticks(bins)
# figHist1.figure(figsize = (16,12))
plt.title('Total cost distribution')
plt.axvline(x = avg_cost, color = 'r')
plt.savefig(savepath+'avg_cost_histo.png', bbox_inches='tight',dpi = 200)

# %% Example for plotting
data = all_data[0]
paf = 1.1

fig1, ax1 = plt.subplots(4, 1, num='Example plot', figsize = (16,12))

ax1 = plt.subplot(4, 1, 1)
plt.plot(data[:, time_idx], data[:,angle_idx])
plt.axhline(y = 0.34, color = 'r')
plt.axhline(y = -0.34, color = 'r')
plt.ylim(-np.pi*paf, np.pi*paf)

ax1 = plt.subplot(4, 1, 2)
plt.plot(data[:, time_idx], data[:, position_idx])
plt.ylim(-exp_info[TrackHalfLength_idx]*paf, exp_info[TrackHalfLength_idx]*paf)

ax1 = plt.subplot(4, 1, 3)
plt.plot(data[:, time_idx], data[:, u_idx])
plt.ylim(-exp_info[u_max_param_idx]*paf, exp_info[u_max_param_idx]*paf)

plt.subplot(4, 1, 4)
plt.semilogy(data[:, time_idx], costs[0, :])
plt.locator_params(axis='y', numticks=4)

# plt.subplot(5, 1, 5)
# plt.semilogy(data[:, time_idx], ravg[0, :])
# plt.locator_params(axis='y', numticks=4)

plt.savefig(savepath+'example_plot.png', bbox_inches='tight',dpi = 200)


#%%
fig2, ax2 = plt.subplots(1, 1, num='Only angles', figsize = (16,12))
plt.title('Angles')
plt.plot(all_data[0,:, time_idx], np.swapaxes(all_data[..., angle_idx],0,1))
plt.axhline(y = 0.34, color = 'r')
plt.axhline(y = -0.34, color = 'r')
plt.ylim(-np.pi*paf, np.pi*paf)
plt.savefig(savepath+'Angles', bbox_inches='tight',dpi = 200)

#%%
fig3, ax3 = plt.subplots(1, 1, num='Only position', figsize = (16,12))
plt.title('Positions')
plt.plot(all_data[0,:, time_idx], np.swapaxes(all_data[..., position_idx],0,1))
# plt.axhline(y = 0.34, color = 'r')
# plt.axhline(y = -0.34, color = 'r')
plt.ylim(-exp_info[TrackHalfLength_idx]*paf, exp_info[TrackHalfLength_idx]*paf)
plt.savefig(savepath+'Positions.png', bbox_inches='tight',dpi = 200)

#%%
fig4, ax4 = plt.subplots(1, 1, num='Only mot power', figsize = (16,12))
plt.title('Motor Powers')
plt.plot(all_data[0,:,time_idx], np.swapaxes(all_data[...,u_idx],0,1))
plt.ylim(-exp_info[u_max_param_idx]*paf, exp_info[u_max_param_idx]*paf)
plt.axhline(y = exp_info[u_max_param_idx], color = 'r')
plt.axhline(y = -exp_info[u_max_param_idx], color = 'r')
plt.savefig(savepath+'Motor_power.png', bbox_inches='tight',dpi = 200)



#%%
fig5, ax5 = plt.subplots(1, 1, num='Averaged cost', figsize = (16,12))
plt.title('Averaged cost')
plt.semilogy(all_data[0,:,time_idx], tf.math.reduce_mean(costs, axis = 0))
plt.savefig(savepath+'Avg_cost.png', bbox_inches='tight',dpi = 200)
plt.show()
pass

