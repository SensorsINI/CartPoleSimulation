#file to evaluate experiment data

import csv
import re
import numpy as np
import tensorflow as tf
import os
import glob

import matplotlib.pyplot as plt
import matplotlib
import sys
# from others.cost_functions.CartPole.quadratic_boundary_grad import q as stage_cost #use correct stage cost here, probably need to slightly adjust cost in controller
from others.cost_functions.CartPole.quadratic_boundary_grad import q_debug as stage_cost
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
matplotlib.use('Qt5Agg')

## Imports and definitions
def runnig_avg(costs, horizon):
    filt = tf.ones([horizon, 1, 1], dtype=tf.float32)
    return tf.squeeze(tf.nn.conv1d(costs[:, :, tf.newaxis], filt, 1, 'SAME', data_format="NWC")) / filt.shape[0]


def swingup_time_calc(S, target_pos, TrackHalfLength):
    invalid = tf.cast((tf.abs(S[..., ANGLE_IDX]) > 20*np.pi/180) \
                      | (tf.abs(S[..., POSITION_IDX] - target_pos) > 0.15 * TrackHalfLength), tf.float32)
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
######################################################################################################
"""Enter name of experiment and relevant data"""
######################################################################################################
# %% extract all data from all experiments
Expname = 'Exp-mppi-optimize-swingup-nn-A'
isSwingup = True #is it a swingup experiment?
clipExpNum = True #do you want to show all experiments or only a subset?
ExpClipNum = 100 #how many do you want to show?
CherryPick = False #Cherry pick a single experiment to evaluate?
CherryPickNum = 2 #which one?


#create paths and directorys
path = 'Experiment_Recordings/'+Expname+'*.csv' #save path of recording
savepath = 'Experiment_Setups/'+Expname+'/' #directory for experiment plots
os.makedirs(savepath, exist_ok = True)

#find all relevant files
files = glob.glob(path)
print("{} experiments total".format(len(files)))

""" ***********************************
    Start of import procedure
    ***********************************"""
#import all data
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
#%% #weed out to short experiments
for experiment in all_data:
    if len(experiment) < exp_tick_length:
        all_data.remove(experiment)
print("{} experiments usable".format(len(all_data)))
if clipExpNum:
    all_data = all_data[0:ExpClipNum]
if CherryPick:
    all_data = all_data[CherryPickNum:CherryPickNum+1]

#%%
#extract experiment info
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

#set up some variables that describe the experiment
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

Q = all_data[..., u_idx]/exp_info[u_max_param_idx]

Q = tf.constant(Q, dtype=tf.float32)
S = tf.constant(S, dtype=tf.float32)
num_rol = all_data.shape[0]


""" ***********************************
    End of import procedure
    ***********************************"""


#%%
costs, dd_cost, ep_cost, cc_cost, ccrc_cost = stage_cost(S, Q, target_pos, Q[0, 0])



#%%evaluation
""" ***********************************
    Evaluate different measures here!!!
    ***********************************"""

# cost parts
ravg = runnig_avg(costs, 20) #running cost over a window
avg_cost = tf.math.reduce_mean(costs)
avg_dd = tf.math.reduce_mean(dd_cost)
avg_ep = tf.math.reduce_mean(ep_cost)
avg_cc = tf.math.reduce_mean(cc_cost)
avg_ccrc = tf.math.reduce_mean(ccrc_cost)
avg_cost_per = tf.math.reduce_mean(costs, axis = 1)
print(avg_cost.numpy())

#%%
#mean running cost
ravg_mean = tf.math.reduce_mean(ravg, axis = 0)

#for swingup extract swingup time and succesful swingups
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
    print("Mean swingup time: {}".format(swingup_mean))

#%%
#plot cost
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

#plot all relevant things for a single experiment
fig1, ax1 = plt.subplots(4, 1, num='Example plot', figsize = (16,12))
#angle plot
ax1 = plt.subplot(4, 1, 1)
plt.plot(data[:, time_idx], data[:,angle_idx])
plt.axhline(y = 0.34, color = 'r')
plt.axhline(y = -0.34, color = 'r')
plt.ylim(-np.pi*paf, np.pi*paf)

#
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
#end of example plot

#%% plot all angles at once
fig2, ax2 = plt.subplots(1, 1, num='Only angles', figsize = (16,12))
plt.title('Angles')
plt.plot(all_data[0,:, time_idx], np.swapaxes(all_data[..., angle_idx],0,1))
plt.axhline(y = 0.34, color = 'r')
plt.axhline(y = -0.34, color = 'r')
plt.ylim(-np.pi*paf, np.pi*paf)
plt.savefig(savepath+'Angles.png', bbox_inches='tight',dpi = 200)


#%% plot all positions
fig3, ax3 = plt.subplots(1, 1, num='Only position', figsize = (16,12))
plt.title('Positions')
plt.plot(all_data[0,:, time_idx], np.swapaxes(all_data[..., position_idx],0,1))
# plt.axhline(y = 0.34, color = 'r')
# plt.axhline(y = -0.34, color = 'r')
plt.ylim(-exp_info[TrackHalfLength_idx]*paf, exp_info[TrackHalfLength_idx]*paf)
plt.savefig(savepath+'Positions.png', bbox_inches='tight',dpi = 200)


#%% plot all motor powers
fig4, ax4 = plt.subplots(1, 1, num='Only mot power', figsize = (16,12))
plt.title('Motor Powers')
plt.plot(all_data[0,:,time_idx], np.swapaxes(all_data[...,u_idx],0,1))
plt.ylim(-exp_info[u_max_param_idx]*paf, exp_info[u_max_param_idx]*paf)
plt.axhline(y = exp_info[u_max_param_idx], color = 'r')
plt.axhline(y = -exp_info[u_max_param_idx], color = 'r')
plt.savefig(savepath+'Motor_power.png', bbox_inches='tight',dpi = 200)



#%% plot averaged cost over time
fig5, ax5 = plt.subplots(1, 1, num='Averaged cost', figsize = (16,12))
plt.title('Averaged cost')
plt.semilogy(all_data[0,:,time_idx], tf.math.reduce_mean(costs, axis = 0))
plt.savefig(savepath+'Avg_cost.png', bbox_inches='tight',dpi = 200)



#%% first report plot: position and angles
fig6, ax6 = plt.subplots(2, 1, num='Rap. plot 1', figsize = (16,12))
ax61 = plt.subplot(2,1,1)
ax61.set_ylabel('Position (m)')
#plt.title('Positions')
plt.axhline(y = exp_info[TrackHalfLength_idx], color = 'r')
plt.axhline(y = -exp_info[TrackHalfLength_idx], color = 'r')
plt.axhline(y = 0.15 * TrackHalfLength, color = 'darkorange')
plt.axhline(y = -0.15 * TrackHalfLength, color = 'darkorange')
plt.plot(all_data[0,:, time_idx], np.swapaxes(all_data[..., position_idx],0,1))
plt.ylim(-exp_info[TrackHalfLength_idx]*paf, exp_info[TrackHalfLength_idx]*paf)

ax62 = plt.subplot(2,1,2)
ax62.set_ylabel('Angle (deg)')
ax62.set_xlabel('Time (s)')
# plt.title('Angles')
plt.plot(all_data[0,:, time_idx], np.swapaxes(all_data[..., angle_idx],0,1)*180/np.pi)
plt.axhline(y = 20, color = 'darkorange')
plt.axhline(y = -20, color = 'darkorange')
plt.ylim(-np.pi*paf, np.pi*paf)
plt.yticks([-180, -90, 0, 90, 180])
plt.savefig(savepath+'rapport1.svg', bbox_inches='tight',dpi = 200)


#%% second report plot, positions and motorpower
fig7, ax7 = plt.subplots(2, 1, num='Rap. plot 2', figsize = (16,12))
ax71 = plt.subplot(2,1,1)
ax71.set_ylabel('Position (m)')
#plt.title('Positions')
plt.axhline(y = exp_info[TrackHalfLength_idx], color = 'r')
plt.axhline(y = -exp_info[TrackHalfLength_idx], color = 'r')
plt.plot(all_data[0,:, time_idx], target_pos[0,:], color = "darkorange")
plt.plot(all_data[0,:, time_idx], np.swapaxes(all_data[..., position_idx],0,1))
plt.ylim(-exp_info[TrackHalfLength_idx]*paf, exp_info[TrackHalfLength_idx]*paf)

ax72 = plt.subplot(2,1,2)
ax72.set_ylabel('Motorpower (%)')
ax72.set_xlabel('Time (s)')
# plt.title('Angles')
plt.plot(all_data[0,:,time_idx], np.swapaxes(all_data[...,u_idx],0,1)/exp_info[u_max_param_idx]*100.0)
plt.axhline(y = 100, color = 'r')
plt.axhline(y = -100, color = 'r')
plt.ylim(-100*paf, 100*paf)
plt.savefig(savepath+'rapport2.svg', bbox_inches='tight',dpi = 200)

plt.show()
pass

