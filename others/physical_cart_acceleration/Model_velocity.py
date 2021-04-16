import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Change backend for matplotlib to plot interactively in pycharm
# This import must go before pyplot
from matplotlib import use
# use('TkAgg')
use('macOSX')


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# file_path = 'cart_acceleration_friction.csv'
file_path = 'cart_acceleration.csv'

# Load data
data: pd.DataFrame = pd.read_csv(file_path, comment='#')

# try:
#     data['dt'] = data['deltaTimeMs']/1000.0
#     data['Q'] = data['actualMotorCmd']
# except:
#     pass


# Calculate dt
data['dt'] = data['time'].shift(-1) - data['time']
data['positionD_last'] = data['positionD'].shift(1)
data['positionD_smoothed'] = smooth(data['positionD'], 30)
data = data.iloc[1:-1]
data = data.reset_index(drop=True)

# Parameter guess
a = 48
b = 45.0

def acceleration_first_guess(Q,v):
    return a*Q-b*v

Q = data['Q'].to_numpy()
dt = data['dt'].to_numpy()
time = data['time'].to_numpy()
v = np.zeros_like(time)
for i in range(len(v)-1):
    if i == 0:
        v[0] = 0.0
    v[i+1] = v[i]+(a*Q[i]-b*v[i])*dt[i]


fontsize_labels = 14
fontsize_ticks = 12

# fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
#
# axs[0].set_ylabel("Speed (m/s)", fontsize=fontsize_labels)
# axs[0].plot(data['time'], data['positionD'],
#             'g', markersize=12, label='Velocity')
# axs[0].plot(time, v,
#             'orange', markersize=12, label='Velocity modeled')
# axs[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
# axs[0].legend()
#
# axs[1].set_ylabel("Motor Input Q (-)", fontsize=fontsize_labels)
# axs[1].plot(data['time'], data['Q'],
#             'r', markersize=12, label='Q')
# axs[1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
#
# axs[1].set_xlabel('Time (s)', fontsize=fontsize_labels)
#
# fig.align_ylabels()
# plt.tight_layout()
# plt.show()


gb = data.groupby(['Q'], as_index=False)
data_stat = gb.size().reset_index()
# data_stat['v_max'] = gb['positionD'].max()['positionD']
data_stat['v_max'] = gb['positionD_smoothed'].max()['positionD_smoothed']

data_stat = data_stat[data_stat['Q'] > 0]
data_stat_linear = data_stat[data_stat['Q'] < 0.5]

motor_for_fit = data_stat_linear['Q'].to_numpy().reshape(-1, 1)
v_max_for_fit = data_stat_linear['v_max'].to_numpy().reshape(-1, 1)
reg = LinearRegression().fit(motor_for_fit, v_max_for_fit)

slope = reg.coef_[0][0]
intercept_y = reg.intercept_[0]
intercept_x = -intercept_y/slope
print('Coefficients of linear regression:')
print('Slope:')
print(np.around(slope,2))
print('Intercept Y:')
print(np.around(intercept_y,2))
print('Intercept X:')
print(np.around(intercept_x,2))

motor_for_preduction = data_stat['Q'].to_numpy().reshape(-1, 1)
v_max_predicted = reg.predict(motor_for_preduction)

# plt.figure(figsize=(16, 9))
# plt.plot(data_stat['Q'], data_stat['v_max'], marker='o', label='Measured v_max')
# plt.plot(data_stat['Q'], v_max_predicted, marker='', label='Linear regression for small v_max')
# plt.xlabel('Motor input Q [-]')
# plt.ylabel('Maximal reached speed [m/s]')
# plt.legend()
# plt.show()


# Second guess, including classincal friction:

# Parameter guess
b = 20.0
a = 1.4*b
c = 0.08*b

v = np.zeros_like(time)
for i in range(len(v)-1):
    if i == 0:
        v[0] = 0.0
    v[i+1] = v[i]+(a*Q[i]-b*v[i]-c*np.sign(v[i]))*dt[i]


fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

axs[0].set_ylabel("Speed (m/s)", fontsize=fontsize_labels)
# axs[0].plot(data['time'], data['positionD_smoothed'],
#             'g', markersize=12, label='Velocity')
axs[0].plot(data['time'], data['positionD'],
            'g', markersize=12, label='Velocity')
axs[0].plot(time, v,
            'orange', markersize=12, label='Velocity modeled')
axs[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

axs[1].set_ylabel("Motor Input Q (-)", fontsize=fontsize_labels)
axs[1].plot(data['time'], data['Q'],
            'r', markersize=12, label='Q')
axs[1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

axs[1].set_xlabel('Time (s)', fontsize=fontsize_labels)

fig.align_ylabels()

plt.tight_layout()

plt.show()





# Third guess, including correction with v^2

# Parameter guess
b = 20.0
a = 1.4*b
c = 0.08*b
d = 10.0




v = np.zeros_like(time)
for i in range(len(v)-1):
    if i == 0:
        v[0] = 0.0
    v[i+1] = v[i]+(a*Q[i]-b*v[i]-c*np.sign(v[i])-d*(v[i]**2)*np.sign(v[i]))*dt[i]


# fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
#
# axs[0].set_ylabel("Speed (m/s)", fontsize=fontsize_labels)
# # axs[0].plot(data['time'], data['positionD_smoothed'],
# #             'g', markersize=12, label='Velocity')
# axs[0].plot(data['time'], data['positionD'],
#             'g', markersize=12, label='Velocity')
# axs[0].plot(time, v,
#             'orange', markersize=12, label='Velocity modeled')
# axs[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
#
# axs[1].set_ylabel("Motor Input Q (-)", fontsize=fontsize_labels)
# axs[1].plot(data['time'], data['Q'],
#             'r', markersize=12, label='Q')
# axs[1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
#
# axs[1].set_xlabel('Time (s)', fontsize=fontsize_labels)
#
# fig.align_ylabels()
#
# plt.tight_layout()
#
# plt.show()
