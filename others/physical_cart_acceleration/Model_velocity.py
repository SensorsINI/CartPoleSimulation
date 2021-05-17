import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Change backend for matplotlib to plot interactively in pycharm
# This import must go before pyplot
from matplotlib import use
# use('TkAgg')
use('macOSX')

MOTOR_RANGE = 8192.0  # One side, it changes in [-MOTOR_RANGE, MOTOR_RANGE]

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def process_one_sided_file(file_path, side='pos', Q_normed=True):
    # Load data
    data: pd.DataFrame = pd.read_csv(file_path, comment='#')

    # Calculate dt
    data['dt'] = data['time'].shift(-1) - data['time']
    data['positionD_last'] = data['positionD'].shift(1)
    data['positionD_smoothed'] = smooth(data['positionD'], 30)
    data = data.iloc[1:-1]
    data = data.reset_index(drop=True)

    # Parameter guess
    slope = 0.000113

    if side == 'pos':
        b = 20.0
        a = slope * b * MOTOR_RANGE
    else:
        b = 20.0
        a = slope * b * MOTOR_RANGE


    def acceleration_first_guess(Q,v):
        return a*Q-b*v

    if Q_normed:
        data['Q'] = data['Q']*MOTOR_RANGE

    Q = data['Q'].to_numpy()
    dt = data['dt'].to_numpy()
    time = data['time'].to_numpy()
    v = np.zeros_like(time)
    for i in range(len(v)-1):
        if i == 0:
            v[0] = 0.0
        v[i+1] = v[i]+acceleration_first_guess(Q[i]/MOTOR_RANGE, v[i])*dt[i]


    fontsize_labels = 14
    fontsize_ticks = 12

    fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    axs[0].set_ylabel("Speed (m/s)", fontsize=fontsize_labels)
    axs[0].plot(data['time'], data['positionD_smoothed'],
                'g', markersize=12, label='Velocity')
    axs[0].plot(time, v,
                'orange', markersize=12, label='Velocity modeled')
    axs[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    axs[0].legend()

    axs[1].set_ylabel("Motor Input Q (-)", fontsize=fontsize_labels)
    axs[1].plot(data['time'], data['Q'],
                'r', markersize=12, label='Q')
    axs[1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    axs[1].set_xlabel('Time (s)', fontsize=fontsize_labels)

    fig.align_ylabels()
    plt.tight_layout()
    plt.show()


    gb = data.groupby(['Q'], as_index=False)
    data_stat = gb.size().reset_index()
    # data_stat['v_max'] = gb['positionD'].max()['positionD']

    if side == 'pos':
        data_stat['v_max'] = gb['positionD_smoothed'].max()['positionD_smoothed']
        data_stat = data_stat[data_stat['Q'] > 0]
        data_stat_linear = data_stat[data_stat['Q'] < MOTOR_RANGE/2.2   ]
    else:
        data_stat['v_max'] = gb['positionD_smoothed'].min()['positionD_smoothed']
        data_stat = data_stat[data_stat['Q'] < 0]
        data_stat_linear = data_stat[data_stat['Q'] > -MOTOR_RANGE/2]

    motor_for_fit = data_stat_linear['Q'].to_numpy().reshape(-1, 1)
    v_max_for_fit = data_stat_linear['v_max'].to_numpy().reshape(-1, 1)
    reg = LinearRegression().fit(motor_for_fit, v_max_for_fit)

    slope = reg.coef_[0][0]
    intercept_y = reg.intercept_[0]
    intercept_x = -intercept_y/slope
    print('Coefficients of linear regression:')
    print('Slope:')
    print(np.around(slope,7))
    print('Intercept Y:')
    print(np.around(intercept_y,2))
    print('Intercept X:')
    print(np.around(intercept_x,2))

    motor_for_prediction = data_stat['Q'].to_numpy().reshape(-1, 1)
    motor_for_prediction = np.concatenate((-500*np.ones(shape=(1, 1)), np.zeros(shape=(1, 1)), 500*np.ones(shape=(1, 1)), motor_for_prediction))
    v_max_predicted = reg.predict(motor_for_prediction)

    # plt.figure(figsize=(16, 9))
    # plt.plot(data_stat['Q'], data_stat['v_max'], marker='o', label='Measured v_max')
    # plt.plot(motor_for_prediction, v_max_predicted, marker='', label='Linear regression for small v_max')
    # plt.xlabel('Motor input Q [-]')
    # plt.ylabel('Maximal reached speed [m/s]')
    # plt.legend()
    # plt.show()

    Q = data_stat['Q'].to_numpy()[:, np.newaxis]
    v = data_stat['v_max'].to_numpy()[:, np.newaxis]
    Q_pred = motor_for_prediction


    return Q, v, Q_pred, v_max_predicted

# Without correction
file_path_neg = 'cartpole-2021-05-03-17-29-30_leftstep_wpole.csv'
file_path_pos = 'cartpole-2021-05-03-17-32-44_rightstep_wpole.csv'
Q_pos, v_pos, Q_pos_pred, v_pred_pos = process_one_sided_file(file_path_pos, side='pos', Q_normed=False)
Q_neg, v_neg, Q_neg_pred, v_pred_neg = process_one_sided_file(file_path_neg, side='neg', Q_normed=False)

# # With velocity dependent correction
# file_path_neg = 'left-velocity-2-cartpole-2021-05-14-22-51-04.csv'
# file_path_pos = 'right-velocity-2-cartpole-2021-05-14-22-53-04.csv'
# Q_pos, v_pos, Q_pos_pred, v_pred_pos = process_one_sided_file(file_path_pos, side='pos')
# Q_neg, v_neg, Q_neg_pred, v_pred_neg = process_one_sided_file(file_path_neg, side='neg')

# # With motro input dependent correction
# file_path_neg = 'leftstep_motorCmd_cartpole-2021-05-14-20-19-12.csv'
# file_path_pos = 'rightstep_motorCmd_cartpole-2021-05-14-20-16-22.csv'
# Q_pos, v_pos, Q_pos_pred, v_pred_pos = process_one_sided_file(file_path_pos, side='pos')
# Q_neg, v_neg, Q_neg_pred, v_pred_neg = process_one_sided_file(file_path_neg, side='neg')

# plt.figure(figsize=(16, 9))
# plt.plot(Q_pos, v_pos, marker='o', label='Measured v_saturation')
# # plt.plot(Q_pos_pred, v_pred_pos, marker='', label='Linear regression for small v_saturation')
# plt.plot(Q_neg, v_neg, marker='o', label='Measured v_saturation')
# # plt.plot(Q_neg_pred, v_pred_neg, marker='', label='Linear regression for small v_saturation')
# plt.plot(Q_ideal, v_ideal, marker='', label='Linear regression for small v_saturation')
# plt.xlabel('Motor input Q [-]')
# plt.ylabel('Maximal reached speed [m/s]')
# # plt.title('Before correction')
# plt.title('Velocity corrected')
# # plt.title('Motor power corrected')
# plt.legend()
# plt.grid()
# plt.show()

Q_ideal = np.array([min(Q_neg), max(Q_pos)])
slope = 0.000113
v_ideal = slope*Q_ideal

plt.figure(figsize=(16, 9))
# plt.plot(Q_pos-495, v_pos, marker='o', label='Measured v_saturation')
# plt.plot(Q_neg+365, v_neg, marker='o', label='Measured v_saturation')
plt.plot(Q_pos+0, v_pos, marker='o', label='Measured v_saturation')
plt.plot(Q_neg-0, v_neg, marker='o', label='Measured v_saturation')
# plt.plot(Q_pos_pred-645, v_pred_pos, marker='', label='Linear regression for small v_saturation')
# plt.plot(Q_neg_pred+514, v_pred_neg, marker='', label='Linear regression for small v_saturation')
# plt.plot(Q_pos_pred+155, v_pred_pos, marker='', label='Linear regression for small v_saturation')
# plt.plot(Q_neg_pred-101, v_pred_neg, marker='', label='Linear regression for small v_saturation')
plt.plot(Q_ideal, v_ideal, marker='', label='Linear regression for small v_saturation')
plt.xlabel('Motor input Q [-]')
plt.ylabel('Maximal reached speed [m/s]')
plt.title('Before correction')
plt.legend()
plt.grid()
plt.show()




    #
    # # Second guess, including classincal friction:
    #
    # # Parameter guess
    # b = 20.0
    # a = 0.98*b
    # c = 0.05*b
    #
    # print('a = {}'.format(a))
    # print('b = {}'.format(b))
    # print('c = {}'.format(c))
    #
    # v = np.zeros_like(time)
    # for i in range(len(v)-1):
    #     if i == 0:
    #         v[0] = 0.0
    #     v[i+1] = v[i]+(a*Q[i]-b*v[i]-c*np.sign(v[i]))*dt[i]
    #
    #
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
    #
    #
    #
    #
    #
    # # Third guess, including correction with v^2
    #
    # # Parameter guess
    # b = 20.0
    # a = 1.4*b
    # c = 0.08*b
    # d = 10.0
    #
    #
    #
    #
    # v = np.zeros_like(time)
    # for i in range(len(v)-1):
    #     if i == 0:
    #         v[0] = 0.0
    #     v[i+1] = v[i]+(a*Q[i]-b*v[i]-c*np.sign(v[i])-d*(v[i]**2)*np.sign(v[i]))*dt[i]
    #
    #
    # # fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    # #
    # # axs[0].set_ylabel("Speed (m/s)", fontsize=fontsize_labels)
    # # # axs[0].plot(data['time'], data['positionD_smoothed'],
    # # #             'g', markersize=12, label='Velocity')
    # # axs[0].plot(data['time'], data['positionD'],
    # #             'g', markersize=12, label='Velocity')
    # # axs[0].plot(time, v,
    # #             'orange', markersize=12, label='Velocity modeled')
    # # axs[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    # #
    # # axs[1].set_ylabel("Motor Input Q (-)", fontsize=fontsize_labels)
    # # axs[1].plot(data['time'], data['Q'],
    # #             'r', markersize=12, label='Q')
    # # axs[1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    # #
    # # axs[1].set_xlabel('Time (s)', fontsize=fontsize_labels)
    # #
    # # fig.align_ylabels()
    # #
    # # plt.tight_layout()
    # #
    # # plt.show()
