import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')

#Path to Experiment Recording
folder_path = "swingup_experiments/adaptive_imitator/"
experiment_path = '007L.csv'


def plot_angle(data, controller, pole_length, start_time, end_time, max_y=200, min_y=-200):
    recording_section = data[(data['time'] >= start_time) & (data['time'] <= end_time)]
    plt.figure(figsize=(14, 7))
    plt.plot(recording_section['time'], recording_section['angle'], label= 'Angle')
    plt.plot(recording_section['time'], np.zeros(recording_section['time'].shape), linestyle='dotted', label='Target Equilibrium')
    title = 'Swing-Up Experiment\n Controller: ' + str(controller) + '\n L/2: ' + str(pole_length)
    #plt.title(title)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Angle [rad]', fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(min_y, max_y)
    plt.tight_layout()
    plt.savefig(folder_path + 'angle_plot_'+ controller + '_'+str(pole_length)+'.png', dpi=600)

def plot_position(data, controller, pole_length, start_time, end_time, max_y=200, min_y=-200):
    recording_section = data[(data['time'] >= start_time) & (data['time'] <= end_time)]
    plt.figure(figsize=(14, 7))
    plt.plot(recording_section['time'], recording_section['position'], label= 'Position')
    plt.plot(recording_section['time'], np.zeros(recording_section['time'].shape), linestyle='dotted', label='Target Position')
    title = 'Swing-Up Experiment\n Controller: ' + str(controller) + '\n L/2: ' + str(pole_length)
    #plt.title(title)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Position [m]', fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(min_y, max_y)
    plt.tight_layout()
    plt.savefig(folder_path + 'position_plot_'+ controller + '_'+str(pole_length)+'.png', dpi=600)

def plot_speed(mppi_data, adaptive_imitator_data, cascaded_imitator_data, start_time, end_time):
    recording_section_mppi = mppi_data[(mppi_data['time'] >= start_time) & (mppi_data['time'] <= end_time)]
    recording_section_ai = adaptive_imitator_data[(adaptive_imitator_data['time'] >= start_time) & (adaptive_imitator_data['time'] <= end_time)]
    recording_section_ci = cascaded_imitator_data[(cascaded_imitator_data['time'] >= start_time) & (cascaded_imitator_data['time'] <= end_time)]
    plt.figure(figsize=(18, 6))
    average_mppi = np.mean(recording_section_mppi['Q_update_time'])
    average_ai = np.mean(recording_section_ai['Q_update_time'])
    average_ci = np.mean(recording_section_ci['Q_update_time'])
    plt.hist(recording_section_mppi['Q_update_time'], color='red', density=False, bins=60, alpha=0.7)
    plt.hist(recording_section_ai['Q_update_time'], color='blue', density=False, bins=60, alpha=0.7)
    plt.hist(recording_section_ci['Q_update_time'], color='green', density=False, bins=60, alpha=0.7)
    plt.axvline(average_mppi, color='red', linestyle='--', linewidth=2, label='MPPI Average')
    plt.axvline(average_ai, color='blue', linestyle='--', linewidth=2, label='Adaptive Imitator Average')
    plt.axvline(average_ci, color='green', linestyle='--', linewidth=2, label='Cascaded Imitator Average')

    plt.text(average_mppi, plt.ylim()[1], f'Avg: {average_mppi:.5f}s', color='red', va='bottom', ha='left', fontsize=18)
    plt.text(average_ai, plt.ylim()[1], f'Avg: {average_ai:.5f}s', color='blue', va='bottom', ha='left', fontsize=18)
    plt.text(average_ci, plt.ylim()[1], f'Avg: {average_ci:.5f}s', color='green', va='bottom', ha='left', fontsize=18)

    plt.xlabel('Update Time [s]', fontsize=16)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=18, loc='upper center')
    plt.ylim(0, 400)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(folder_path + 'speed_plot.png', dpi=600)




def swingup_experiment(ex_path, fol_path, controller, pole_length, runtime):
    if controller == 'MPPI':
        skip_rows = 28
    else:
        skip_rows = 27
    dataframe = pd.read_csv(fol_path + ex_path, skiprows=skip_rows)
    plot_angle(dataframe, controller, pole_length, 0, runtime, 4, -4)
    plot_position(dataframe, controller, pole_length, 0, runtime, 0.3, -0.3)
    plot_speed(dataframe, controller, pole_length, 0, runtime)

#swingup_experiment(experiment_path, folder_path, 'Static Imitator', 0.07, 20)
dataframe1 = pd.read_csv('swingup_experiments/mppi_informed/01975L.csv', skiprows=28)
dataframe3 = pd.read_csv('swingup_experiments/adaptive_imitator/01975L.csv', skiprows=27)
dataframe4 = pd.read_csv('swingup_experiments/cascaded_imitator/01975L.csv', skiprows=27)
plot_speed(dataframe1,dataframe3,dataframe4,0,20)




