import numpy as np
import matplotlib.pyplot as plt

foldernames = ['./Ground Truth', './Dense-48', './GRU-16', './Dense-96', './GRU-32-0', './GRU-32-1', './Dense-96-L',
               './GRU-32-L-0', './GRU-32-L-1']


def swing_up_task(file_start=0, file_end=4, folder_number=0, pole_length='L395', fig_size=(6, 8), save=False,
                  save_name=''):
    # create Experiment file names
    if file_start == 0:
        experiments = ['/Experiment.csv']
        file_start += 1
    else:
        experiments = []
    for l in range(file_start, file_end + 1):
        experiments.append('/Experiment-' + str(l) + '.csv')

    sw_up_time = 0

    # plot the sub figures
    fig, axs = plt.subplots(len(experiments), sharex=True, sharey=True, figsize=fig_size)
    for j, file in enumerate(experiments):
        filename = foldernames[folder_number] + file
        data = np.genfromtxt(filename, skip_header=28, delimiter=',')
        time = data[:, 0]
        angle = data[:, 1] * 180 / np.pi
        axs[j].plot(time, angle, marker='.', linestyle='', markersize=2)
        axs[j].set(ylabel='Run ' + str(j + 1))
        axs[j].grid(True)

        # Determine Swing-Up time
        sw_up = False
        sw_try = False
        for k in range(len(angle) - 100):
            if np.abs(angle[k]) < 10:
                sw_try = True
                sw_up = True
                for n in range(100):
                    if np.abs(angle[k + n]) >= 10:  # pole must stay upright for 2s
                        sw_up = False

                if sw_up:
                    sw_up_time += time[k]
                    axs[j].vlines(time[k], ymin=-180, ymax=180, label='Time: ' + str(np.around(time[k], 1)) + 's',
                                  color='tab:orange')
                    axs[j].legend(loc='upper right')
                    break
        if sw_try and sw_up == False:
            sw_up_time += 10  # if try was made count 10s for swing up
            axs[j].vlines(10, ymin=-180, ymax=180, label='Nice try: 10s',
                          color='tab:orange')
            axs[j].legend(loc='upper right')

    sw_up_time /= len(experiments)

    # add title and save figure
    axs[len(experiments) - 1].set(xlabel='time /s')
    fig.suptitle(foldernames[folder_number].replace('./', '') + '\n Angle Plot Swing-Up ' + pole_length +
                 '\n \n Average Swing-Up time: ' + str(np.around(sw_up_time, 1)) + 's')
    plt.tight_layout()
    if save:
        plt.savefig(foldernames[folder_number] + '/Angle Plot Swing-Up ' + pole_length + save_name + '.pdf')
    plt.show()

    return sw_up_time


if __name__ == '__main__':
    # for i in range(9):
    swing_up_task(file_start=0, file_end=9, folder_number=5, pole_length='L395', fig_size=(6.5, 10), save=True,
                  save_name='-1')
