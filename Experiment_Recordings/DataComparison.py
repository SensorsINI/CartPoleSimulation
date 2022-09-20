import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

"""Generates and saves Plots of the swing-up task from data files as well as generates plots of discrete fourier
 transforms."""

foldernames = ['./Ground Truth', './Dense-48', './GRU-16', './Dense-96', './GRU-32-0', './GRU-32-1', './GRU-32-2',
               './Dense-96-L', './GRU-32-L-0', './GRU-32-L-1', './GRU-32-L-3', './GRU-32-L-5', './GRU-32-new',
               './GRU-32-new-2', './GRU-32-new-2 single', './Dense new-2', './Dense new-2 single']


def swing_up_task(file_start=0, file_end=4, folder_number=0, pole_length='L395', fig_size=(6, 8), save=False,
                  save_name=''):
    # create Experiment file names
    if file_start == 0:
        experiments = ['/Experiment.csv']
        file_start += 1
    else:
        experiments = []
    for m in range(file_start, file_end + 1):
        experiments.append('/Experiment-' + str(m) + '.csv')

    sw_up_time = 0

    # plot the sub figures
    fig, axs = plt.subplots(len(experiments), sharex=True, sharey=True, figsize=fig_size, dpi=200)
    for j, file in enumerate(experiments):
        filename = foldernames[folder_number] + file
        data = np.genfromtxt(filename, skip_header=28, delimiter=',')[500:]
        time = data[:, 0]
        angle = data[:, 1] * 180 / np.pi
        axs[j].plot(time, angle, marker='.', linestyle='', markersize=2)
        axs[j].set(ylabel='Run ' + str(j + 1))
        axs[j].grid(True)

        # Determine Swing-Up time
        sw_up = False
        sw_try = False
        for k in range(len(angle) - 75):
            if np.abs(angle[k]) < 10:
                sw_try = True
                sw_up = True
                for n in range(75):
                    if np.abs(angle[k + n]) >= 10:  # pole must stay upright for 1.5s
                        sw_up = False

                if sw_up:
                    sw_up_time += time[k]
                    axs[j].vlines(time[k], ymin=-180, ymax=180, label='Time: ' + str(np.around(time[k], 1)) + 's',
                                  color='tab:orange')
                    axs[j].legend(loc='upper right')
                    break
        if sw_try and sw_up == False:
            sw_up_time += 10  # if try was made count 10/20 s for swing up
            axs[j].vlines(10, ymin=0, ymax=1, label='Attempt',
                          color='tab:orange')
            axs[j].legend(loc='upper right')

        if not sw_try:
            sw_up_time += 30
    if sw_up_time > 80:
        sw_up_time = 0.0
    else:
        sw_up_time /= len(experiments)

    # add title and save figure
    axs[len(experiments) - 1].set(xlabel='time /s')
    fig.suptitle('Angle Plot Swing-Up ' + pole_length + '\n') \
        # + '\n \n Average Swing-Up time: ' + str(np.around(sw_up_time, 1))+ 's')
    plt.tight_layout()

    if save:
        # plt.savefig(foldernames[folder_number] + '/' + foldernames[folder_number].replace('./', '') + ' Angle Plot Swing-Up ' + pole_length + save_name + '.pdf')
        plt.savefig(
            foldernames[folder_number].replace('./', '') + ' Angle Plot Swing-Up ' + pole_length + save_name + '.png')
    plt.show()

    return sw_up_time


def fourier_transforms(file, fig_size=(6, 4), save=False):
    filename = './Fouriertransforms/' + file + '.csv'

    data = np.genfromtxt(filename, skip_header=28, delimiter=',')
    time = data[:, 0]
    angle = data[:, 1] * 180 / np.pi
    position = data[:, 6]

    # convert the angle to have the pole swing about the 0 angle at upside down position
    f = lambda x: np.sign(x) * 180 - x
    angle = f(angle)

    # perform fouriertransform
    fft = rfft(angle)
    freq = rfftfreq(len(angle), 0.02)
    maxima = np.abs(fft) > max(np.abs(fft)) - max(np.abs(fft)) / 10  # top 10 per cent
    peaks = freq[maxima]
    peaks_rounded = np.around(peaks, 3)

    fig, axs = plt.subplots(2, dpi=300, figsize=fig_size)
    axs[0].plot(time, angle, marker='.', linestyle='', markersize=2)
    axs[0].set(xlabel='time/s', ylabel='angle')
    axs[0].grid(True)

    axs[1].step(freq, np.abs(fft), where='mid')
    axs[1].scatter(peaks, np.abs(fft[maxima]), s=10, color='tab:orange',
                   label='Dominant frequencies: \n' + str(peaks_rounded) + ' at bin size ' + str(np.around(freq[1], 3)))
    axs[1].set(ylabel='DFT', xlabel='frequency /Hz')
    axs[1].grid(True)
    axs[1].legend()

    # fig.suptitle(file)
    fig.tight_layout()
    plt.show()
    if save:
        # fig.savefig('./Fouriertransforms/Plots/' + file + ' DCT angle' + '.pdf')
        fig.savefig('./Fouriertransforms/Plots/' + file + ' DCT angle' + '.png')


def main():
    x = [7, 8]
    for i in x:
        swing_up_task(file_start=1, file_end=3, folder_number=i, pole_length='L395', fig_size=(6, 7), save=True,
                      save_name='')


if __name__ == '__main__':
    # fourier_transforms('Repeated swing-up-2 L790', fig_size=(6, 4.5), save=False)
    swing_up_task(file_start=0, file_end=2, folder_number=16, pole_length='L395', fig_size=(6, 7), save=True,
                  save_name='')

    # main()
