import numpy as np
import matplotlib.pyplot as plt

from CartPole.cartpole_parameters import u_max


# Method plotting the dynamic evolution over time of the CartPole
# It should be called after an experiment and only if experiment data was saved
def summary_plots(dict_history, adaptive_mode=False, title=''):
    if adaptive_mode:
        number_of_subplots = 5
        fontsize_labels = 10
        fontsize_ticks = 10
    else:
        number_of_subplots = 4
        fontsize_labels = 14
        fontsize_ticks = 12

    fig, axs = plt.subplots(number_of_subplots, 1, figsize=(16, 9), sharex=True)  # share x axis so zoom zooms all plots
    fig.suptitle(title, fontsize=16)

    # Plot angle error
    axs[0].set_ylabel("Angle (deg)", fontsize=fontsize_labels)
    axs[0].plot(np.array(dict_history['time']), np.array(dict_history['angle']) * 180.0 / np.pi,
                'b', markersize=12, label='Ground Truth')
    axs[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    # Plot position
    axs[1].set_ylabel("position (m)", fontsize=fontsize_labels)
    axs[1].plot(dict_history['time'], dict_history['position'], 'g', markersize=12,
                label='Ground Truth')
    axs[1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    # Plot motor input command
    try:
        axs[2].set_ylabel("motor (N)", fontsize=fontsize_labels)
        axs[2].plot(dict_history['time'], dict_history['u'], 'r', markersize=12,
                    label='motor')
        axs[2].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
        axs[2].set_ylim(bottom=-1.05 * u_max, top=1.05 * u_max)
    except KeyError:
        axs[2].set_ylabel("motor normalized (-)", fontsize=fontsize_labels)
        axs[2].plot(dict_history['time'], dict_history['Q'], 'r', markersize=12,
                    label='motor')
        axs[2].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
        axs[2].set_ylim(bottom=-1.05, top=1.05)

    # Plot target position
    axs[3].set_ylabel("position target (m)", fontsize=fontsize_labels)
    axs[3].plot(dict_history['time'], dict_history['target_position'], 'k')
    axs[3].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    if adaptive_mode:
        ...
        axs[4].set_xlabel('Time (s)', fontsize=fontsize_labels)
    else:
        axs[3].set_xlabel('Time (s)', fontsize=fontsize_labels)

    fig.align_ylabels()

    plt.show()

    return fig, axs

# endregion