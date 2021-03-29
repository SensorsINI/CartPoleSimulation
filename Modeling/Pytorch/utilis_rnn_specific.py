import matplotlib.pyplot as plt
import numpy as np



def plot_results_specific(targets_pd, rnn_outputs, time_axis, comment, closed_loop_enabled, close_loop_idx):

    if time_axis == []:
        time_axis = np.arange(0, targets_pd.shape[0])
        time_axis_string = 'Sample number'
    else:
        time_axis = time_axis[1:]
        time_axis = time_axis-min(time_axis) # Start at 0, comment out if you want to relate to a true experiment
        time_axis_string = 'Time [s]'


    if ('angle' in targets_pd) and ('angle' in rnn_outputs) and ('position' in targets_pd) and (
            'position' in rnn_outputs):
        angle_target = np.rad2deg(targets_pd['angle'].to_numpy())
        position_target = targets_pd['position'].to_numpy()
        angle_output = np.rad2deg(rnn_outputs['angle'].to_numpy())
        position_output = rnn_outputs['position'].to_numpy()

        # Create a figure instance
        fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        plt.subplots_adjust(hspace=0.4)
        start_idx = 0
        axs[0].set_title(comment, fontsize=20)

        axs[0].set_ylabel("Position (m)", fontsize=18)
        axs[0].plot(time_axis, position_target, 'k:', markersize=12, label='Ground Truth')
        axs[0].plot(time_axis, position_output, 'b', markersize=12, label='Predicted position')

        axs[0].plot(time_axis[start_idx], position_target[start_idx], 'g.', markersize=16, label='Start')
        axs[0].plot(time_axis[start_idx], position_output[start_idx], 'g.', markersize=16)
        axs[0].plot(time_axis[-1], position_target[-1], 'r.', markersize=16, label='End')
        axs[0].plot(time_axis[-1], position_output[-1], 'r.', markersize=16)
        if closed_loop_enabled:
            axs[0].plot(time_axis[close_loop_idx], position_target[close_loop_idx], '.', color='darkorange', markersize=16, label='connect output->input')
            axs[0].plot(time_axis[close_loop_idx], position_output[close_loop_idx], '.', color='darkorange', markersize=16)

        axs[0].legend()


        axs[1].set_ylabel("Angle (deg)", fontsize=18)
        axs[1].plot(time_axis, angle_target, 'k:', markersize=12, label='Ground Truth')
        axs[1].plot(time_axis, angle_output, 'b', markersize=12,
                    label='Predicted angle')

        axs[1].plot(time_axis[start_idx], angle_target[start_idx], 'g.', markersize=16,
                    label='Start')
        axs[1].plot(time_axis[start_idx], angle_output[start_idx], 'g.', markersize=16)
        axs[1].plot(time_axis[-1], angle_target[-1], 'r.', markersize=16, label='End')
        axs[1].plot(time_axis[-1], angle_output[-1], 'r.', markersize=16)
        if closed_loop_enabled:
            axs[1].plot(time_axis[close_loop_idx], angle_target[close_loop_idx], '.',
                        color='darkorange', markersize=16, label='connect output->input')
            axs[1].plot(time_axis[close_loop_idx], angle_output[close_loop_idx], '.',
                        color='darkorange', markersize=16)

        axs[1].tick_params(axis='both', which='major', labelsize=16)

        axs[1].set_xlabel(time_axis_string, fontsize=18)
        axs[1].legend()


    if 'Q' in targets_pd:
        motor_power_target = targets_pd['Q'].to_numpy()
        motor_power_output = targets_pd['Q'].to_numpy()

        # Create a figure instance
        fig, axs = plt.subplots(1, 1, figsize=(18, 10))
        plt.subplots_adjust(hspace=0.4)
        start_idx = 0
        axs.set_title(comment, fontsize=20)

        axs.set_ylabel("Motor power (-)", fontsize=18)
        axs.plot(time_axis, motor_power_target, 'k:', markersize=12, label='Ground Truth')
        axs.plot(time_axis, motor_power_output, 'b', markersize=12, label='Predicted position')

        axs.plot(time_axis[start_idx], motor_power_target[start_idx], 'g.', markersize=16, label='Start')
        axs.plot(time_axis[start_idx], motor_power_output[start_idx], 'g.', markersize=16)
        axs.plot(time_axis[-1], motor_power_target[-1], 'r.', markersize=16, label='End')
        axs.plot(time_axis[-1], motor_power_output[-1], 'r.', markersize=16)

        axs.legend()


    return fig, axs