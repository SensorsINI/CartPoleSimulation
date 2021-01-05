import matplotlib.pyplot as plt
import numpy as np



def plot_results_specific(targets_pd, rnn_outputs, features_pd_denorm, time_axis, comment, closed_loop_enabled, close_loop_idx):

    if time_axis == []:
        time_axis = np.arange(0, targets_pd.shape[0])
        time_axis_string = 'Sample number'
    else:
        time_axis = time_axis[1:]
        time_axis = time_axis-min(time_axis) # Start at 0, comment out if you want to relate to a true experiment
        time_axis_string = 'Time [s]'

    figs = []

    if ('s.position' in targets_pd) and ('s.position' in rnn_outputs):
        if ('s.angle' in targets_pd) and ('s.angle' in rnn_outputs):
            angle_target = np.rad2deg(targets_pd['s.angle'].to_numpy())
            angle_output = np.rad2deg(rnn_outputs['s.angle'].to_numpy())
        elif ('s.angle.sin' in rnn_outputs) and ('s.angle.cos' in rnn_outputs):
            angle_target = np.rad2deg(np.arctan2(targets_pd['s.angle.sin'].to_numpy(), targets_pd['s.angle.cos'].to_numpy()))
            angle_output = np.rad2deg(np.arctan2(rnn_outputs['s.angle.sin'].to_numpy(), rnn_outputs['s.angle.cos'].to_numpy()))


        position_target = targets_pd['s.position'].to_numpy()
        position_output = rnn_outputs['s.position'].to_numpy()

        Q = features_pd_denorm['Q'].to_numpy()

        # Create a figure instance
        fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
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
        axs[1].legend()


        axs[2].set_ylabel("Motor (-)", fontsize=18)
        axs[2].plot(time_axis, Q, 'r', markersize=12, label='Ground Truth')

        axs[2].plot(time_axis[start_idx], Q[start_idx], 'g.', markersize=16,
                    label='Start')
        axs[2].plot(time_axis[-1], Q[-1], 'r.', markersize=16, label='End')
        if closed_loop_enabled:
            axs[2].plot(time_axis[close_loop_idx], Q[close_loop_idx], '.',
                        color='darkorange', markersize=16, label='connect output->input')

        axs[2].tick_params(axis='both', which='major', labelsize=16)

        axs[2].set_xlabel(time_axis_string, fontsize=18)
        axs[2].legend()

        figs.append(fig)

    if ('s.positionD' in targets_pd) and ('s.positionD' in rnn_outputs) and \
            ('s.angleD' in targets_pd) and ('s.angleD' in rnn_outputs):

        positionD_target = targets_pd['s.positionD'].to_numpy()
        positionD_output = rnn_outputs['s.positionD'].to_numpy()

        angleD_target = targets_pd['s.angleD'].to_numpy()
        angleD_output = rnn_outputs['s.angleD'].to_numpy()

        Q = features_pd_denorm['Q'].to_numpy()

        # Create a figure instance
        fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
        plt.subplots_adjust(hspace=0.4)
        start_idx = 0
        axs[0].set_title(comment, fontsize=20)

        axs[0].set_ylabel("PositionD (m/s)", fontsize=18)
        axs[0].plot(time_axis, positionD_target, 'k:', markersize=12, label='Ground Truth')
        axs[0].plot(time_axis, positionD_output, 'b', markersize=12, label='Predicted position')

        axs[0].plot(time_axis[start_idx], positionD_target[start_idx], 'g.', markersize=16, label='Start')
        axs[0].plot(time_axis[start_idx], positionD_output[start_idx], 'g.', markersize=16)
        axs[0].plot(time_axis[-1], positionD_target[-1], 'r.', markersize=16, label='End')
        axs[0].plot(time_axis[-1], positionD_output[-1], 'r.', markersize=16)
        if closed_loop_enabled:
            axs[0].plot(time_axis[close_loop_idx], positionD_target[close_loop_idx], '.', color='darkorange', markersize=16, label='connect output->input')
            axs[0].plot(time_axis[close_loop_idx], positionD_output[close_loop_idx], '.', color='darkorange', markersize=16)

        axs[0].legend()


        axs[1].set_ylabel("AngleD (deg/s)", fontsize=18)
        axs[1].plot(time_axis, angleD_target, 'k:', markersize=12, label='Ground Truth')
        axs[1].plot(time_axis, angleD_output, 'b', markersize=12,
                    label='Predicted angle')

        axs[1].plot(time_axis[start_idx], angleD_target[start_idx], 'g.', markersize=16,
                    label='Start')
        axs[1].plot(time_axis[start_idx], angleD_output[start_idx], 'g.', markersize=16)
        axs[1].plot(time_axis[-1], angleD_target[-1], 'r.', markersize=16, label='End')
        axs[1].plot(time_axis[-1], angleD_output[-1], 'r.', markersize=16)
        if closed_loop_enabled:
            axs[1].plot(time_axis[close_loop_idx], angleD_target[close_loop_idx], '.',
                        color='darkorange', markersize=16, label='connect output->input')
            axs[1].plot(time_axis[close_loop_idx], angleD_output[close_loop_idx], '.',
                        color='darkorange', markersize=16)
        axs[1].legend()


        axs[2].set_ylabel("Motor (-)", fontsize=18)
        axs[2].plot(time_axis, Q, 'r', markersize=12, label='Ground Truth')

        axs[2].plot(time_axis[start_idx], Q[start_idx], 'g.', markersize=16,
                    label='Start')
        axs[2].plot(time_axis[-1], Q[-1], 'r.', markersize=16, label='End')
        if closed_loop_enabled:
            axs[2].plot(time_axis[close_loop_idx], Q[close_loop_idx], '.',
                        color='darkorange', markersize=16, label='connect output->input')

        axs[2].tick_params(axis='both', which='major', labelsize=16)

        axs[2].set_xlabel(time_axis_string, fontsize=18)
        axs[2].legend()

        figs.append(fig)


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

        figs.append(fig)


    return figs