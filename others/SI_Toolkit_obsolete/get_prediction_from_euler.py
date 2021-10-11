from tqdm import trange

import numpy as np

from CartPole._CartPole_mathematical_helpers import wrap_angle_rad
from CartPole.state_utilities import create_cartpole_state, \
    cartpole_state_varnames_to_indices, \
    ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX

from CartPole.cartpole_model import Q2u, cartpole_ode


DEFAULT_SAMPLING_INTERVAL = 0.02  # s, Corresponds to our lab cartpole
def get_prediction_for_testing_gui_from_euler(a, dataset, dt_sampling, dt_sampling_by_dt_fine=1):

    # region In either case testing is done on a data collected offline
    output_array = np.zeros(shape=(a.test_max_horizon+1, a.test_len, len(a.features)+1))

    Q = dataset['Q'].to_numpy()
    Q_array = [Q[i:-a.test_max_horizon + i] for i in range(a.test_max_horizon)]
    Q_array = np.vstack(Q_array)
    output_array[:-1, :, -1] = Q_array
    print('Calculating predictions...')

    FEATURES_INDICES = cartpole_state_varnames_to_indices(a.features)

    for timestep in trange(a.test_len):

        state_dict = dataset.loc[dataset.index[[timestep]], :].to_dict('records')[0]
        s = create_cartpole_state(state_dict)
        output_array[0, timestep, :-1] = s[FEATURES_INDICES]
        # Progress max_horison steps
        # save the data for every step in a third dimension of an array
        for i in range(0, a.test_max_horizon):
            Q = Q_array[i, timestep]
            u = Q2u(Q)

            for _ in range(dt_sampling_by_dt_fine):

                angleDD, positionDD = cartpole_ode(s, u)

                t_step = (dt_sampling/float(dt_sampling_by_dt_fine))
                # Calculate next state
                s[POSITION_IDX] += s[POSITIOND_IDX] * t_step
                s[POSITIOND_IDX] += positionDD * t_step
                s[ANGLE_IDX] += s[ANGLED_IDX] * t_step
                s[ANGLED_IDX] += angleDD * t_step

                s[ANGLE_IDX] = \
                    wrap_angle_rad(s[ANGLE_IDX])

            # Append s to outputs matrix
            s[ANGLE_COS_IDX] = np.cos(s[ANGLE_IDX])
            s[ANGLE_SIN_IDX] = np.sin(s[ANGLE_IDX])

            output_array[i+1, timestep, :-1] = s[FEATURES_INDICES]

    output_array = np.swapaxes(output_array, 0, 1)
    # time_axis is a time axis for ground truth
    return output_array


