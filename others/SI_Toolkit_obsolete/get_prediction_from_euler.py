from tqdm import trange

import numpy as np

from CartPole._CartPole_mathematical_helpers import wrap_angle_rad
from CartPole.state_utilities import create_cartpole_state, \
    cartpole_state_varnames_to_indices, cartpole_state_varname_to_index

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

    for timestep in trange(a.test_len):

        state_dict = dataset.loc[dataset.index[[timestep]], :].to_dict('records')[0]
        s = create_cartpole_state(state_dict)
        output_array[0, timestep, :-1] = s[cartpole_state_varnames_to_indices(a.features)]
        # Progress max_horison steps
        # save the data for every step in a third dimension of an array
        for i in range(0, a.test_max_horizon):
            Q = Q_array[i, timestep]
            u = Q2u(Q)

            for _ in range(dt_sampling_by_dt_fine):

                angleDD, positionDD = cartpole_ode(s, u)

                t_step = (dt_sampling/float(dt_sampling_by_dt_fine))
                # Calculate next state
                s[cartpole_state_varname_to_index("position")] += s[cartpole_state_varname_to_index("positionD")] * t_step
                s[cartpole_state_varname_to_index("positionD")] += positionDD * t_step
                s[cartpole_state_varname_to_index("angle")] += s[cartpole_state_varname_to_index("angleD")] * t_step
                s[cartpole_state_varname_to_index("angleD")] += angleDD * t_step

                s[cartpole_state_varname_to_index('angle')] = \
                    wrap_angle_rad(s[cartpole_state_varname_to_index('angle')])

            # Append s to outputs matrix
            s[cartpole_state_varnames_to_indices(['angle_cos', 'angle_sin'])] = \
                [np.cos(s[cartpole_state_varname_to_index('angle')]),
                 np.sin(s[cartpole_state_varname_to_index('angle')])]
            output_array[i+1, timestep, :-1] = s[cartpole_state_varnames_to_indices(a.features)]

    output_array = np.swapaxes(output_array, 0, 1)
    # time_axis is a time axis for ground truth
    return output_array


