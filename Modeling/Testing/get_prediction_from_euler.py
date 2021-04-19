from Modeling.load_and_normalize import \
    load_data, get_paths_to_datafiles, get_sampling_interval_from_datafile, load_cartpole_parameters

from tqdm import trange

import numpy as np

from CartPole._CartPole_mathematical_helpers import wrap_angle_rad
from CartPole.state_utilities import create_cartpole_state, \
    cartpole_state_varnames_to_indices, cartpole_state_varname_to_index, \
    cartpole_state_indices_to_varnames
from types import SimpleNamespace

from CartPole.cartpole_model import Q2u, cartpole_ode

DEFAULT_SAMPLING_INTERVAL = 0.005  # s, Corresponds to our lab cartpole
def get_prediction_from_euler(a, dt_sampling_by_dt_fine=1):

    # region In either case testing is done on a data collected offline
    paths_to_datafiles_test = get_paths_to_datafiles(a.test_files)
    test_dfs = load_data(paths_to_datafiles_test)
    if a.test_len=='max':
        a.test_len=len(test_dfs[0])-a.test_max_horizon-1 # FIXME: I am not sure if this -1 is necessary
    dataset = test_dfs[0].iloc[a.test_start_idx:a.test_start_idx+a.test_len+a.test_max_horizon, :]

    s = create_cartpole_state()
    outputs = cartpole_state_indices_to_varnames(range(len(s)))
    inputs = outputs + ['Q']
    dataset.reset_index(drop=True, inplace=True)

    dt_sampling = get_sampling_interval_from_datafile(paths_to_datafiles_test[0])
    if dt_sampling is None:
        dt_sampling=DEFAULT_SAMPLING_INTERVAL
        print('No information about sampling interval found, I use DEFAULT_SAMPLING_INTERVAL with value {}'.format(DEFAULT_SAMPLING_INTERVAL))

    time_axis = dataset['time'].to_numpy()[:a.test_len]
    output_array = np.zeros(shape=(a.test_max_horizon, a.test_len, len(outputs)))

    print('Calculating predictions...')

    for timestep in trange(a.test_len):

        state_dict = dataset.loc[dataset.index[[timestep]], :].to_dict('records')[0]
        s = create_cartpole_state(state_dict)
        # Calculate second derivatives for first step - they may or may not be present in s
        Q = float(dataset.loc[dataset.index[[timestep]], 'Q'])
        u = Q2u(Q)
        s[cartpole_state_varnames_to_indices(['angleDD', 'positionDD'])] = cartpole_ode(s, u)

        # Progress max_horison steps
        # save the data for every step in a third dimension of an array
        for i in range(0, a.test_max_horizon):
            Q = float(dataset.loc[dataset.index[[timestep+i]], 'Q'])
            u = Q2u(Q)
            # We assume control input is the first variable
            # All other variables are in closed loop
            for _ in range(dt_sampling_by_dt_fine):
                s[cartpole_state_varnames_to_indices(['position', 'positionD', 'angle', 'angleD'])] += \
                    s[cartpole_state_varnames_to_indices(['positionD','positionDD', 'angleD', 'angleDD'])] * (dt_sampling/dt_sampling_by_dt_fine)

                s[cartpole_state_varname_to_index('angle')] = \
                    wrap_angle_rad(s[cartpole_state_varname_to_index('angle')])

                s[cartpole_state_varnames_to_indices(['angle_cos', 'angle_sin'])] = \
                    [np.cos(s[cartpole_state_varname_to_index('angle')]), np.sin(s[cartpole_state_varname_to_index('angle')])]

                s[cartpole_state_varnames_to_indices(['angleDD', 'positionDD'])] = cartpole_ode(s, u)

            # Append s to outputs matrix
            output_array[i, timestep, :] = s



    # Data tailored for plotting
    ground_truth = dataset.to_numpy()
    ground_truth = ground_truth[:a.test_len, :]

    # time_axis is a time axis for ground truth
    return inputs, outputs, 'Euler', ground_truth, output_array, time_axis


