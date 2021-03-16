from Modeling.load_and_normalize import \
    load_data, get_paths_to_datafiles, get_sampling_interval_from_datafile, load_cartpole_parameters

from tqdm import trange

import numpy as np

from CartPole._CartPole_mathematical_helpers import wrap_angle_rad
from types import SimpleNamespace

from CartPole.cartpole_model import Q2u, cartpole_ode


def get_prediction_from_euler(a):

    # region In either case testing is done on a data collected offline
    paths_to_datafiles_test = get_paths_to_datafiles(a.test_files)
    test_dfs = load_data(paths_to_datafiles_test)
    dataset = test_dfs[0].iloc[a.test_start_idx:a.test_start_idx+a.test_len+a.test_max_horizon, :]
    outputs = ['s.angle', 's.angle.cos', 's.angle.sin', 's.angleD', 's.angleDD',
               's.position', 's.positionD', 's.positionDD']
    inputs = outputs + ['Q']
    dataset[inputs]
    dataset.reset_index(drop=True, inplace=True)

    p = load_cartpole_parameters(paths_to_datafiles_test[0])
    dt_sampling = get_sampling_interval_from_datafile(paths_to_datafiles_test[0])
    s = SimpleNamespace()

    time_axis = dataset['time'].to_numpy()[:a.test_len]

    output_array = np.zeros(shape=(a.test_max_horizon, a.test_len, len(outputs)))

    print('Calculating predictions...')
    for timestep in trange(a.test_len):

        s.position = float(dataset.loc[dataset.index[[timestep]], 's.position'])
        s.positionD = float(dataset.loc[dataset.index[[timestep]], 's.positionD'])
        s.positionDD = float(dataset.loc[dataset.index[[timestep]], 's.positionDD'])
        s.angle = float(dataset.loc[dataset.index[[timestep]], 's.angle'])
        s.angle_cos = float(dataset.loc[dataset.index[[timestep]], 's.angle.cos'])
        s.angle_sin = float(dataset.loc[dataset.index[[timestep]], 's.angle.sin'])
        s.angleD = float(dataset.loc[dataset.index[[timestep]], 's.angleD'])
        s.angleDD = float(dataset.loc[dataset.index[[timestep]], 's.angleDD'])

        # Progress max_horison steps
        # save the data for every step in a third dimension of an array
        for i in range(0, a.test_max_horizon):
            Q = float(dataset.loc[dataset.index[[timestep+i]], 'Q'])
            u = Q2u(Q, p)
            # We assume control input is the first variable
            # All other variables are in closed loop

            # Integrate
            s.position += s.positionD * dt_sampling
            s.positionD += s.positionDD * dt_sampling

            s.angle += s.angleD * dt_sampling
            s.angleD += s.angleDD * dt_sampling

            s.angle = wrap_angle_rad(s.angle)

            s.angle_cos = np.cos(s.angle)
            s.angle_sin = np.sin(s.angle)

            s.angleDD, s.positionDD = cartpole_ode(p, s, u)

            state = np.array([s.angle, s.angle_cos, s.angle_sin, s.angleD, s.angleDD,
                          s.position, s.positionD, s.positionDD])
            # Append s to outputs matrix
            output_array[i, timestep, :] = state



    # Data tailored for plotting
    ground_truth = dataset.to_numpy()
    ground_truth = ground_truth[:a.test_len, :]

    # time_axis is a time axis for ground truth
    return inputs, outputs, 'Euler', ground_truth, output_array, time_axis


