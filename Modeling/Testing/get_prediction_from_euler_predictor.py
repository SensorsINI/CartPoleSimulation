from Modeling.load_and_normalize import \
    load_data, get_paths_to_datafiles, get_sampling_interval_from_datafile, load_cartpole_parameters

from tqdm import trange

import numpy as np

from CartPole._CartPole_mathematical_helpers import wrap_angle_rad
from CartPole.state_utilities import create_cartpole_state, \
    cartpole_state_varnames_to_indices, cartpole_state_varname_to_index, \
    cartpole_state_indices_to_varnames, STATE_VARIABLES
from types import SimpleNamespace

from CartPole.cartpole_model import Q2u, cartpole_ode

from Predictores.predictor_ideal import predictor_ideal

DEFAULT_SAMPLING_INTERVAL = 0.02  # s, Corresponds to our lab cartpole
def get_prediction_from_euler_predictor(a, dataset, dt_sampling, dt_sampling_by_dt_fine=1):

    states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]
    Q = dataset['Q'].to_numpy()
    Q_array = [Q[i:-a.test_max_horizon+i] for i in range(a.test_max_horizon)]
    Q_array = np.vstack(Q_array).transpose()

    predictor = predictor_ideal(horizon=a.test_max_horizon, dt=dt_sampling)

    predictor.setup(initial_state=states_0, prediction_denorm=True)
    output_array = predictor.predict(Q_array)  # Should be shape=(a.test_max_horizon, a.test_len, len(outputs))
    output_array = output_array[..., list(cartpole_state_varnames_to_indices(a.features))+[-1]]

    # time_axis is a time axis for ground truth
    return output_array


