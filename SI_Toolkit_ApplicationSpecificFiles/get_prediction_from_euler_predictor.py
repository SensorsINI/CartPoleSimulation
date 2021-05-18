import numpy as np

from CartPole.state_utilities import cartpole_state_varnames_to_indices, STATE_VARIABLES

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

    return output_array


