import timeit
import glob

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Predictores.predictor_tests_plotting_helpers import pd_plotter_simple

from SI_Toolkit.TF.TF_Functions.predictor_autoregressive_tf import (
    predictor_autoregressive_tf,
)

tested_predictor = predictor_autoregressive_tf

DT = 0.1 # This is DT fed into predictor at initialization - meaning may differ between predictors
downsampling = 1
horizon = 10 // downsampling
start_at = 20
autoregres_at_after_start = 20

datafile = glob.glob('./Experiment_Recordings/Test/' + '*.csv')[0]
features = ['angle_cos', 'angle_sin', 'angle', 'angleD', 'position', 'positionD']
feature_to_plot = features[4]

df = pd.read_csv(datafile, comment='#', dtype=np.float32)
df = df.iloc[::downsampling].reset_index(drop=True)
df = df.iloc[start_at:].reset_index(drop=True)
df = df.iloc[0:autoregres_at_after_start + horizon + 1, df.columns.get_indexer(
    ['time', 'Q', 'angle', 'angle_cos', 'angle_sin', 'angleD', 'position', 'positionD'])]

pd_plotter_simple(df, 'time', feature_to_plot, idx_range=[0, autoregres_at_after_start + horizon],
                  vline=df.loc[df.index[autoregres_at_after_start], 'time'], marker='o',
                  title='Ground truth (warm-up + prediction region)')

predictor = tested_predictor(horizon=horizon, dt=DT)
# predictor = tested_predictor(horizon=horizon*5, dt=0.02) # To get ground truth

# In fact the block of code for this controller does nothing
# It has to be ensured however that it does nothing, to ensure this predictor is compatible
# with the main program
t0 = timeit.default_timer()
for row_number in range(autoregres_at_after_start):
    initial_state = df.iloc[[row_number], :]
    Q = np.atleast_1d(df.loc[df.index[row_number], 'Q'])
    predictor.setup(initial_state)
    predictor.update_internal_state(Q)
t1 = timeit.default_timer()

# Prepare initial state for predictions
initial_state = df.iloc[[autoregres_at_after_start], :]
predictor.setup(initial_state, prediction_denorm=True)
# Prepare control inputs for future predictions
Q = np.atleast_1d(df.loc[df.index[autoregres_at_after_start: autoregres_at_after_start + horizon], 'Q'] \
                  .to_numpy(copy=True, dtype=np.float32).squeeze())

# Make predictions.
t2 = timeit.default_timer()
# Option for multiple evaluations - to measure performance
number_of_repetitions = 1
# Q = np.repeat(Q, 5) # To get ground truth
for i in range(number_of_repetitions):
    prediction = predictor.predict(Q)
t3 = timeit.default_timer()

fig1 = pd_plotter_simple(df, y_name=feature_to_plot,
                         idx_range=[autoregres_at_after_start, autoregres_at_after_start + horizon + 1], dt=DT,
                         title='Ground truth (zoom on prediction region)')

fig2 = pd_plotter_simple(prediction, y_name=feature_to_plot, idx_range=[0, horizon + 1], color='red',
                         dt=DT, title='Prediction')

# fig2 = pd_plotter_simple(prediction, y_name=feature_to_plot, idx_range=[0, horizon*5 + 1], color='red',
#                          dt=0.02) # To get ground truth

target = df[feature_to_plot].to_numpy().squeeze()[autoregres_at_after_start: autoregres_at_after_start + horizon + 1]
prediction_single = prediction[feature_to_plot].to_numpy().squeeze()
fig3 = plt.figure()
plt.plot(target - prediction_single)
plt.title('Error: (ground truth - prediction)')
plt.show()

# Print performance summary
update_rnn_t = (t1 - t0) / autoregres_at_after_start
print('Update predictor {} us/eval'.format(update_rnn_t * 1.0e6))
predictor_t = (t3 - t2) / horizon / number_of_repetitions
print('Predict {} us/eval'.format(predictor_t * 1.0e6))