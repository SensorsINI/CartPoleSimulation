import glob

import pandas as pd
import numpy as np

from Predictores.predictor_tests_plotting_helpers import pd_plotter_simple, pd_plotter_compare_2, get_predictions

from SI_Toolkit.TF.TF_Functions.predictor_autoregressive_tf import (
    predictor_autoregressive_tf,
)
from Predictores.predictor_ideal import predictor_ideal


DT = 0.1 # This is DT fed into predictor at initialization - meaning may differ between predictors
downsampling = 1
horizon = 50 // downsampling
start_at = 20+320
autoregres_at_after_start = 50

tested_predictor_1 = predictor_autoregressive_tf(horizon=horizon, dt=DT)
tested_predictor_2 = predictor_ideal(horizon=horizon, dt=DT)

# datafile = glob.glob('./data/validate/' + '*.csv')[0]
datafile = glob.glob('./Experiment_Recordings/Test/' + '*.csv')[0]
features = ['angle_cos', 'angle_sin', 'angle', 'angleD', 'position', 'positionD']
feature_to_plot = features[0]

df = pd.read_csv(datafile, comment='#', dtype=np.float32)
df = df.iloc[::downsampling].reset_index(drop=True)
df = df.iloc[start_at:].reset_index(drop=True)
df = df.iloc[0:autoregres_at_after_start + horizon + 1, df.columns.get_indexer(
    ['time', 'Q', 'angle_cos', 'angle_sin', 'angle', 'angleD', 'position', 'positionD'])]

pd_plotter_simple(df, 'time', feature_to_plot, idx_range=[0, autoregres_at_after_start + horizon],
                  vline=df.loc[df.index[autoregres_at_after_start], 'time'], marker='o',
                  title = 'Ground truth (warm-up + prediction region)')


predictions_1 = get_predictions(tested_predictor_1, df, autoregres_at_after_start, N_predictions=1, horizon=horizon)
predictions_2 = get_predictions(tested_predictor_2, df, autoregres_at_after_start, N_predictions=1, horizon=horizon)

fig3 = pd_plotter_compare_2(df.iloc[autoregres_at_after_start:autoregres_at_after_start + horizon + 1],
                            dfs=[predictions_1[0], predictions_2[0]], names=['rnn', 'equations'], y_name=feature_to_plot,
                            colors=['blue', 'darkgreen'],
                            idx_range=[0, horizon + 1], dt=DT, title= '', marker='o')