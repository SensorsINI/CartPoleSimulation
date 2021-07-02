import glob

import pandas as pd
import numpy as np

from Predictores.predictor_tests_plotting_helpers import pd_plotter_simple, pd_plotter_compare_2, get_predictions

from SI_Toolkit.load_and_normalize import normalize_df

from SI_Toolkit.TF.TF_Functions.predictor_autoregressive_tf import (
    predictor_autoregressive_tf,
)
from Predictores.predictor_ideal import predictor_ideal
from tqdm import tqdm

import matplotlib.pyplot as plt




DT = 0.1 # This is DT fed into predictor at initialization - meaning may differ between predictors
downsampling = 1
horizon = 2 // downsampling
start_at = 0
autoregres_at_after_start = 20
N_predictions = 10
prediction_denorm=False

tested_predictor_1 = predictor_autoregressive_tf(horizon=horizon, dt=DT)
tested_predictor_2 = predictor_ideal(horizon=horizon, dt=DT)

try:
    normalization_info = tested_predictor_1.normalization_info
except:
    print('No normalizoation info in p1')
    try:
        normalization_info = tested_predictor_2.normalization_info
    except:
        print('No normalizoation info in p1')
        raise AttributeError('Normalization info not found.')

# datafile = glob.glob('./data/validate/' + '*.csv')[0]
datafile = glob.glob('./Experiment_Recordings/Test/' + '*.csv')[0]
features = ['angle_cos', 'angle_sin', 'angle', 'angleD', 'position', 'positionD']
feature_to_plot = features[2]

df = pd.read_csv(datafile, comment='#', dtype=np.float32)
df = df.iloc[::downsampling].reset_index(drop=True)
df = df.iloc[start_at:].reset_index(drop=True)
N_predictions = len(df)-autoregres_at_after_start-horizon-180#-30000-9000
df = df.iloc[:, df.columns.get_indexer(
    ['time', 'Q'] + features)]



pd_plotter_simple(df, 'time', feature_to_plot, idx_range=[0, autoregres_at_after_start + N_predictions + horizon+1],
                  vline=df.loc[df.index[autoregres_at_after_start], 'time'], marker='o',
                  title = 'Ground truth (warm-up + prediction region)')

print('Predicting with predictor 1')
predictions_1 = get_predictions(tested_predictor_1, df, autoregres_at_after_start,
                                N_predictions=N_predictions, horizon=horizon, prediction_denorm=prediction_denorm)
print('Predicting with predictor 2')
predictions_2 = get_predictions(tested_predictor_2, df, autoregres_at_after_start,
                                N_predictions=N_predictions, horizon=horizon, prediction_denorm=prediction_denorm)

max_error_1 = pd.DataFrame(0, index=np.arange(N_predictions), columns=features)
max_error_2 = pd.DataFrame(0, index=np.arange(N_predictions), columns=features)

# For relative error (meaningless?)
# max_targets_1 = pd.DataFrame(0, index=np.arange(N_predictions), columns=features)
# max_targets_2 = pd.DataFrame(0, index=np.arange(N_predictions), columns=features)
print('Calculating errors')
for i in tqdm(range(N_predictions)):
    prediction_1 = predictions_1[i][features]
    prediction_2 = predictions_2[i][features]
    target = df[features]
    target = target.iloc[autoregres_at_after_start + i: autoregres_at_after_start + i + horizon + 1]
    target = target.reset_index(drop=True)
    if not prediction_denorm:
        target = normalize_df(target, normalization_info)


    error_1 = prediction_1-target
    error_2 = prediction_2-target

    # error_1_abs = error_1.abs()
    # error_2_abs = error_2.abs()
    #
    # idx_max_1 = error_1_abs.idxmax(axis=0).values
    # idx_max_2 = error_2_abs.idxmax(axis=0).values

    # For relative error
    # max_val_1 = []
    # max_val_2 = []
    # for idx in range(len(features)):
    #     max_val_1.append(target.at[target.index[idx_max_1[idx]], features[idx]])
    #     max_val_2.append(target.at[target.index[idx_max_2[idx]], features[idx]])
    # max_targets_1.iloc[i] = np.asarray(max_val_1)
    # max_targets_2.iloc[i] = np.asarray(max_val_2)


    maxCol = lambda x: max(x.min(), x.max(), key=abs)
    me1 = error_1.apply(maxCol, axis=0)
    me2 = error_2.apply(maxCol, axis=0)

    max_error_1.iloc[i] = me1.values
    max_error_2.iloc[i] = me2.values

    df_plot = df[features]
    if not prediction_denorm:
        df_plot = normalize_df(df_plot, normalization_info)

    # fig3 = pd_plotter_compare_2(df_plot.iloc[autoregres_at_after_start + i:autoregres_at_after_start + i + horizon + 1],
    #                             dfs=[predictions_1[i], predictions_2[i]], names=['rnn', 'equations'], y_name=feature_to_plot,
    #                             colors=['blue', 'darkgreen'],
    #                             idx_range=[0, horizon + 1], dt=DT, title= '', marker='o')

ground_truth_error = max_error_1.copy()
for col in ground_truth_error.columns:
    ground_truth_error[col].values[:] = 0


for i in range(len(features)):
    feature_to_plot = features[i]
    e1 = max_error_1.abs()[feature_to_plot].to_numpy().squeeze()
    e2 = max_error_2.abs()[feature_to_plot].to_numpy().squeeze()
    str_title = 'Absolute error (normed data) for {}'.format(feature_to_plot)

    fig1 = pd_plotter_compare_2(ground_truth_error,
                                dfs=[max_error_1.abs(), max_error_2.abs()], names=['rnn', 'equations'],
                                y_name=feature_to_plot,
                                colors=['blue', 'darkgreen'],
                                idx_range=[0, len(max_error_1)], title=str_title, marker='o')

    fig2 = plt.figure(figsize=(16,10))
    bins = np.linspace(min((min(e1), min(e2))), max((max(e1), max(e2))), 200)
    plt.hist(e1, bins, alpha=0.5, label='rnn')
    plt.hist(e2, bins, alpha=0.5, label='equations')
    plt.legend(loc='upper right')
    plt.title(label=str_title, fontdict={'fontsize': 20})
    plt.legend(prop={'size': 20})
    plt.xticks(fontsize=16)
    plt.show()


# print('')
# print('max_error_1')
# print(max_error_1)
# print('')
# print('max_error_2')
# print(max_error_2)

# print('')
# print('max_targets_1')
# print(max_targets_1)
# print('')
# print('max_targets_2')
# print(max_targets_2)

