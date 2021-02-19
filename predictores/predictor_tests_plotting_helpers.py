# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:28:34 2020

@author: Marcin
"""

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt


def pd_plotter_simple(df, x_name=None, y_name=None, idx_range=None,
                      color='blue', dt=None, marker=None, vline=None, title=''):

    if idx_range is None:
        idx_range = [0, -1]

    if y_name is None:
        raise ValueError ('You must provide y_name')
    y = df[y_name].iloc[idx_range[0]:idx_range[1]]

    if x_name is None:
        x = np.arange(len(y))
        if dt is not None:
            x = x*dt
    else:
        x = df[x_name].iloc[idx_range[0]:idx_range[1]]

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y, color=color, linewidth=3, marker=marker)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_xlabel(x_name, fontsize=18)
    if vline is not None:
        plt.axvline(x=vline, color='orange')
    plt.title(label=title, fontdict={'fontsize': 20})
    plt.xticks(fontsize=16)

    plt.show()


def pd_plotter_compare_1(d_ground, df, x_name=None, y_name=None, idx_range=None,
                      color='blue', dt=None, marker=None, vline=None, title=''):

    if idx_range is None:
        idx_range = [0, -1]

    if y_name is None:
        raise ValueError ('You must provide y_name')
    y_ground = d_ground[y_name].to_numpy(copy=True)[idx_range[0]:idx_range[1]]
    y_f = df[y_name].to_numpy(copy=True)[idx_range[0]:idx_range[1]]


    if x_name is None:
        x = np.arange(len(y_ground))
        if dt is not None:
            x = x*dt
            x_name = 'time (s)'
    else:
        x = d_ground[x_name].iloc[idx_range[0]:idx_range[1]]

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y_ground, color='k', linewidth=2, linestyle='dotted')
    ax.plot(x, y_f, color=color, linewidth=3, marker=marker)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_xlabel(x_name, fontsize=18)
    if vline is not None:
        plt.axvline(x=vline, color='orange')
    plt.title(label=title, fontdict={'fontsize': 20})
    plt.xticks(fontsize=16)

    plt.show()


def pd_plotter_compare_2(d_ground, dfs, names, x_name=None, y_name=None, idx_range=None,
                      colors=None, dt=None, marker=None, vline=None, title=''):

    if idx_range is None:
        idx_range = [0, -1]

    if colors is None:
        colors=['blue', 'green']

    if y_name is None:
        raise ValueError ('You must provide y_name')
    y_ground = d_ground[y_name].to_numpy(copy=True)[idx_range[0]:idx_range[1]]
    y_f = dfs[0][y_name].to_numpy(copy=True)[idx_range[0]:idx_range[1]]
    y_h = dfs[1][y_name].to_numpy(copy=True)[idx_range[0]:idx_range[1]]


    if x_name is None:
        x = np.arange(len(y_ground))
        if dt is not None:
            x = x*dt
            x_name = 'time (s)'
    else:
        x = d_ground[x_name].iloc[idx_range[0]:idx_range[1]]

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y_ground, color='k', linewidth=2, linestyle='dotted', label='ground truth')
    ax.plot(x, y_f, color=colors[0], linewidth=3, marker=marker, label=names[0])
    ax.plot(x, y_h, color=colors[1], linewidth=3, marker=marker, label=names[1])
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_xlabel(x_name, fontsize=18)
    if vline is not None:
        plt.axvline(x=vline, color='orange')
    plt.title(label=title, fontdict={'fontsize': 20})
    plt.legend(prop={'size': 20})
    plt.xticks(fontsize=16)

    plt.show()


def get_predictions(predictor, df, autoregres_at_after_start, N_predictions, horizon, prediction_denorm=True):
    for row_number in range(autoregres_at_after_start):
        initial_state = df.iloc[[row_number], :]
        Q = np.atleast_1d(df.loc[df.index[row_number], 'Q'])
        predictor.setup(initial_state)
        predictor.update_internal_state(Q)

    predictions = []
    for i in tqdm(range(N_predictions)):
        # Prepare initial state for predictions
        initial_state = df.iloc[[autoregres_at_after_start+i], :]
        predictor.setup(initial_state, prediction_denorm=prediction_denorm)

        Q = np.atleast_1d(df.loc[df.index[autoregres_at_after_start+i: autoregres_at_after_start+i + horizon], 'Q'] \
                          .to_numpy(copy=True, dtype=np.float32).squeeze())

        prediction = predictor.predict(Q)
        predictions.append(prediction)
        predictor.update_internal_state(Q[0])


    return predictions