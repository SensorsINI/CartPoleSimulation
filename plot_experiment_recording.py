# Uncomment if you want to get interactive plots for MPPI in Pycharm on MacOS
# On other OS you have to chose a different interactive backend.
from matplotlib import use
# # use('TkAgg')
use('macOSX')
from cycler import cycler
import numpy as np

from CartPole import CartPole

import matplotlib.pyplot as plt

# csv_name = ['Experiment-GT-Smooth.csv', 'Experiment-8-Eq-Frozen-Smooth.csv', 'Experiment-9-Dense-Smooth.csv']
csv_name = ['Experiment-GT-Smooth.csv', 'Experiment-1-Eq-Frozen-Smooth.csv', 'Experiment-2-Dense-Smooth.csv']
final_index = -1
dict_datasets = {}

for experiment_name in csv_name:

    CartPoleInstance = CartPole()

    # Load experiment history
    history_pd, filepath = CartPoleInstance.load_history_csv(csv_name=experiment_name)

    # Augment the experiment history with simulation time step size
    dt = []
    row_iterator = history_pd.iterrows()
    _, last = next(row_iterator)  # take first item from row_iterator
    for i, row in row_iterator:
        dt.append(row['time'] - last['time'])
        last = row
    dt.append(dt[-1])
    history_pd['dt'] = np.array(dt)

    CartPoleInstance.dict_history = history_pd.to_dict(orient='list')

    # CartPoleInstance.summary_plots(title=experiment_name[:-4])

    dict_datasets[experiment_name[:-4]] = history_pd



# feature = 'cost_trajectory_from_u_predicted'
# feature = 'cost_trajectory_from_u_true_equations'
feature = 'stage_cost_realized_trajectory'
# feature = 'relative_cost'

fontsize_labels = 10
fontsize_ticks = 10

# try:
fig, axs = plt.subplots(1, 1, figsize=(16, 9), sharex=True)  # share x axis so zoom zooms all plots
fig.suptitle(feature, fontsize=20)

my_cycler = cycler(color=['g', 'b', 'orange', 'r'])
plt.rc('axes', prop_cycle=my_cycler)
axs.set_prop_cycle(my_cycler)

df = dict_datasets[csv_name[0][:-4]]

lns = []

# Plot parameter change and moments of training
axs.set_ylabel("Pole length [cm]", fontsize=fontsize_labels)
lns.append(axs.plot(np.array(df['time']), np.array(df['L']) * 2.0 * 100.0,
                            markersize=12, label='Pole length')[0])
axs.set_ylim(bottom=1.1 * min(np.array(df['L']) * 2.0 * 100.0),
                top=1.1 * max(np.array(df['L']) * 2.0 * 100.0))
axs.set_yscale('log')
axs.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
axs.tick_params(axis='both', which='minor', labelsize=fontsize_ticks)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


idx_change = df[df['L'].diff() != 0].index.tolist()
idx_time = idx_change + [len(df['time']) - 1]
cost_time = np.array(df['time'])[idx_time]

ax_cost = axs.twinx()
ax_cost._get_lines.prop_cycler = axs._get_lines.prop_cycler
# ax_cost.set_ylabel('Cost difference (%)', fontsize=fontsize_labels)
ax_cost.set_ylabel('Cost', fontsize=fontsize_labels)

ax_cost.set_yscale('log')
ax_cost.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax_cost.tick_params(axis='both', which='minor', labelsize=fontsize_ticks)

# Plot the difference in cost of best trajectory vs. predicted one

for key, dataset in dict_datasets.items():

    idx_change = dataset[df['L'].diff() != 0].index.tolist()
    idx_time = idx_change + [len(dataset['time']) - 1]
    cost_time = np.array(dataset['time'])[idx_time]

    gb = dataset.groupby(['L'], sort=False)
    # cost = gb['cost_trajectory_from_u_true_equations'].mean().values
    cost = gb[feature].mean().values

    cost = np.insert(cost, 0, cost[0])
    # except:
    # cost_time = np.array(self.dict_history['time'])
    # # cost = np.array(self.dict_history['cost_trajectory_from_u_predicted'])-np.array(self.dict_history['cost_trajectory_from_u_true_equations'])
    # # cost = np.array(self.dict_history['relative_cost'])
    # cost = np.array(self.dict_history['cost_trajectory_from_u_predicted'])
    # # cost = np.array(self.dict_history['cost_trajectory_from_u_true_equations'])
    # # cost = median_filter(cost, size=40)
    # cost = smooth(cost, 1000)



    # lns_cost = ax_cost.plot(cost_time, cost, drawstyle='steps', label='Cost difference (%)')
    lns.append(ax_cost.plot(cost_time, cost, drawstyle='steps', label=key)[0])
    # ax_cost.set_ylim(bottom=1.1 * min(cost), top=1.1 * max(cost))


labs = [l.get_label() for l in lns]
ax_cost.legend(lns, labs, fontsize=fontsize_labels)
# # except:
# #     pass
#
axs.set_xlabel('Time (s)', fontsize=fontsize_labels)
plt.show()
