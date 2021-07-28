# Uncomment if you want to get interactive plots for MPPI in Pycharm on MacOS
# On other OS you have to chose a different interactive backend.
from matplotlib import use
# # use('TkAgg')
use('macOSX')

import numpy as np

from CartPole import CartPole

csv_name = 'Experiment.csv'
final_index = -1

CartPoleInstance = CartPole()

# Load experiment history
history_pd, filepath = CartPoleInstance.load_history_csv(csv_name=csv_name)

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

CartPoleInstance.summary_plots()