"""
If AUGMENT_DATA in config_training is True the augment_data is applied to the dataset before training and after every epoch.
The data and labels (input and output of the neural network) provided to this function are always the original data and labels
- the subsequent calls of this function do not accumulate.
Modify this function according to your needs.
"""

from tqdm import trange
from time import sleep


def augment_data(data, labels):

    print('Augmenting data...')
    sleep(0.002)
    for i in trange(len(data)):
        data[i] = data[i]

    return data, labels
