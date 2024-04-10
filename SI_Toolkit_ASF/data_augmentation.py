from tqdm import trange
from time import sleep


def augment_data(data, labels):

    print('Augmenting data...')
    sleep(0.002)
    for i in trange(len(data)):
        data[i] = data[i]

    return data, labels
