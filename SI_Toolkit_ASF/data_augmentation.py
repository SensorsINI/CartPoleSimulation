from tqdm import trange
from time import sleep

AUGMENT_DATA = False


def augment_data(data, labels):

    if AUGMENT_DATA:
        print('Augmenting data...')
        sleep(0.002)
        for i in trange(len(data)):
            data[i] = data[i]
    return data, labels
