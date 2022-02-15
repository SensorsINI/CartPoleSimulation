"""
Script which takes data recording and returns the smoothed version.
It should be placed in the same folder where the recording you want to smooth is.
"""
import numpy as np
import pandas as pd
import csv

dt = 0.005
FILE_NAME = 'PID.csv'

PATH_TO_DATA = ''
file_path = PATH_TO_DATA + FILE_NAME

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

data: pd.DataFrame = pd.read_csv(file_path, comment='#')

data['position'] = smooth(data['position'], 30)
data['angle'] = smooth(data['angle'], 30)

data['positionD'] = smooth(data['positionD'], 30)
data['angleD'] = smooth(data['angleD'], 30)

with open(file_path[:-4]+'_s30.csv', "a") as outfile:
    writer = csv.writer(outfile)

    # FIXME: This is wrong. It just takes the fix dt, not even taken from the loaded csv.
    writer.writerow(['#'])
    writer.writerow(['# Time intervals dt:'])
    writer.writerow(['# Simulation: {} s'.format(str(dt))])
    writer.writerow(['# Controller update: {} s'.format(str(dt))])
    writer.writerow(['# Saving: {} s'.format(str(dt))])

data.to_csv(file_path[:-4]+'_s30.csv', index=False, header=True, mode='a')
