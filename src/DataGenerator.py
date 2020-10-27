from src.CartClass import *
from src.utilis import *
from src.utilis import *

from tqdm import tqdm
csv = 'data_rnn'
number_of_experiments = 10
length_of_experiment = 1e3

dt_main_simulation = dt_main_simulation_globals
track_relative_complexity = 0.5  # randomly placed points/s
track_complexity = int(dt_main_simulation*length_of_experiment*track_relative_complexity)  # Total number of randomly placed points
mode = 2

MyCart = Cart()

for i in range(number_of_experiments):
    print(i)
    sleep(0.1)
    Generate_Experiment(MyCart,
                        mode=mode,
                        exp_len=length_of_experiment,
                        dt=dt_main_simulation,
                        track_complexity=track_complexity,
                        csv=csv)