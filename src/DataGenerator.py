from src.CartClass import *
from src.utilis import *
from src.utilis import *

from tqdm import tqdm
csv = 'data_rnn_very_big'
number_of_experiments = 4
length_of_experiment = 4e5+1

dt_main_simulation = dt_main_simulation_globals
track_relative_complexity = 0.5  # randomly placed target points/s
track_complexity = int(dt_main_simulation*length_of_experiment*track_relative_complexity)  # Total number of randomly placed points
mode = 2

MyCart = Cart()

for i in range(number_of_experiments):
    print(i)
    sleep(0.1)
    gen_start = timeit.default_timer()
    Generate_Experiment(MyCart,
                        mode=mode,
                        exp_len=length_of_experiment,
                        dt=dt_main_simulation,
                        track_relative_complexity=track_relative_complexity,
                        csv=csv,
                        save_csv_online=False)
    gen_end = timeit.default_timer()
    gen_dt = (gen_end-gen_start)*1000.0
    print('time to generate data: {} ms'.format(gen_dt))