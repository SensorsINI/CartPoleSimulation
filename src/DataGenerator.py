from src.CartClass import *
from src.utilis import *

# User defined simulation settings - check the effect first in GUI before you launch big data generation
csv = 'data_free_fall_1'  # Name with which data is saved, consecutive experiments will be save with increasing index attached
number_of_experiments = 1  # How many experiments will be generated
length_of_experiment = 8000.0  # Length of each experiment in s
dt_main_simulation = dt_main_simulation_globals  # simulation timestep
track_relative_complexity = 0.01  # randomly placed target points/s
controller = 'do-mpc'  # Controller which should be used in generated experiment
# Possible options for controller:
# 'manual-stabilization', 'do-mpc', 'do-mpc-discrete', 'lqr', 'rnn_as_mpc_tf'
interpolation_type = 'previous'  # Sets how to interpolate between turning points of random trace
# Possible choices: '0-derivative-smooth', 'linear', 'previous'
turning_points_period = 'regular'  # How turning points should be distributed
# Possible choices: 'regular', 'random'
# Where the target position of the random experiment starts and end
start_random_target_position_at = 0.0
end_random_target_position_at = 0.0
# The list of turning points is set to None, no matter what is in globals.py

save_data_online = True  # It was intended to save memory usage, but it doesn't seems to help. Leave it false.

print(controller)

for i in range(number_of_experiments):

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # You may also specify some of the variables from above here, to make them change at each iteration.#
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    print(i)
    sleep(0.1)
    MyCart = Cart()
    gen_start = timeit.default_timer()
    Generate_Experiment(MyCart,
                        controller=controller,
                        exp_len=length_of_experiment,
                        dt=dt_main_simulation,
                        track_relative_complexity=track_relative_complexity,
                        interpolation_type=interpolation_type,
                        turning_points_period=turning_points_period,
                        start_random_target_position_at=start_random_target_position_at,
                        end_random_target_position_at=end_random_target_position_at,
                        csv=csv,
                        save_data_online=save_data_online)

    gen_end = timeit.default_timer()
    gen_dt = (gen_end-gen_start)*1000.0
    print('time to generate data: {} ms'.format(gen_dt))