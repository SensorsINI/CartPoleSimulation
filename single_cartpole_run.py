from numpy.core.records import record
from CartPole import CartPole
from CartPole.cartpole_model import create_cartpole_state, TrackHalfLength
from others.p_globals import P_GLOBALS
from CartPole.state_utilities import cartpole_state_varname_to_index, cartpole_state_varnames_to_indices

import os
from time import sleep
import timeit
import cProfile
from pstats import Stats, SortKey

import numpy as np
# Uncomment if you want to get interactive plots for MPPI in Pycharm on MacOS
# On other OS you have to chose a different interactive backend.
# from matplotlib import use
# # use('TkAgg')
# use('macOSX')

# User defined simulation settings
# Automatically create new path to save everything in

import yaml, os
config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml')), Loader=yaml.FullLoader)

record_path = "./ExperimentRecordings/"

############ CHANGE THESE PARAMETERS AS YOU LIKE ############
number_of_experiments = 1  # How many experiments will be generated

###### Train/Val/Test split
frac_train = 0.8
frac_val = 0.19

save_mode = 'offline'  # It was intended to save memory usage, but it doesn't seems to help. Leave it false.

###### Timescales
dt_simulation_DataGen = 0.002  # simulation timestep
dt_controller_update_DataGen = 0.02  # control rate
dt_save_DataGen = 0.02  # save datapoints in csv in this interval

###### CartPole settings
### Length of each experiment in s:
length_of_experiment_DataGen = 12  

### Controller which should be used in generated experiment:
controller_DataGen = 'mppi'
# Possible options: 'manual-stabilization', 'do-mpc', 'do-mpc-discrete', 'lqr', 'mppi'

### Randomly placed target points/s
track_relative_complexity_DataGen = 1

### How to interpolate between turning points of random trace
interpolation_type_DataGen = 'previous'
# Possible options: '0-derivative-smooth', 'linear', 'previous'

### How turning points should be distributed
turning_points_period_DataGen = 'regular'
# Possible options: 'regular', 'random'

### Set the max for smoothly interpolated random target position to avoid bumping into track ends.
used_track_fraction = 0.9

### Where the target positions of the random experiment start and end
start_random_target_position_at_DataGen = used_track_fraction * TrackHalfLength * np.random.uniform(-1.0, 1.0)
end_random_target_position_at_DataGen = used_track_fraction * TrackHalfLength * np.random.uniform(-1.0, 1.0)

### List of target positions, can be None to simulate with random targets
turning_points_DataGen = None
# Example: turning_points_DataGen = [0.0, 0.1, -0.1, 0.0]

### Show popup window in the end with summary of experiment?
show_summary_plots = True

### Initial state
# This is just one possibility how to set the initial state. Feel free to modify this code
# [position, positionD, angle, angleD]
# Unassigned variables will be randomly initialized (see below if interested)
initial_state = [start_random_target_position_at_DataGen, None, None, None]

############ END OF PARAMETERS SECTION ############

initial_state_DataGen = create_cartpole_state()

for i in range(number_of_experiments):
    try: os.makedirs(record_path)
    except: pass

    csv = os.path.join(record_path, "Experiment")

    start_random_target_position_at_DataGen = used_track_fraction * TrackHalfLength * np.random.uniform(-1.0, 1.0)
    # initial_state = [start_random_target_position_at_DataGen, None, 0.0, None]

    if initial_state[0] is None:
        initial_state_DataGen[cartpole_state_varname_to_index('position')] = np.random.uniform(
            low=-TrackHalfLength / 2.0,
            high=TrackHalfLength / 2.0)
    else:
        initial_state_DataGen[cartpole_state_varname_to_index('position')] = initial_state[0]

    if initial_state[1] is None:
        initial_state_DataGen[cartpole_state_varname_to_index('positionD')] = np.random.uniform(low=-1.0,
                                                                                                high=1.0) * P_GLOBALS.TrackHalfLength *0.01
    else:
        initial_state_DataGen[cartpole_state_varname_to_index('positionD')] = initial_state[1]

    if initial_state[2] is None:
        if np.random.uniform()>0.5:
            initial_state_DataGen[cartpole_state_varname_to_index('angle')] = np.random.uniform(low=0 * (np.pi / 180.0),
                                                                                                high=180 * (np.pi / 180.0))
        else:
            initial_state_DataGen[cartpole_state_varname_to_index('angle')] = np.random.uniform(low=-180 * (np.pi / 180.0),
                                                                                                high=-0 * (np.pi / 180.0))
    else:
        initial_state_DataGen[cartpole_state_varname_to_index('angle')] = initial_state[2]

    if initial_state[3] is None:
        initial_state_DataGen[cartpole_state_varname_to_index('angleD')] = np.random.uniform(low=-10.0 * (np.pi / 180.0),
                                                                                             high=10.0 * (np.pi / 180.0))
    else:
        initial_state_DataGen[cartpole_state_varname_to_index('angleD')] = initial_state[3]
    
    # Add cos/sin values to state
    initial_state_DataGen[cartpole_state_varnames_to_indices(['angle_cos', 'angle_sin'])] = [
        np.cos(initial_state_DataGen[cartpole_state_varname_to_index('angle')]),
        np.sin(initial_state_DataGen[cartpole_state_varname_to_index('angle')])
    ]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # You may also specify some of the variables from above here, to make them change at each iteration.#
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    print(i)
    sleep(0.1)
    CartPoleInstance = CartPole()
    CartPoleInstance.setup_cartpole_random_experiment(
        # Initial state
        s0=initial_state_DataGen,

        # controller to be used in performed experiment
        controller=controller_DataGen,

        # Timescales
        dt_simulation=dt_simulation_DataGen,
        dt_controller=dt_controller_update_DataGen,
        dt_save=dt_save_DataGen,

        # Settings related to random trace generation
        track_relative_complexity=track_relative_complexity_DataGen,
        length_of_experiment=length_of_experiment_DataGen,
        interpolation_type=interpolation_type_DataGen,
        turning_points_period=turning_points_period_DataGen,
        start_random_target_position_at=start_random_target_position_at_DataGen,
        end_random_target_position_at=end_random_target_position_at_DataGen,
        turning_points=turning_points_DataGen,
        used_track_fraction=used_track_fraction,
    )
    gen_start = timeit.default_timer()

    ############ Profiling ############
    # with cProfile.Profile() as pr:
    #     CartPoleInstance.run_cartpole_random_experiment(
    #         csv=csv,
    #         save_mode=save_mode
    #     )
    # with open('profiling_stats.txt', 'w') as stream:
    #     stats = Stats(pr, stream=stream)
    #     stats.strip_dirs()
    #     stats.sort_stats('time')
    #     stats.dump_stats('.prof_stats')
    #     stats.print_stats()
    ###################################

    CartPoleInstance.run_cartpole_random_experiment(
        csv=csv,
        save_mode=save_mode,
        show_summary_plots=show_summary_plots
    )

    gen_end = timeit.default_timer()
    gen_dt = (gen_end - gen_start) * 1000.0
    print('time to generate data: {} ms'.format(gen_dt))

    try:
        CartPoleInstance.controller.controller_report()
    except:
        pass

