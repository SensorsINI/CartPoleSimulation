from CartPole import CartPole
from CartPole.cartpole_model import create_cartpole_state, TrackHalfLength
from others.p_globals import TrackHalfLength
from CartPole.state_utilities import ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX

import os
from time import sleep
import timeit
from datetime import datetime
import cProfile
from pstats import Stats, SortKey

import numpy as np
from numpy.random import SFC64, Generator
# Uncomment if you want to get interactive plots for MPPI in Pycharm on MacOS
# On other OS you have to chose a different interactive backend.
# from matplotlib import use
# # use('TkAgg')
# use('macOSX')

import yaml, os
config_CartPole = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

def run_data_generator(run_for_ML_Pipeline=False, record_path=None):

    seed = config_CartPole["data_generator"]["SEED"]
    if seed == "None":
        seed = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0*7.0)  # Fully random

    reset_seed_for_each_experiment = False

    rng_data_generator = Generator(SFC64(seed))

    #csv = './adaptive_test/Experiment.csv'
    if record_path is None:
        record_path = config_CartPole["cartpole"]["PATH_TO_EXPERIMENT_RECORDINGS_DEFAULT"]
        csv = record_path + '/Exp-mppi-tf-2k-jitter-A'

    # User defined simulation settings
    ############ CHANGE THESE PARAMETERS AS YOU LIKE ############
    # How many experiments will be generated
    number_of_experiments = 60

    ###### Train/Val/Test split - only matters if you run it in ML Pipeline mode
    frac_train = 0.8
    frac_val = 0.19

    save_mode = 'online'  # It was intended to save memory usage, but it doesn't seems to help

    ###### Timescales
    dt_simulation_DataGen = 0.002  # simulation timestep
    dt_controller_update_DataGen = 0.02  # control rate
    dt_save_DataGen = 0.02  # save datapoints in csv in this interval

    ###### CartPole settings
    ### Length of each experiment in s:
    length_of_experiment_DataGen = 7

    ### Controller which should be used in generated experiment:
    controller_DataGen = 'mppi-tf'
    # Possible options: 'manual-stabilization', 'do-mpc', 'do-mpc-discrete', 'lqr', 'mppi'

    ### Randomly placed target points/s
    track_relative_complexity_DataGen = 1

    ### How to interpolate between turning points of random trace
    interpolation_type_DataGen = '0-derivative-smooth'
    # Possible options: '0-derivative-smooth', 'linear', 'previous'

    ### How turning points should be distributed
    turning_points_period_DataGen = 'regular'
    # Possible options: 'regular', 'random'

    ### Set the max for smoothly interpolated random target position to avoid bumping into track ends.
    used_track_fraction = 0

    ### List of target positions, can be None to simulate with random targets
    turning_points_DataGen = None
    # Example: turning_points_DataGen = [0.0, 0.1, -0.1, 0.0]

    ### Show popup window in the end with summary of experiment?
    show_summary_plots = False
    show_controller_report = False

    ############ END OF PARAMETERS SECTION ############

    initial_state_DataGen = create_cartpole_state()

    for i in range(number_of_experiments):

        # Take care - the seed will be the same as every experiment!
        if reset_seed_for_each_experiment:
            rng_data_generator = Generator(SFC64(seed))

        ### Where the target positions of the random experiment start and end
        end_random_target_position_at_DataGen = used_track_fraction * TrackHalfLength * rng_data_generator.uniform(-1.0,
                                                                                                                   1.0)
        ### Initial state
        # This is just one possibility how to set the initial state. Feel free to modify this code
        # [position, positionD, angle, angleD]
        # Unassigned variables will be randomly initialized (see below if interested)
        # initial_state = [start_random_target_position_at_DataGen, None, None, None]

        if run_for_ML_Pipeline:
            if i < int(frac_train*number_of_experiments):
                csv = record_path + "/Train"
            elif i < int((frac_train+frac_val)*number_of_experiments):
                csv = record_path + "/Validate"
            else:
                csv = record_path + "/Test"

            try: os.makedirs(csv)
            except: pass

            csv += "/Experiment"

        start_random_target_position_at_DataGen = used_track_fraction * TrackHalfLength * rng_data_generator.uniform(-1.0, 1.0)
        initial_state = [start_random_target_position_at_DataGen, 0.0, 0.0, 0.0]
        # initial_state = [start_random_target_position_at_DataGen, None, None, None]
        # initial_state = [0.0, None, 0.0, None]
        if initial_state[0] is None:
            initial_state_DataGen[POSITION_IDX] = rng_data_generator.uniform(
                low=-TrackHalfLength / 2.0,
                high=TrackHalfLength / 2.0)
        else:
            initial_state_DataGen[POSITION_IDX] = initial_state[0]

        if initial_state[1] is None:
            initial_state_DataGen[POSITIOND_IDX] = rng_data_generator.uniform(low=-1.0,
                                                                                                    high=1.0) * TrackHalfLength *0.01
        else:
            initial_state_DataGen[POSITIOND_IDX] = initial_state[1]

        if initial_state[2] is None:
            if rng_data_generator.uniform()>0.5:
                initial_state_DataGen[ANGLE_IDX] = rng_data_generator.uniform(low=0 * (np.pi / 180.0),
                                                                                                    high=180 * (np.pi / 180.0))
            else:
                initial_state_DataGen[ANGLE_IDX] = rng_data_generator.uniform(low=-180 * (np.pi / 180.0),
                                                                                                    high=-0 * (np.pi / 180.0))
        else:
            initial_state_DataGen[ANGLE_IDX] = initial_state[2]

        if initial_state[3] is None:
            initial_state_DataGen[ANGLED_IDX] = rng_data_generator.uniform(low=-10.0 * (np.pi / 180.0),
                                                                                                 high=10.0 * (np.pi / 180.0))
        else:
            initial_state_DataGen[ANGLED_IDX] = initial_state[3]

        # Add cos/sin values to state
        initial_state_DataGen[ANGLE_COS_IDX] = np.cos(initial_state_DataGen[ANGLE_IDX])
        initial_state_DataGen[ANGLE_SIN_IDX] = np.sin(initial_state_DataGen[ANGLE_IDX])

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # You may also specify some of the variables from above here, to make them change at each iteration.#
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        print('{}/{}'.format(i+1, number_of_experiments))
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
        # Uncommenting this block will save a file profiling_stats.txt in top-level directory
        # Visualize bottlenecks and code runtime using
        # snakeviz profiling_stats.txt
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

        if show_controller_report:
            try:
                CartPoleInstance.controller.controller_report()
            except:
                pass


if __name__ == '__main__':
    run_data_generator()
