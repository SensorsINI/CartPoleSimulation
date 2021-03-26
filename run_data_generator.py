from CartPole import CartPole
from CartPole.cartpole_model import s0, TrackHalfLength, P_GLOBALS
from CartPole._CartPole_mathematical_helpers import cartpole_state_varname_to_index

from time import sleep
import timeit

import numpy as np

# User defined simulation settings
# csv = '2500/Test'  # Name with which data is saved, consecutive experiments will be save with increasing index attached
# csv = '25/Train/Train'
# csv = '25/Test/Test'
csv = '25/Validate/Validate'
number_of_experiments = 10  # How many experiments will be generated
save_mode = 'online'  # It was intended to save memory usage, but it doesn't seems to help. Leave it false.

# Timescales
dt_simulation_DataGen = 0.002  # simulation timestep
dt_controller_update_DataGen = 0.1
dt_save_DataGen = 0.1

# CartPole settings - check the effect first in GUI before you launch big data generation
length_of_experiment_DataGen = 600.0  # Length of each experiment in s
controller_DataGen = 'do-mpc'  # Controller which should be used in generated experiment
# Possible options for controller:
# 'manual-stabilization', 'do-mpc', 'lqr'
track_relative_complexity_DataGen = 0.1  # randomly placed target points/s
interpolation_type_DataGen = '0-derivative-smooth'  # Sets how to interpolate between turning points of random trace
# Possible choices: '0-derivative-smooth', 'linear', 'previous'
turning_points_period_DataGen = 'regular'  # How turning points should be distributed
# Possible choices: 'regular', 'random'
# Where the target position of the random experiment starts and end
start_random_target_position_at_DataGen = 0.0
end_random_target_position_at_DataGen = 0.0
# The list of turning points is set to None, no matter what is in globals.py
turning_points_DataGen = None

# initial state
# This is just one possibility how to set the initial state. Feel free to modify this code
# [position, positionD, angle, angleD]
initial_state = [None, None, None, None]
initial_state_DataGen = s0

# Set the max for smoothly interpolated random target position to avoid bumping into track ends.
used_track_fraction = 0.5


for i in range(number_of_experiments):

    start_random_target_position_at_DataGen = 0.0
    end_random_target_position_at_DataGen = 0.0

    if initial_state[0] is None:
        initial_state_DataGen[cartpole_state_varname_to_index('position')] = np.random.uniform(
            low=-TrackHalfLength / 2.0,
            high=TrackHalfLength / 2.0)
    else:
        initial_state_DataGen[cartpole_state_varname_to_index('position')] = initial_state[0]

    if initial_state[1] is None:
        initial_state_DataGen[cartpole_state_varname_to_index('positionD')] = np.random.uniform(low=-1.0,
                                                                                                high=1.0) * P_GLOBALS.TrackHalfLength *0.1
    else:
        initial_state_DataGen[cartpole_state_varname_to_index('positionD')] = initial_state[1]

    if initial_state[2] is None:
        initial_state_DataGen[cartpole_state_varname_to_index('angle')] = np.random.uniform(low=-3.5 * (np.pi / 180.0),
                                                                                            high=3.5 * (np.pi / 180.0))
    else:
        initial_state_DataGen[cartpole_state_varname_to_index('angle')] = initial_state[2]

    if initial_state[3] is None:
        initial_state_DataGen[cartpole_state_varname_to_index('angleD')] = np.random.uniform(low=-3.0 * (np.pi / 180.0),
                                                                                             high=3.0 * (np.pi / 180.0))
    else:
        initial_state_DataGen[cartpole_state_varname_to_index('angleD')] = initial_state[3]

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
    CartPoleInstance.run_cartpole_random_experiment(
        csv=csv,
        save_mode=save_mode
    )

    try:
        CartPoleInstance.controller.controller_report()
    except:
        pass

    gen_end = timeit.default_timer()
    gen_dt = (gen_end - gen_start) * 1000.0
    print('time to generate data: {} ms'.format(gen_dt))

# os.system('say "Antonio! Todo ha terminado!"')
