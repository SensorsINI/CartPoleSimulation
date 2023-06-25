import os
import timeit
from time import sleep

import numpy as np

from CartPole import CartPole
from CartPole.cartpole_model import TrackHalfLength, create_cartpole_state, L
from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                      ANGLED_IDX, POSITION_IDX, POSITIOND_IDX)
from others.globals_and_utils import create_rng, load_config
from others.p_globals import TrackHalfLength

# Uncomment if you want to get interactive plots for MPPI in Pycharm on MacOS
# On other OS you have to chose a different interactive backend.
# from matplotlib import use
# # use('TkAgg')
# use('macOSX')


class random_experiment_setter:
    def __init__(self):
        config = load_config("config_data_gen.yml")

        self.length_of_experiment = config["length_of_experiment"]

        self.controller = config["controller"]
        
        self.position_init = config["random_initial_state"]["position"]
        self.positionD_init = config["random_initial_state"]["positionD"]
        self.angle_init = config["random_initial_state"]["angle"]
        self.angleD_init = config["random_initial_state"]["angleD"]

        self.target_position_init = config["random_initial_state"]["target_position"]

        self.position_init_limits = config["random_initial_state"]["init_limits"]["position"]
        self.positionD_init_limits = config["random_initial_state"]["init_limits"]["positionD"]
        self.angle_init_limits = config["random_initial_state"]["init_limits"]["angle"]
        self.angleD_init_limits = config["random_initial_state"]["init_limits"]["angleD"]
        self.init_limits = [self.position_init_limits, self.positionD_init_limits, self.angle_init_limits, self.angleD_init_limits]

        self.start_at_target = config["start_at_target"]
        self.track_fraction_usable_for_target_position = config["track_fraction_usable_for_target_position"]
        self.target_position_end = config["target_position_end"]

        self.dt_simulation = config["dt"]["simulation"]
        self.dt_controller_update = config["dt"]["control"]
        self.dt_save = config["dt"]["saving"]

        self.track_relative_complexity = config["turning_points"]["track_relative_complexity"]
        self.interpolation_type = config["turning_points"]["interpolation_type"]
        self.turning_points = config["turning_points"]["turning_points"]
        self.turning_points_period = config["turning_points"]["turning_points_period"]
        if isinstance(self.turning_points_period, str) and self.turning_points_period == 'inf':
            self.turning_points_period = np.inf

        self.change_target_equilibrium_every_x_second = config['change_target_equilibrium_every_x_second']
        if isinstance(self.change_target_equilibrium_every_x_second, str) and self.change_target_equilibrium_every_x_second == 'inf':
            self.change_target_equilibrium_every_x_second = np.inf

        self.initial_target_equilibrium = config['initial_target_equilibrium']

        self.L_initial_mode = config['L']['L_initial']
        self.L_initial = None
        self.change_L_every_x_second = config['L']['change_L_every_x_second']
        if isinstance(self.change_L_every_x_second, str) and self.change_L_every_x_second == 'inf':
            self.change_L_every_x_second = np.inf
        self.L_discount_factor = config['L']['L_discount_factor']
        self.L_range = config['L']['L_range']
        self.L_informed_controller = config['L']['informed_controller']
        self.L_change_mode = config['L']['L_change_mode']
        self.L_step = config['L']['L_step']

        self.rng = create_rng(self.__class__.__name__, config["seed"])
        
    def set(self, CartPoleInstance: CartPole):
        
        # set initial_state

        initial_state_stub = create_cartpole_state()

        initial_state_stub[POSITION_IDX] = self.position_init
        initial_state_stub[POSITIOND_IDX] = self.positionD_init
        initial_state_stub[ANGLE_IDX] = self.angle_init
        initial_state_stub[ANGLED_IDX] = self.angleD_init

        initial_state = generate_random_initial_state(initial_state_stub, init_limits=self.init_limits, rng=self.rng)
        
        if self.start_at_target:
            start_random_target_position_at = initial_state[POSITION_IDX]
        else:
            if self.target_position_init is None:
               start_random_target_position_at = self.track_fraction_usable_for_target_position * \
                                                  TrackHalfLength * self.rng.uniform(-1.0, 1.0)
            else:
                start_random_target_position_at = self.target_position_init

        if self.target_position_end is None:
            end_random_target_position_at =\
                self.track_fraction_usable_for_target_position * TrackHalfLength * self.rng.uniform(-1.0, 1.0)
        else:
            end_random_target_position_at = self.target_position_end

        if self.initial_target_equilibrium == 'up' or self.initial_target_equilibrium == 1:
            target_equilibrium = 1
        elif self.initial_target_equilibrium == 'down' or self.initial_target_equilibrium == -1:
            target_equilibrium = -1
        elif self.initial_target_equilibrium == 'random':
            target_equilibrium = int(2*np.random.binomial(1, 0.5)-1)
        else:
            Exception('{} is not a valid specification for target equilibrium'.format(self.initial_target_equilibrium))

        global L
        if self.L_initial_mode == 'uniform':
            self.L_initial = np.random.uniform(*self.L_range)
        elif self.L_initial_mode == 'default':
            self.L_initial = L[...]
        else:
            self.L_initial = self.L_initial_mode

        CartPoleInstance.setup_cartpole_random_experiment(
            # Initial state
            s0=initial_state,

            # controller to be used in performed experiment
            controller=self.controller,

            # Timescales
            dt_simulation=self.dt_simulation,
            dt_controller=self.dt_controller_update,
            dt_save=self.dt_save,

            # Settings related to random trace generation
            track_relative_complexity=self.track_relative_complexity,
            length_of_experiment=self.length_of_experiment,
            interpolation_type=self.interpolation_type,
            turning_points_period=self.turning_points_period,
            start_random_target_position_at=start_random_target_position_at,
            end_random_target_position_at=end_random_target_position_at,
            turning_points=self.turning_points,
            used_track_fraction=self.track_fraction_usable_for_target_position,

            target_equilibrium=target_equilibrium,
            change_target_equilibrium_every_x_second=self.change_target_equilibrium_every_x_second,

            L_initial=self.L_initial,
            change_L_every_x_seconds=self.change_L_every_x_second,
            L_discount_factor=self.L_discount_factor,
            L_range=self.L_range,
            L_informed_controller=self.L_informed_controller,
            L_change_mode=self.L_change_mode,
            L_step=self.L_step,


        )

        return CartPoleInstance # ready to run a random experiment

def generate_random_initial_state(init_state_stub, init_limits, rng):

    position_init_limits, positionD_init_limits, angle_init_limits, angleD_init_limits = init_limits

    initial_state_post = create_cartpole_state()

    if np.isnan(init_state_stub[POSITION_IDX]):
        initial_state_post[POSITION_IDX] = rng.uniform(
            low=-1.0, high=1.0) * TrackHalfLength * position_init_limits
    else:
        initial_state_post[POSITION_IDX] = init_state_stub[POSITION_IDX]

    if np.isnan(init_state_stub[POSITIOND_IDX]):
        initial_state_post[POSITIOND_IDX] = rng.uniform(low=-1.0, high=1.0) * TrackHalfLength * positionD_init_limits
    else:
        initial_state_post[POSITIOND_IDX] = init_state_stub[POSITIOND_IDX]

    if np.isnan(init_state_stub[ANGLE_IDX]):
        if rng.uniform() > 0.5:
            initial_state_post[ANGLE_IDX] = rng.uniform(low=angle_init_limits[0], high=angle_init_limits[1]) * (np.pi / 180.0)
        else:
            initial_state_post[ANGLE_IDX] = rng.uniform(low=-angle_init_limits[1], high=-angle_init_limits[0]) * (np.pi / 180.0)
    else:
        initial_state_post[ANGLE_IDX] = init_state_stub[ANGLE_IDX]

    if np.isnan(init_state_stub[ANGLED_IDX]):
        initial_state_post[ANGLED_IDX] = rng.uniform(low=-1.0, high=1.0) * angleD_init_limits * (np.pi / 180.0)
    else:
        initial_state_post[ANGLED_IDX] = init_state_stub[ANGLED_IDX]

    # Add cos/sin values to state
    initial_state_post[ANGLE_COS_IDX] = np.cos(initial_state_post[ANGLE_IDX])
    initial_state_post[ANGLE_SIN_IDX] = np.sin(initial_state_post[ANGLE_IDX])

    return initial_state_post

def run_data_generator(run_for_ML_Pipeline=False, record_path=None):
    config = load_config("config_data_gen.yml")

    if record_path is None:
        record_path = config["PATH_TO_EXPERIMENT_RECORDINGS_DEFAULT"]
        csv = record_path + '/Experiment'

    number_of_experiments = config["number_of_experiments"]

    frac_train = config["split"][0]
    frac_val = config["split"][1]

    save_mode = config["save_mode"]

    show_summary_plots = config["show_summary_plots"]
    show_controller_report = config["show_controller_report"]

    if save_mode == 'online':
        if show_summary_plots is True or show_controller_report is True:
            raise PermissionError("You cannot plot summary if save_mode is online")


    RES = random_experiment_setter()

    ############ END OF PARAMETERS SECTION ############

    for i in range(number_of_experiments):

        if run_for_ML_Pipeline:
            if i < int(frac_train*number_of_experiments):
                csv = record_path + "/Train"
            elif i < int((frac_train+frac_val)*number_of_experiments):
                csv = record_path + "/Validate"
            else:
                csv = record_path + "/Test"

            try:
                os.makedirs(csv)
            except:
                pass

            csv += "/Experiment"

        

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # You may also specify some of the variables from above here, to make them change at each iteration.#
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        print('{}/{}'.format(i+1, number_of_experiments))
        sleep(0.1)
        CartPoleInstance = CartPole()

        CartPoleInstance = RES.set(CartPoleInstance)
        
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
        # with open('profiling_stats.txt', 'w', newline='') as stream:
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
        gen_dt = (gen_end - gen_start)
        print('time to generate data: {} ms'.format(gen_dt * 1000.0))
        print('Speed-up: {}'.format(float(config['length_of_experiment'])/gen_dt))

        if show_controller_report:
            try:
                CartPoleInstance.controller.controller_report()
            except:
                pass


if __name__ == '__main__':
    run_data_generator()
