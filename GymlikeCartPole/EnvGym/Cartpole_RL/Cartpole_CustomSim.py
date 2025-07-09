import math

import numpy as np

from gymnasium import logger, spaces

from GymlikeCartPole.EnvGym.state_utils import *
from GymlikeCartPole.EnvGym.Cartpole_RL._cartpole_rl_template import CartPoleSimulatorBase

from SI_Toolkit.computation_library import NumpyLibrary
from CartPole.cartpole_equations import CartPoleEquations
from CartPole.data_generator import random_experiment_setter
from others.globals_and_utils import load_config


class Cartpole_CustomSim(CartPoleSimulatorBase):

    def __init__(self, **kwargs):

        config = load_config("cartpole_physical_parameters.yml")["cartpole"]
        self.cpe = CartPoleEquations(
            lib=NumpyLibrary(),
            second_derivatives_mode=config["second_derivatives_mode"],
            second_derivatives_neural_model_path=config["second_derivatives_neural_model_path"]
            )

        self.RES = random_experiment_setter()

        self.simulation_time_step = self.RES.dt_simulation
        self.number_of_intermediate_integration_steps = int(self.RES.dt_controller_update/self.RES.dt_simulation)

        # Angle at which to fail the episode
        self.x_limit = self.cpe.params.TrackHalfLength

        self.pole_length = self.cpe.params.L

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
               np.pi,
                np.inf,
                1.0,
                1.0,
                self.x_limit * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.steps_beyond_terminated = None

    def next_state(self, state, action):
        new_state = np.squeeze(self.cpe.cartpole_fine_integration(state, Q=action, t_step=self.simulation_time_step, intermediate_steps=self.number_of_intermediate_integration_steps))
        return new_state.astype(np.float32)
