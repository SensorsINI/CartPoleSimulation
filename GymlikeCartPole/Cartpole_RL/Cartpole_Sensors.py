import math

import numpy as np

from gymnasium import logger, spaces

from GymlikeCartPole.EnvGym.state_utils import *
from GymlikeCartPole.Cartpole_RL._cartpole_rl_template import CartPoleRLTemplate

from SI_Toolkit.computation_library import NumpyLibrary
from CartPole.cartpole_equations import CartPoleEquations
from CartPole.data_generator import random_experiment_setter
from others.globals_and_utils import load_config


class Cartpole_Sensors(CartPoleRLTemplate):

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

        self.pole_length_rendering = self.cpe.params.L  # Heuristic, for rendering only, proportional to physical pole length
        self.angle_rotation_direction_rendering = 1  # Heuristic, for rendering only, 1 or -1

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = self.cpe.params.TrackHalfLength

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.theta_threshold_radians * 2,
                np.inf,
                1.0,
                1.0,
                self.x_threshold * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.steps_beyond_terminated = None

    def get_next_state(self, state, action):
        new_state = np.squeeze(self.cpe.cartpole_fine_integration(state, Q=action, t_step=self.simulation_time_step, intermediate_steps=self.number_of_intermediate_integration_steps))
        return new_state

    def reward_assignment(self, state, action, terminated):
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        return reward

    def termination_condition(self, state):
        terminated = bool(
            state[POSITION_IDX] < -self.x_threshold
            or state[POSITION_IDX] > self.x_threshold
            or state[ANGLE_IDX] < -self.theta_threshold_radians
            or state[ANGLE_IDX] > self.theta_threshold_radians
        )

        return terminated

    def reset(self):
        self.steps_beyond_terminated = 0
        return
