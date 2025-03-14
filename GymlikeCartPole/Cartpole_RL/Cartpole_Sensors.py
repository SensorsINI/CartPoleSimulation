import math

import numpy as np

from gymnasium import logger, spaces
from numpy.ma.core import absolute

from GymlikeCartPole.EnvGym.state_utils import *
from GymlikeCartPole.Cartpole_RL._cartpole_rl_template import CartPoleRLTemplate

from SI_Toolkit.computation_library import NumpyLibrary
from CartPole.cartpole_equations import CartPoleEquations
from CartPole.data_generator import random_experiment_setter


class Cartpole_Sensors(CartPoleRLTemplate):

    def __init__(self, **kwargs):

        self.cpe = CartPoleEquations(lib=NumpyLibrary())

        self.RES = random_experiment_setter()

        self.simulation_time_step = self.RES.dt_simulation
        self.number_of_intermediate_integration_steps = int(self.RES.dt_controller_update/self.RES.dt_simulation)

        self.pole_length_rendering = self.cpe.params.L  # Heuristic, for rendering only, proportional to physical pole length
        self.angle_rotation_direction_rendering = 1  # Heuristic, for rendering only, 1 or -1

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = self.cpe.params.TrackHalfLength

        self.episode_reward = 0
        self.steps = 0
        # self.x_threshold = 0.17


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
        u = self.cpe.Q2u(action)
        new_state = np.squeeze(self.cpe.cartpole_fine_integration(state, u=u, t_step=self.simulation_time_step, intermediate_steps=self.number_of_intermediate_integration_steps))
        return new_state

    def reward_assignment(self, state, action, terminated, steps):
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
            # print("were giving -1.0")
            # reward = 0.0
            reward = -10.0

        # print(action)
        self.steps = steps
        angle = abs(state[ANGLE_IDX]/math.pi)
        pos = (abs(state[POSITION_IDX])/(44.0e-2/2))
        angle_scale = (state[ANGLE_COS_IDX]+1)/2
        time_scale = steps/500
        # print(time_scale)


        '''IRL REWARD IDEAS'''

        # #penalize deviation from upright position:
        # reward -= 5*angle
        # # print("upright reward: " + str(abs(state[ANGLE_IDX]/math.pi)))
        # #penalize high pole velocity near upright position:
        # # reward -= 0.1 * (((state[ANGLE_COS_IDX]+1)/2)**2) * abs(state[ANGLED_IDX])
        #
        # if abs(state[ANGLE_IDX]) < 0.3:
        #     # print('gfhg')
        #     reward += 5
        #     reward -= 2 * (((state[ANGLE_COS_IDX] + 1) / 2)**2) * abs(state[ANGLED_IDX])
        #     print("upright reward: ", 5 - 2 * (((state[ANGLE_COS_IDX] + 1) / 2)**2) * abs(state[ANGLED_IDX]))
        #
        # if terminated and steps < 25:
        #     reward -= 10 * (25 - steps)
        #     # print("early termination!")
        # #weighted towards end of epidose
        # # reward -= 2.5 * time_scale * (((state[ANGLE_COS_IDX] + 1) / 2)**2) * abs(state[ANGLED_IDX])
        #
        # # #any time
        # reward -= 0.5 * (((state[ANGLE_COS_IDX] + 1) / 2)**2) * abs(state[ANGLED_IDX])
        # #
        # # #penalty for being not in origin when near upright:
        # reward -=  2.5 * time_scale * (angle_scale * pos)
        #
        # # reward -= time_scale * 5 * angle
        # #
        # # reward -= pos
        #
        # reward -= 0.1 * abs(action)
        # #Track length -> 44.0e-2
        # # print("reward: " + str(reward))
        #
        # self.episode_reward += reward
        ''' End of IRL reward ideas'''


        ''' original reward of sac_cartpole_64size_10kbatch_timescale_1011.zip: '''
        reward -= angle
        # reward -= 2.5 * time_scale * (angle_scale * pos)
        reward -= angle_scale * pos
        reward -= 0.1 * abs(action)

        #new part of reward for velocity
        # reward -= 0.1 * (((state[ANGLE_COS_IDX] + 1) / 2)) * abs(state[ANGLED_IDX])

        self.episode_reward += reward
        ''' End of original reward of sac_cartpole_64size_10kbatch_timescale_1011.zip: '''

        return reward

    def termination_condition(self, state):
        # terminated = bool(
        #     state[POSITION_IDX] < -self.x_threshold
        #     or state[POSITION_IDX] > self.x_threshold
        #     # or state[ANGLE_IDX] < -self.theta_threshold_radians
        #     # or state[ANGLE_IDX] > self.theta_threshold_radians
        # )K
        # print("x: " + str(state[POSITION_IDX]))
        terminated = bool(
            state[POSITION_IDX] < -0.18
            or state[POSITION_IDX] > 0.18
            # or state[ANGLE_IDX] < -self.theta_threshold_radians
            # or state[ANGLE_IDX] > self.theta_threshold_radians
        )

        return terminated

    def get_angular_velocity(self, state):
        return state[ANGLED_IDX]

    def get_angle(self, state):
        return state[ANGLE_IDX]

    def reset(self):
        self.steps_beyond_terminated = 0
        # print(self.steps)
        # print("WE ARE RESETTING", self.steps)
        # print("episode length: " + str(self.steps))
        # print("episode reward: " + str(self.episode_reward))
        # print("reward per step: " + str(self.episode_reward/max(self.steps,1)))
        # self.steps = 0
        return
