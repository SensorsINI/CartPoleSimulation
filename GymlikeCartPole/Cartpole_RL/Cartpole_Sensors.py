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
            reward = 0.0
            # reward = -1.0

        # print(action)
        angle = abs(state[ANGLE_IDX]/math.pi)
        pos = (abs(state[POSITION_IDX])/(44.0e-2/2))
        angle_scale = (state[ANGLE_COS_IDX]+1)/2
        time_scale = steps/500
        # print(time_scale)
        # ###OLD REWARD
        #penalize deviation from upright position:
        reward -= angle
        # print("upright reward: " + str(abs(state[ANGLE_IDX]/math.pi)))
        #penalize high pole velocity near upright position:
        # reward -= 0.1 * (((state[ANGLE_COS_IDX]+1)/2)**2) * abs(state[ANGLED_IDX])
        # reward -= 0.1 * ((state[ANGLE_COS_IDX] + 1) / 2) * abs(state[ANGLED_IDX])
        #penalty for being not in origin when near upright:
        reward -= 2.5 * time_scale * (angle_scale * pos)

        reward -= 0.1 * abs(action)
        #Track length -> 44.0e-2
        # reward -= 0.1 * ( (state[ANGLE_COS_IDX]+1)/2 ) * max(1, 50*abs(state[POSITION_IDX]))
        # reward -= 0.1 * (((state[ANGLE_COS_IDX] + 1) / 2) *
        #                  (math.sqrt(abs(state[POSITION_IDX]) / (44.0e-2 / 2)) + abs(state[POSITION_IDX])/(44.0e-2/2)))
        # reward -= 0.5 * (((state[ANGLE_COS_IDX]+1)/2)**2) * (abs(state[POSITION_IDX])/(44.0e-2/2))
        # reward -= 0.5 * ((state[ANGLE_COS_IDX]+1)/2) * (abs(state[POSITION_IDX])/(44.0e-2/2))
        # reward -= 0.5 * ((state[ANGLE_COS_IDX] + 1) / 2) * (1-(1/(1 + ((abs(state[POSITION_IDX])/(44.0e-2/2))/0.1)**2)))
        # xtol =

        # atol = 0.001 #3e-4
        # xtol = 0.001
        # # # goal = abs(state[POSITION_IDX]) < xtol and abs(state[ANGLE_IDX]) < atol
        # # # print(goal)
        # # # print(abs(state[ANGLE_IDX]))
        # if abs(state[POSITION_IDX]) < xtol and abs(state[ANGLE_IDX]) < atol:
        #     print("I DID IT")
        # #     # print(0.5 * (abs(state[ANGLE_IDX]) / atol + abs(state[POSITION_IDX]) / xtol))
        # #     reward -= 0.5 * abs(action)
        #     reward += 10 * (1 - 0.25 * (abs(state[ANGLE_IDX])/atol + abs(state[POSITION_IDX])/xtol))
            # if abs(state[ANGLE_IDX]) < 1e-4:
            #     print("BETTER I DID IT")
            #     reward += 0.5
        # ### OLD REWARD

        # angle = state[ANGLE_IDX]**2
        # pos = (((state[ANGLE_COS_IDX]+1)/2) * state[POSITION_IDX])**2
        # reward += -0.1 * (5*angle + pos)
        # print(abs(state[ANGLE_IDX]))
        # print("pos: " + str(abs(state[POSITION_IDX])))
        # reward -= 0.1 * abs(state[POSITION_IDX]/(44.0e-2 / 2))
        # print(abs(state[POSITION_IDX]))
        # print(1 - abs(state[ANGLE_IDX])/atol)
        # print("reward: " + str(reward))
        # print("norm: " + str(abs(state[POSITI
        # ON_IDX]/(44.0e-2 / 2))))
        # print("swing: " + str((state[ANGLE_COS_IDX]+1)/2))
        # print(0.1 * ( (state[ANGLE_COS_IDX]+1)/2 ) * max(1, 10*abs(state[POSITION_IDX])))
        # print("pos reward: " + str(0.5 * ((state[ANGLE_COS_IDX]+1)/2) * (abs(state[POSITION_IDX])/(44.0e-2/2))))
        # print("original pos: " + str(abs(state[POSITION_IDX])/(44.0e-2/2)))

        # print("Lorenzian test: " + str(1-(1/(1 + ((abs(state[POSITION_IDX])/(44.0e-2/2))/0.1)**2))))
        ###debug prints:
        # print('Pos Term: ' + str(abs(state[POSITION_IDX])/(44.0e-2/2)))
        # print('Scaled Pos Term: ' + str(0.5 * ((state[ANGLE_COS_IDX]+1)/2) * (abs(state[POSITION_IDX])/(44.0e-2/2))))
        #
        # print('Upright term: ' + str((abs(state[ANGLE_IDX]/math.pi))))
        # print('Velocity term: ' + str(0.1 * ((state[ANGLE_COS_IDX]+1)/2) * abs(state[ANGLED_IDX])))

        #TODO: could try to add a reward which takes the proportion of uprightness to

        # reward -= abs((state[ANGLE_IDX]/math.pi)) + 0.1 * (state[ANGLED_IDX] ** 2) + 0.001*((state[POSITION_IDX]/(44.0e-2/2)) ** 2)

        return reward

    def termination_condition(self, state):
        # terminated = bool(
        #     state[POSITION_IDX] < -self.x_threshold
        #     or state[POSITION_IDX] > self.x_threshold
        #     # or state[ANGLE_IDX] < -self.theta_threshold_radians
        #     # or state[ANGLE_IDX] > self.theta_threshold_radians
        # )K
        terminated = bool(
            state[POSITION_IDX] < -0.17
            or state[POSITION_IDX] > 0.17
            # or state[ANGLE_IDX] < -self.theta_threshold_radians
            # or state[ANGLE_IDX] > self.theta_threshold_radians
        )

        return terminated

    def reset(self):
        self.steps_beyond_terminated = 0
        return
