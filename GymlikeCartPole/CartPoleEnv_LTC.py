
import gym
from CartPole import CartPole
from CartPole.cartpole_model import ANGLE_IDX, POSITION_IDX, cartpole_fine_integration_s
from others.p_globals import TrackHalfLength

import math
import numpy as np
import yaml
from datetime import datetime

from typing import Optional, Union

import pygame
from pygame import gfxdraw

# FIXME: Set reset properly
# FIXME: set random position - pregenerate
# FIXME: Check saving of first step
# TODO: Version for swing-up not stabilisation
# TODO: saving the episode after finished
# TODO: Make rendering with GUI/matplotlib animation, including also target position


config = yaml.load(open("GymlikeCartPole/config_gym.yml", "r"), Loader=yaml.FullLoader)
intermediate_steps = config["intermediate_steps"]
dt_control = config["dt_control"]
length_of_episode = config["length_of_episode"]
mode = config["mode"]

class CartPoleEnv_LTC(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):

        self.CartPoleInstance = CartPole()
        self.mode = mode

        self.intermediate_steps = intermediate_steps
        self.t_step_fine = np.float32(dt_control / float(self.intermediate_steps))
        self.CartPoleInstance.dt_simulation = dt_control  # This is because fine stepping is not done with CartPoleInstance method

        self.CartPoleInstance.length_of_experiment = length_of_episode

        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_stabilization_radians = 24 * np.pi / 360  # Fail episode if goes beyond this
        self.x_threshold = 0.9 * TrackHalfLength  # Takes care that the cart is not going beyond the boundary


        observation_space_boundary = np.array([
            np.float32(TrackHalfLength),
            np.finfo(np.float32).max,
            np.float32(np.pi),
            np.finfo(np.float32).max])

        self.action_space = gym.spaces.Box(
            low=np.float32(self.min_action),
            high=np.float32(self.max_action),
            shape=(1,)
        )

        self.observation_space = gym.spaces.Box(-observation_space_boundary, observation_space_boundary)

        self.viewer = None
        self.screen = None
        self.isopen = False

        self.state = None
        self.action = None
        self.reward = None
        self.target = None
        self.done = False

        self.steps_beyond_done = None

        self.reset()

    def step(self, action):

        self.action = np.atleast_1d(action).astype(np.float32)

        assert self.action_space.contains(self.action), \
            "%r (%s) invalid" % (self.action, type(self.action))

        self.step_physics()

        self.step_termination_and_reward()

        return self.state, self.CartPoleInstance.target_position, self.reward, self.done, {}

    def step_physics(self):

        # Cast action to float to strip np trappings
        self.CartPoleInstance.Q = self.action

        # Convert dimensionless motor power to a physical force acting on the Cart
        self.CartPoleInstance.Q2u()

        self.CartPoleInstance.s = cartpole_fine_integration_s(self.CartPoleInstance.s, self.CartPoleInstance.u,
                                                              self.t_step_fine, self.intermediate_steps)

        self.CartPoleInstance.add_noise_and_latency()

        # Update the total time of the simulation
        self.CartPoleInstance.step_time()

        # Update target position depending on the mode of operation
        # self.CartPoleInstance.update_target_position()
        self.CartPoleInstance.target_position = 0.0  # TODO: Make option of random target position

        # Save data to internal dictionary
        self.CartPoleInstance.save_csv_routine()

        self.state = self.CartPoleInstance.s_with_noise_and_latency

    def step_termination_and_reward(self):
        if self.mode ==  'stabilization':
            if not self.done:
                self.done = self.state[POSITION_IDX] < -self.x_threshold \
                            or self.state[POSITION_IDX] > self.x_threshold \
                            or self.state[ANGLE_IDX] < -self.theta_threshold_stabilization_radians \
                            or self.state[ANGLE_IDX] > self.theta_threshold_stabilization_radians
                self.done = bool(self.done)

            reached_final_time = bool(self.CartPoleInstance.time >= self.CartPoleInstance.length_of_experiment)

            if not self.done:
                if reached_final_time:
                    self.done = True
                    self.reward = 10.0
                else:
                    self.reward = 1.0
            elif self.steps_beyond_done is None:
                # Pole just fell!
                self.steps_beyond_done = 0
                self.reward = 1.0
            else:
                if self.steps_beyond_done == 0:
                    gym.logger.warn("""
            You are calling 'step()' even though this environment has already returned
            done = True. You should always call 'reset()' once you receive 'done = True'
            Any further steps are undefined behavior.
                            """)
                self.steps_beyond_done += 1
                self.reward = 0.0

        elif self.mode == 'follow target position':
            raise NotImplementedError  # TODO What is a suitable reward&termination condition for following target position?
        elif self.mode == 'swing-up':
            raise NotImplementedError  # TODO What is a suitable reward&termination condition for swing-up task?
        else:
            raise ValueError('Unknown mode (definition of the task)')

    def reset(self):
        # TODO: Generate random target positions
        self.CartPoleInstance.set_cartpole_state_at_t0()  # TODO Make setting of options for initial state richer like in Data Generator
        self.state = self.CartPoleInstance.s
        self.target = self.CartPoleInstance.target_position
        self.done = False

    def render(self, mode="human"):
        screen_width = 1200
        screen_height = 800

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * 0.1
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = self.state[POSITION_IDX] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-self.state[ANGLE_IDX])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False