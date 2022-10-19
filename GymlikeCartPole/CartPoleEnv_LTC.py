import gym
import os
from gym.core import ObsType
from CartPole import CartPole
from CartPole.cartpole_model import ANGLE_IDX, POSITION_IDX
from CartPole.cartpole_numba import cartpole_fine_integration_s_numba
from others.p_globals import TrackHalfLength
from run_data_generator import random_experiment_setter

import numpy as np

from typing import Optional, Tuple, Union

from others.globals_and_utils import load_config, my_logger
logger = my_logger(__name__)

# FIXME: Set reset properly
# FIXME: set random position - pregenerate
# FIXME: Check saving of first step
# TODO: Version for swing-up not stabilisation
# TODO: saving the episode after finished
# TODO: Make rendering with GUI/matplotlib animation, including also target position

config = load_config(os.path.join("GymlikeCartPole", "config_gym.yml"))
length_of_episode = config["length_of_episode"]
mode = config["mode"]


class CartPoleEnv_LTC(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "video.frames_per_second": 50, "render_fps": 50}

    def __init__(self):

        self.CartPoleInstance = CartPole()
        self.RES = random_experiment_setter()
        self.mode = mode

        self.intermediate_steps = int(self.RES.dt_controller_update/self.RES.dt_simulation)

        self.RES.length_of_experiment = length_of_episode

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

        return self.state, self.reward, self.done, {"target": self.target}

    def step_physics(self):

        # Cast action to float to strip np trappings
        self.CartPoleInstance.Q = self.action

        # Convert dimensionless motor power to a physical force acting on the Cart
        self.CartPoleInstance.Q2u()

        self.CartPoleInstance.s = cartpole_fine_integration_s_numba(self.CartPoleInstance.s, self.CartPoleInstance.u,
                                                                    np.float32(self.CartPoleInstance.dt_simulation), self.intermediate_steps)

        self.CartPoleInstance.add_noise_and_latency()

        # Update the total time of the simulation
        self.CartPoleInstance.step_time()

        # Update target position depending on the mode of operation
        # self.CartPoleInstance.update_target_position()
        # self.CartPoleInstance.target_position = 0.0  # TODO: Make option of random target position

        # Save data to internal dictionary
        # FIXE: Not working for some reason
        # self.CartPoleInstance.save_csv_routine()

        self.state = self.CartPoleInstance.s_with_noise_and_latency

    def step_termination_and_reward(self):
        if self.mode == 'stabilization':
            if not self.done:
                # self.done = self.state[POSITION_IDX] < -self.x_threshold \
                #             or self.state[POSITION_IDX] > self.x_threshold \
                #             or self.state[ANGLE_IDX] < -self.theta_threshold_stabilization_radians \
                #             or self.state[ANGLE_IDX] > self.theta_threshold_stabilization_radians
                self.done = bool(self.done)

            self.reward = self.get_reward(self.state, self.action)
            self.done = self.is_done(self.state)

        elif self.mode == 'follow target position':
            raise NotImplementedError  # TODO What is a suitable reward&termination condition for following target position?
        elif self.mode == 'swing-up':
            raise NotImplementedError  # TODO What is a suitable reward&termination condition for swing-up task?
        else:
            raise ValueError('Unknown mode (definition of the task)')

    def get_reward(self, state, action):
        reached_final_time = bool(self.CartPoleInstance.time >= self.CartPoleInstance.length_of_experiment)

        if not self.is_done(state):
            if reached_final_time:
                reward = 10.0
            else:
                reward = 1.0
        elif self.steps_beyond_done is None:
            reward = 1.0
        else:
            reward = 0.0
        
        return reward
    
    def is_done(self, state):
        reached_final_time = bool(self.CartPoleInstance.time >= self.CartPoleInstance.length_of_experiment)
        if reached_final_time:
            done = True
        else:
            done = False
            
        if self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn("""
        You are calling 'step()' even though this environment has already returned
        done = True. You should always call 'reset()' once you receive 'done = True'
        Any further steps are undefined behavior.
                        """)
            self.steps_beyond_done += 1
        
        return done

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        self.CartPoleInstance = self.RES.set(self.CartPoleInstance)
        self.state = self.CartPoleInstance.s
        self.CartPoleInstance.target_position = 0.0
        self.target = self.CartPoleInstance.target_position
        self.done = False

        self.steps_beyond_done = None

        return self.state, {}

    def render(self):
        assert self.render_mode in self.metadata["render_modes"]
        import pygame
        from pygame import gfxdraw
        
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
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:  # render mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((screen_width, screen_height))

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
            coord = pygame.math.Vector2(coord).rotate_rad(+self.state[ANGLE_IDX])
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
        if hasattr(self, "target_position"):
            gfxdraw.filled_circle(
                self.surf,
                int(self.target * scale + screen_width / 2.0),
                int(carty),
                int(10),
                (231, 76, 60),
            )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.display.flip()

        if self.render_mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
