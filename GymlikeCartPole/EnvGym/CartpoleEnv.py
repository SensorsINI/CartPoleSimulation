
from typing import Optional, Union

import numpy as np

import gymnasium as gym

from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from GymlikeCartPole.Cartpole_RL.Cartpole_Sensors import Cartpole_Sensors
from GymlikeCartPole.EnvGym.state_utils import *

import math #testing


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
    ):
        self.max_episode_steps = max_episode_steps

        # self.cartpole_rl = Cartpole_OpenAI()
        self.cartpole_rl = Cartpole_Sensors()

        self.action_space = self.cartpole_rl.action_space
        self.observation_space = self.cartpole_rl.observation_space

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps = 0

        self.target_position = 0.0

        self.reset()

    def step(self, action):
        if not self.action_space.contains(
            action
        ):
            f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        self.state = self.cartpole_rl.get_next_state(self.state, action)

        terminated = self.cartpole_rl.termination_condition(self.state)

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        reward = self.cartpole_rl.reward_assignment(self.state, action, terminated, self.steps)
        # print(reward)
        # print(self.state[POSITION_IDX])
        # print(self.state[ANGLE_IDX])

        if self.render_mode == "human":
            self.render()

        if terminated:
            # reward -= 1.0
            print("crashed")
            # self.reset()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        # low, high = utils.maybe_parse_reset_bounds(
        #     options, -0.1, 0.1  # default low
        # )  # default high
        # custom states for reset:
        self.state = self.np_random.uniform(low=low, high=high, size=(6,))
        self.state[ANGLE_IDX] = self.np_random.uniform(low=-3.14, high=3.14, size=(1, ))
        # self.state[ANGLE_IDX] = 3.14
        # self.state[ANGLE_IDX] = -3.14
        # self.state[POSITION_IDX] = 44.0e-2/2
        self.state[ANGLE_COS_IDX] = np.cos(self.state[ANGLE_IDX])
        self.state[ANGLE_SIN_IDX] = np.sin(self.state[ANGLE_IDX])
        self.steps = 0
        self.cartpole_rl.reset()

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.cartpole_rl.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * self.cartpole_rl.pole_length_rendering
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = self.state[POSITION_IDX] * scale + self.screen_width / 2.0  # MIDDLE OF CART
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
            coord = pygame.math.Vector2(coord).rotate_rad(
                self.cartpole_rl.angle_rotation_direction_rendering * self.state[ANGLE_IDX]
            )
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
                int(self.target_position * scale + self.screen_width / 2.0),
                int(carty),
                int(10),
                (231, 76, 60),
            )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
