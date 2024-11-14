
from typing import Optional, Union
#
import numpy as np
#
import gymnasium as gym
import threading
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from GymlikeCartPole.Cartpole_RL.Cartpole_Sensors import Cartpole_Sensors
from GymlikeCartPole.EnvGym.state_utils import *
from Control_Toolkit.Controllers.controller_rl import controller_rl

import tensorflow as tf

import time


import math #testing
#
#TODO: This import gives an error...
# from GymlikeCartPole.Run_Controller_with_Gym import terminated


class PhysCartpoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(self, phys_thread=None, instance=None):
        # self.cartpole_rl = Cartpole_Sensors()
        # self.action_space = self.cartpole_rl.action_space
        # self.observation_space = self.cartpole_rl.observation_space
        # self.physical_cartpole_driver = physical_cartpole_driver
        self.connection_thread = phys_thread
        self.cartpole_instance = instance
        self.other_thread = None

        self.max_episode_steps = 500

        # self.cartpole_rl = Cartpole_OpenAI()
        self.cartpole_rl = Cartpole_Sensors()

        self.action_space = self.cartpole_rl.action_space
        self.observation_space = self.cartpole_rl.observation_space


        #
        # # self.render_mode = render_mode
        #
        # self.screen_width = 600
        # self.screen_height = 400
        # self.screen = None
        # self.clock = None
        # self.isopen = True
        # self.state: np.ndarray | None = None
        #
        self.steps = 0
        #
        # self.target_position = 0.0


    # def step(self, action):
    #     return obs, reward, done, info

    def get_real_state(self):
        return 0

    def give_real_input(self, input):
        pass

    def open_connection(self):
        # self.connection_thread = threading.Thread(target=self.physical_cartpole_driver.run)
        # self.connection_thread.daemon = True  # Allow thread to exit when main program exits
        self.connection_thread.start()

    def run_physical_cartpole(self):
        for i in range(100):
            print("getting state:" + str(self.cartpole_instance.s[0]))
            # reward = self.cartpole_rl.reward_assignment(self.cartpole_instance.s, action, terminated, self.steps)
            # print("reward: " + str(reward))
            time.sleep(0.5)

    def step(self, action):
        # for i in range (10000):
        if not self.action_space.contains(
                action
        ):
            f"{action!r} ({type(action)}) invalid"
        # assert self.state is not None, "Call reset before using step method."

        self.cartpole_instance.controller.action = action
        self.cartpole_instance.controller.set_action_ready()
        # print(self.cartpole_instance.controlEnabled)
        # terminated = False
        # if not self.cartpole_instance.controlEnabled:
        #     terminated = True
        terminated = self.cartpole_rl.termination_condition(self.cartpole_instance.s)
        # print('threshold: ' + str(self.cartpole_rl.x_threshold))
        # print("real x:" +  str(self.cartpole_instance.s[4]))
        # print(terminated)
        self.steps += 1
        # print(self.cartpole_instance.controller.action_ready)
        truncated = self.steps >= self.max_episode_steps
        # truncated = self.steps >= self.max_episode_steps

        # print("getting state:" + str(self.cartpole_instance.s[0]))
        #TODO: set the action for next 0.02 seconds
        time.sleep(0.02)
        #get reward for new state when action was taken for 0.02 seconds
        reward = self.cartpole_rl.reward_assignment(self.cartpole_instance.s, action, terminated, self.steps)
        # print("reward: " + str(reward))
        action_ready = False
        return np.array(self.cartpole_instance.s, dtype=np.float32), reward, terminated, truncated, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        #TODO: if we are sending actions directly, maybe this will be enough?
        print("reset start PRESS K")
        if self.steps >= 10:
            self.cartpole_instance.calibrate()  #AttributeError: 'NoneType' object has no attribute 'write'
        # self.cartpole_instance.calibrate()
        time.sleep(10)
        print("reset end")
        self.steps = 0
        return np.array(self.cartpole_instance.s, dtype=np.float32), {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
