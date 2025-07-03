from SI_Toolkit.Predictors.neural_network_evaluator import neural_network_evaluator
from SI_Toolkit.computation_library import TensorType, NumpyLibrary
from gymnasium import logger, spaces
from CartPole.cartpole_equations import CartPoleEquations
from typing import Optional, Union
from gymnasium.envs.classic_control import utils
from GymlikeCartPole.Cartpole_RL.Cartpole_Sensors import Cartpole_Sensors
import gymnasium as gym

import math
from stable_baselines3 import SAC, PPO

import numpy as np

from Control_Toolkit.Controllers import template_controller

try:
    from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import STATE_INDICES
except ModuleNotFoundError:
    print("SI_Toolkit_ASF not yet created")

class controller_rl_train(template_controller):
    _computation_library = NumpyLibrary

    def configure(self):
        self.action = 0
        self.action_ready = False
        self.Q_last = 0

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        #TODO: wait for learn thread to call step?
        # if train
        # print('step')
        # self.Q_last = self.action
        if self.action_ready == True:
            # print("I GOT HERE")
            # print('action ready')
            self.Q_last = self.action
            self.action_ready = False
        # print("step is being used")
        return self.Q_last

    def set_action_ready(self):
        self.action_ready = True

    def controller_reset(self):
        self.configure()