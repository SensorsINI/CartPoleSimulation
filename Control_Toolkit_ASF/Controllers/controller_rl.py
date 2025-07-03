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


class controller_rl(template_controller):
    _computation_library = NumpyLibrary

    def configure(self):
        self.cpe = CartPoleEquations(lib=NumpyLibrary())
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = self.cpe.params.TrackHalfLength
        self.smooth = False
        self.q_last = 0.0
        self.q_last_last = 0.0

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

        self.model_name = self.config_controller["net_name"]
        print(self.model_name)
        self.rl_model = SAC.load(self.config_controller["PATH_TO_MODELS"] + self.model_name,
                                 custom_objects={"action_space": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                                 "observation_space": spaces.Box(-high, high, dtype=np.float32)})

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        Q, _states = self.rl_model.predict(s, deterministic=True)
        if self.smooth:
            Q = (Q + self.q_last) / 2
            # Q = (Q + self.q_last + self.q_last_last)/3
            # self.q_last_last = self.q_last
            self.q_last = Q
        return Q

    def controller_reset(self):
        self.configure()


class PhysicalEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    # metadata = {
    #     "render_modes": ["human", "rgb_array"],
    #     "render_fps": 50,
    # }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
    ):
        self.controller = controller_rl('CartPole', ([-1.], [1.]),
                                        {'L': 0.1975, 'Q_ccrc': 0.0, 'target_equilibrium': 1.0, 'target_position': 0.0})
        self.cartpole_rl = Cartpole_Sensors()
        self.action_space = self.cartpole_rl.action_space
        self.observation_space = self.cartpole_rl.observation_space

        self.steps = 0
        self.max_episode_steps = 500

        self.reset()

    def step(self, action):
        self.state = self.cartpole_rl.get_next_state(self.state, action)

        terminated = self.cartpole_rl.termination_condition(self.state)

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        reward = self.cartpole_rl.reward_assignment(self.state, action, terminated, self.steps)

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
        self.state = self.np_random.uniform(low=low, high=high, size=(6,))
        # self.state[ANGLE_IDX] = self.np_random.uniform(low=-3.14, high=3.14, size=(1, ))
        # # self.state[ANGLE_IDX] = 3.14
        # # self.state[ANGLE_IDX] = -3.14
        # # self.state[POSITION_IDX] = 44.0e-2/2
        # self.state[ANGLE_COS_IDX] = np.cos(self.state[ANGLE_IDX])
        # self.state[ANGLE_SIN_IDX] = np.sin(self.state[ANGLE_IDX])
        # self.steps = 0
        self.cartpole_rl.reset()

        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        return None

    def close(self):
        return None

test = PhysicalEnv()
test.step(np.array([1]))