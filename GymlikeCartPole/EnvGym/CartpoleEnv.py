# CartpoleEnv.py

from typing import Optional, Union

import numpy as np

import gymnasium as gym

from GymlikeCartPole.EnvGym.Cartpole_RL.Cartpole_CustomSim import Cartpole_CustomSim
from GymlikeCartPole.EnvGym.Cartpole_RL.Cartpole_OpenAI import Cartpole_OpenAI

from GymlikeCartPole.EnvGym.tasks import TASK_REGISTRY, Task


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        task: str = "stabilization",
        cartpole_type: str = "openai",  # "openai", "custom_sim", "physical"
    ):

        self.max_episode_steps = max_episode_steps

        self.cartpole_type = cartpole_type

        if self.cartpole_type == "openai":
            self.cartpole_rl = Cartpole_OpenAI()
        elif self.cartpole_type == "custom_sim":
            self.cartpole_rl = Cartpole_CustomSim()
        elif self.cartpole_type == "physical":
            raise NotImplementedError(
                "Physical Cartpole type is not implemented in this environment."
            )
        else:
            raise ValueError(
                f"Unknown cartpole type: {self.cartpole_type}. "
                "Choose from 'openai', 'custom_sim', or 'physical'."
            )

        self.action_space = self.cartpole_rl.action_space
        self.observation_space = self.cartpole_rl.observation_space

        if isinstance(task, str):
            try:
                self.task: Task = TASK_REGISTRY[task](self.cartpole_rl, horizon=max_episode_steps)  # create Task instance
            except KeyError as e:
                raise ValueError(f"Unknown task '{task}'."
                                 f" Choose one of {list(TASK_REGISTRY)}") from e
        elif isinstance(task, Task):
            self.task = task
        else:
            raise TypeError("task must be either a str key or a Task instance")

        self.render_mode = render_mode
        self._viewer = None

        self.screen_width = 600
        self.screen_height = 400
        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps = 0

        self.target_position = 0.0

        self.reset()



    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"{action!r} not in {self.action_space}")
        assert self.state is not None, "Call reset before using step method."
        self.state = self.cartpole_rl.next_state(self.state, action)

        terminated = self.task.done(self.state)

        self.steps += 1
        reward = self.task.reward(self.state, action, self.steps, terminated)

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        self.state = self.task.init_state(self.np_random)
        self.steps = 0
        self.cartpole_rl.reset()

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    # ─── CartPoleEnv.render (replace the whole method) ───────────────────────────
    def render(self) -> Optional[np.ndarray]:
        """
        • ‘human’ → draw via PygameViewer, return None
        • ‘rgb_array’ → draw off-screen and return np.ndarray (H,W,3)
        • any other render_mode → noop
        """
        if self.render_mode is None:
            return None

        # lazy-load to avoid pygame import when running on a cluster
        if self._viewer is None:
            if self.render_mode not in {"human", "rgb_array"}:
                raise ValueError(f"Unsupported render_mode '{self.render_mode}'")
            from GymlikeCartPole.EnvGym.render import PygameViewer  # local import
            self._viewer = PygameViewer(self.screen_width,
                                        self.screen_height,
                                        self.metadata["render_fps"])

        frame = self._viewer.draw(self.state,
                                  physics=self.cartpole_rl,
                                  target_pos=getattr(self, "target_position", None))

        if self.render_mode == "rgb_array":
            return frame
        return None


    def close(self):
        """Tear down viewer if it was ever created."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self.isopen = False
