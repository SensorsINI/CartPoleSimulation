# GymlikeCartPole/EnvGym/tasks.py
"""
Task definitions for Cart-Pole.  Each Task encapsulates
• how the episode starts (state_init)
• when it ends     (done)
• how it is scored (reward)

New tasks = subclass Task and override 3 small methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

# ----- indices shared with the rest of the code base -----
from GymlikeCartPole.EnvGym.state_utils import (
    ANGLE_IDX, ANGLED_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX,
    POSITION_IDX
)


class Task(ABC):
    """Stateless strategy object plugged into CartPoleEnv."""

    max_episode_steps: int  # injected by the Env after construction

    # -------- public API --------
    def init_state(self, rng: np.random.Generator) -> np.ndarray:
        """
        Provide an initial 6-D state vector.
        May be overridden for fancier starts (energy-based swing-up etc.).
        """
        low, high = -0.05, 0.05
        s = rng.uniform(low=low, high=high, size=(6,))
        s[ANGLE_COS_IDX] = np.cos(s[ANGLE_IDX])
        s[ANGLE_SIN_IDX] = np.sin(s[ANGLE_IDX])
        return s.astype(np.float32)

    @abstractmethod
    def done(self, state: np.ndarray) -> bool: ...

    @abstractmethod
    def reward(self,
               state:    np.ndarray,
               action:   np.ndarray,
               step_idx: int,
               terminated: bool
               ) -> float: ...


# ──────────────────────────────────────────────────────────
# Concrete implementations
# ──────────────────────────────────────────────────────────

class Stabilization(Task):
    """Keep the pole upright and cart centred."""

    ANGLE_LIMIT = 12 * np.pi / 180        # ±12°
    X_LIMIT     = 0.18                    # metres

    def done(self, state: np.ndarray) -> bool:
        return abs(state[ANGLE_IDX]) > self.ANGLE_LIMIT \
            or abs(state[POSITION_IDX]) > self.X_LIMIT

    def reward(self,
               state:    np.ndarray,
               action:   np.ndarray,
               step_idx: int,
               terminated: bool
               ) -> float:
        # small *shaping* to speed up learning:
        r = 1.0 \
            - 0.1  * abs(state[ANGLE_IDX]) \
            - 0.02 * abs(action[0])
        return r


class SwingUp(Task):
    """Swing the pendulum to upright and keep it there."""

    X_LIMIT = 0.18    # track half-length

    def done(self, state: np.ndarray) -> bool:
        # only terminate if we leave the track
        return abs(state[POSITION_IDX]) > self.X_LIMIT

    def reward(self,
               state:    np.ndarray,
               action:   np.ndarray,
               step_idx: int,
               terminated: bool
               ) -> float:
        upright   = (state[ANGLE_COS_IDX] + 1) / 2        # ∈[0,1]
        centred   = 1 - abs(state[POSITION_IDX]) / self.X_LIMIT
        ang_vel_p = 0.05  * upright * abs(state[ANGLED_IDX])
        act_pen   = 0.001 * abs(action[0])
        r = 0.8 * upright + 0.2 * centred - ang_vel_p - act_pen

        # optional penalty if we crashed early
        if terminated and step_idx < self.max_episode_steps:
            r -= 1.0
        return float(r)


class StabilizationOpenAI(Task):
    """
    Replicates the reward / termination rules used by the classic
    Gym CartPole implementation (“+1 per step, +1 on failure”
    unless `sutton_barto_reward=True`).

    It also reproduces the internal `steps_beyond_terminated`
    bookkeeping so reward semantics are identical.
    """

    ANGLE_LIMIT = 12 * np.pi / 180      # ±12°
    X_LIMIT     = 2.4                   # metres  (matches OpenAI track)

    def __init__(self, *, sutton_barto_reward: bool = False):
        self._sutton_barto_reward = sutton_barto_reward
        self.steps_beyond_terminated: int | None = None

    # -------- Task API ----------------------------------------------------
    def init_state(self, rng: np.random.Generator) -> np.ndarray:
        """Clear bookkeeping then delegate to the default random start."""
        self.steps_beyond_terminated = None
        return super().init_state(rng)

    def done(self, state: np.ndarray) -> bool:
        return (
            abs(state[POSITION_IDX]) > self.X_LIMIT or
            abs(state[ANGLE_IDX])    > self.ANGLE_LIMIT
        )

    def reward(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        step_idx:   int,
        terminated: bool,
    ) -> float:
        if not terminated:
            return 0.0 if self._sutton_barto_reward else 1.0

        #  ───── we are *after* termination ─────
        if self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            # first call right after failure
            return -1.0 if self._sutton_barto_reward else 1.0

        # any additional (undefined-behaviour) calls
        self.steps_beyond_terminated += 1
        return -1.0 if self._sutton_barto_reward else 0.0


# handy factory --------------------------------------------------------------

TASK_REGISTRY = {
    "stabilization": Stabilization,
    "swingup":        SwingUp,
    "stabilization_openai": StabilizationOpenAI,
}
