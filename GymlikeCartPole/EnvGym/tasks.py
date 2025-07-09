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

import math
import numpy as np
from typing import Callable, Optional


# ----- indices shared with the rest of the code base -----
from GymlikeCartPole.EnvGym.state_utils import (
    ANGLE_IDX, ANGLED_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX,
    POSITION_IDX
)


def random_target_x_generation_factory(
    x_limit: float,
    N: int,
    rng: Optional[np.random.Generator] = None
) -> Callable[[], float]:
    # ---------- set-up --------------------------------------------------------
    if rng is None:                               # allow dependency-injection
        rng = np.random.default_rng()             # for determinism pass np.random.Generator(seed)
    current_val = rng.uniform(-x_limit, x_limit)  # first target
    counter     = 0                               # how many times we've emitted it so far

    # ---------- the closure ---------------------------------------------------
    def next_target(reset: bool = False) -> float:
        nonlocal current_val, counter

        if reset:
            counter = 0
        # When `counter` reaches N, time to pick a fresh value
        if counter >= N:
            current_val = rng.uniform(-x_limit, x_limit)
            counter = 0                           # restart the tally

        counter += 1
        return current_val

    return next_target



class Task(ABC):
    """Stateless strategy object plugged into CartPoleEnv."""

    max_episode_steps: int  # injected by the Env after construction

    def __init__(self,
                 physics,
                 ):
        self.physics            = physics

        self.next_target_x = random_target_x_generation_factory(physics.x_limit, N=250)
        self.target_x = self.next_target_x(reset=True)

        self._upright_thresh = 0.1  # ~5.7°;
        self.upright_achieved = False  # used by SwingUp

    # -------- public API --------
    def init_state(self, rng: np.random.Generator, **kwargs) -> np.ndarray:
        """
        Provide an initial 6-D state vector.

        Default start:
        • θ ∈ [−π, π] (full circle)
        • θ̇ ∈ [−1200°/s, +1200°/s] converted to rad/s
        • x ∈ [−x_limit, +x_limit]
        • ẋ ∈ [−0.5, +0.5] m/s
        """
        # maximum angular velocity in rad/s (1200°/s)
        max_ang_vel = 1200 * np.pi / 180.0

        # sample raw state components
        angle = rng.uniform(low=-np.pi, high=np.pi)                        # full‐circle angle
        ang_vel = 0.0 * rng.uniform(low=-max_ang_vel, high=max_ang_vel)          # angular speed bound
        pos     = 0.8 * rng.uniform(low=-self.physics.x_limit,
                              high= self.physics.x_limit)                  # cart within track
        vel     = 0.0 * rng.uniform(low=-0.5, high=0.5)                          # linear speed bound

        # assemble the 6-vector: [θ, θ̇, cosθ, sinθ, x, ẋ]
        s = np.empty(6, dtype=np.float32)
        s[ANGLE_IDX]     = angle
        s[ANGLED_IDX]    = ang_vel
        s[ANGLE_COS_IDX] = np.cos(angle)
        s[ANGLE_SIN_IDX] = np.sin(angle)
        s[POSITION_IDX]  = pos
        s[POSITION_IDX+1]= vel

        self.target_x = self.next_target_x(reset=True)

        self.upright_achieved = False  # reset flag for SwingUp task

        return s


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
    def __init__(self, physics):
        super().__init__(physics)
        self.angle_limit = 1.0

    def done(self, state: np.ndarray) -> bool:
        return abs(state[ANGLE_IDX]) > self.angle_limit \
            or abs(state[POSITION_IDX]) > self.physics.x_limit

    def reward(self,
               state:    np.ndarray,
               action:   np.ndarray,
               step_idx: int,
               terminated: bool
               ) -> float:
        self.target_x = self.next_target_x()  # update target position
        if abs(state[ANGLE_IDX]) < self._upright_thresh:
            self.upright_achieved = True  # mark upright reached
        # small *shaping* to speed up learning:
        r = 1.0 \
            - 0.2  * (abs(self.target_x-state[POSITION_IDX])/2*self.physics.x_limit) \
            - 0.02 * abs(action[0])
        return r

    def init_state(self, rng: np.random.Generator, **kwargs) -> np.ndarray:
        """
        Stabilization starts near upright with low velocity, but random position
        anywhere on track. This prevents the agent from overfitting to centered starts.
        """
        low, high = -0.05, 0.05  # narrow bounds for angle and velocities

        angle   = rng.uniform(low=low, high=high)
        ang_vel = rng.uniform(low=low, high=high)
        pos     = rng.uniform(low=-self.physics.x_limit, high=self.physics.x_limit)
        vel     = rng.uniform(low=low, high=high)

        s = np.empty(6, dtype=np.float32)
        s[ANGLE_IDX]     = angle
        s[ANGLED_IDX]    = ang_vel
        s[ANGLE_COS_IDX] = np.cos(angle)
        s[ANGLE_SIN_IDX] = np.sin(angle)
        s[POSITION_IDX]  = pos
        s[POSITION_IDX+1]= vel

        self.target_x = self.next_target_x(reset=True)

        self.upright_achieved = False  # reset flag

        return s


class SwingUp(Task):
    """Swing the pendulum to upright and keep it there."""

    def __init__(self, physics, *, horizon: int = 500):
        super().__init__(physics)
        # once the pole passes within this small angle (radians),
        # we consider "upright reached"
        self._upright_thresh_done = 1.0


    def init_state(self, rng: np.random.Generator, **kwargs) -> np.ndarray:
        # clear the flag at the very start of each episode
        self.upright_achieved = False
        return super().init_state(rng)

    def done(self, state: np.ndarray) -> bool:

        if self.upright_achieved and abs(state[ANGLE_IDX]) > self._upright_thresh_done:
            return True

        # only terminate if we leave the track
        return abs(state[POSITION_IDX]) > self.physics.x_limit

    def reward(self,
               state:    np.ndarray,
               action:   np.ndarray,
               step_idx: int,
               terminated: bool
               ) -> float:

        # detect the moment we enter the "upright" region
        if not self.upright_achieved and abs(state[ANGLE_IDX]) < self._upright_thresh:
            # first time we’re effectively vertical → mark it
            self.upright_achieved = True

        upright   = (state[ANGLE_COS_IDX] + 1) / 2        # ∈[0,1]
        centred   = 1 - abs(state[POSITION_IDX]) / self.physics.x_limit
        ang_vel_p = 0.05  * upright * abs(state[ANGLED_IDX])
        act_pen   = 0.001 * abs(action[0])
        r = 0.8 * upright + 0.2 * centred - ang_vel_p - act_pen

        return float(r)


class StabilizationOpenAI(Task):
    """
    Replicates the reward / termination rules used by the classic
    Gym CartPole implementation (“+1 per step, +1 on failure”
    unless `sutton_barto_reward=True`).

    It also reproduces the internal `steps_beyond_terminated`
    bookkeeping so reward semantics are identical.
    """

    def __init__(self,
                 physics,
                 *,
                 horizon: int = 500,
                 sutton_barto_reward: bool = False):
        super().__init__(physics)
        self.angle_limit = 12 * 2 * math.pi / 360
        self._sutton_barto_reward = sutton_barto_reward
        self.steps_beyond_terminated = None

    # -------- Task API ----------------------------------------------------
    def init_state(self, rng: np.random.Generator, **kwargs) -> np.ndarray:
        """
        Stabilization starts near upright with low velocity, but random position
        anywhere on track. This prevents the agent from overfitting to centered starts.
        """

        low, high = -0.05, 0.05  # narrow bounds for angle and velocities

        angle   = rng.uniform(low=low, high=high)
        ang_vel = rng.uniform(low=low, high=high)
        pos     = rng.uniform(low=-self.physics.x_limit, high=self.physics.x_limit)
        vel     = rng.uniform(low=low, high=high)

        s = np.empty(6, dtype=np.float32)
        s[ANGLE_IDX]     = angle
        s[ANGLED_IDX]    = ang_vel
        s[ANGLE_COS_IDX] = np.cos(angle)
        s[ANGLE_SIN_IDX] = np.sin(angle)
        s[POSITION_IDX]  = pos
        s[POSITION_IDX+1]= vel

        self.steps_beyond_terminated = None
        return super().init_state(rng)

    def done(self, state: np.ndarray) -> bool:
        return (
            abs(state[POSITION_IDX]) > self.physics.x_limit or
            abs(state[ANGLE_IDX])    > self.angle_limit
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
