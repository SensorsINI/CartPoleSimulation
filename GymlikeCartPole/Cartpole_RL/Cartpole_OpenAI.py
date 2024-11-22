"""

# These comments I copied over from original cartpole, with my corrections given changes I've made

## Description

This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
 in the left and right direction on the cart.

## Action Space

The action is a `ndarray` with shape `(1,)` which can take values in the range `[-1, 1]` indicating the direction and magnitude
 of the force the cart is pushed with.

**Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
 the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

## Observation Space

The observation is a `ndarray` with shape `(6,)` with the values corresponding to the following positions and velocities:

| Num | Observation           | Min                 | Max               |
|-----|-----------------------|---------------------|-------------------|
| 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
| 3   | Pole Angular Velocity | -Inf                | Inf               |
| 2   | Pole Angle Cos        | ? | ? |
| 2   | Pole Angle Sin        | ? |? |
| 0   | Cart Position         | -4.8                | 4.8               |
| 1   | Cart Velocity         | -Inf                | Inf               |


**Note:** While the ranges above denote the possible values for observation space of each element,
    it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
-  The cart x-position can be take values between `(-4.8, 4.8)`, but the episode terminates
   if the cart leaves the `(-2.4, 2.4)` range.
-  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
   if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

## Rewards
Since the goal is to keep the pole upright for as long as possible, by default, a reward of `+1` is given for every step taken, including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the time limit on the environment.

If `sutton_barto_reward=True`, then a reward of `0` is awarded for every non-terminating step and `-1` for the terminating step. As a result, the reward threshold is 0 for v0 and v1.

## Starting State
All observations are assigned a uniformly random value in `(-0.05, 0.05)`

## Episode End
The episode ends if any one of the following occurs:

1. Termination: Pole Angle is greater than ±12°
2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
3. Truncation: Episode length is greater than 500 (200 for v0)

## Arguments

Cartpole only has `render_mode` as a keyword for `gymnasium.make`.
On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

```python
import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="rgb_array")
env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
env.reset(seed=123, options={"low": -0.1, "high": 0.1})  # default low=-0.05, high=0.05
(array([ 0.03647037, -0.0892358 , -0.05592803, -0.06312564], dtype=float32), {})

```

| Parameter               | Type       | Default                 | Description                                                                                   |
|-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
| `sutton_barto_reward`   | **bool**   | `False`                 | If `True` the reward function matches the original sutton barto implementation                |

## Vectorized environment

To increase steps per seconds, users can use a custom vector environment or with an environment vectorizor.

```python
import gymnasium as gym
envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")
envs
CartPoleVectorEnv(CartPole-v1, num_envs=3)
envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
envs
SyncVectorEnv(CartPole-v1, num_envs=3)

```

## Version History
* v1: `max_time_steps` raised to 500.
    - In Gymnasium `1.0.0a2` the `sutton_barto_reward` argument was added (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/790))
* v0: Initial versions release.
"""

import math

import numpy as np

from gymnasium import logger, spaces
from GymlikeCartPole.EnvGym.state_utils import *
from GymlikeCartPole.Cartpole_RL._cartpole_rl_template import CartPoleRLTemplate


class Cartpole_OpenAI(CartPoleRLTemplate):

    def __init__(self):
        self._sutton_barto_reward = False

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.pole_length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.pole_length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.pole_length_rendering = 2 * self.pole_length  # Heuristic, for rendering only, proportional to physical pole length
        self.angle_rotation_direction_rendering = -1  # Heuristic, for rendering only, 1 or -1

        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

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
        x, x_dot, theta, theta_dot = state[POSITION_IDX], state[POSITIOND_IDX], state[ANGLE_IDX], state[ANGLED_IDX]
        force = self.force_mag * float(action)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
                       force + self.polemass_length * np.square(theta_dot) * sintheta
               ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.pole_length
                * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        new_state = np.zeros((6,), dtype=np.float64)
        new_state[POSITION_IDX], new_state[POSITIOND_IDX], new_state[ANGLE_IDX], new_state[
            ANGLED_IDX] = x, x_dot, theta, theta_dot

        return new_state

    def reward_assignment(self, state, action, terminated):
        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0

        return reward

    def termination_condition(self, state):
        terminated = bool(
            state[POSITION_IDX] < -self.x_threshold
            or state[POSITION_IDX] > self.x_threshold
            or state[ANGLE_IDX] < -self.theta_threshold_radians
            or state[ANGLE_IDX] > self.theta_threshold_radians
        )

        return terminated

    def reset(self):
        self.steps_beyond_terminated = None
        return
