from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
from typing import Callable

from GymlikeCartPole.EnvGym.CartpoleEnv import CartPoleEnv
from GymlikeCartPole.mpc_cost_function import MPC_CostFunction

from SI_Toolkit.computation_library import NumpyLibrary


import matplotlib.pyplot as plt

import numpy as np

# env_name = "MountainCarContinuous-v0" # Not working
# env_name = "Pendulum-v1"
# env_name = "CartPole-v0"
# env = gym.make(env_name)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

# env = gym.make(env_name, render_mode=None)
env = CartPoleEnv(render_mode=None)
env = DummyVecEnv([lambda: env])
model = SAC('MlpPolicy', env, learning_rate=linear_schedule(0.001), verbose=1)
# model = SAC('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=150000, progress_bar=True)
# model.learn(total_timesteps=30000, progress_bar=True)

model.save('sac_cartpole_angle_dependent_reward')


# env = gym.make(env_name, render_mode="human")
env = CartPoleEnv(render_mode="human")
env = DummyVecEnv([lambda: env])
evaluate_policy(model, env, n_eval_episodes=2, render=True)
env.close()


env = CartPoleEnv(render_mode=None)
env = DummyVecEnv([lambda: env])

initial_environment_attributes = {'target_position': 0.0, 'target_equilibrium': 1.0}
lib = NumpyLibrary
mpc_cost = MPC_CostFunction(lib, initial_environment_attributes)

for episode in range(1, 3):
    score = 0
    action_buf = []
    obs_buf = []
    # score_buf = []
    obs = env.reset()
    done = False
    environment_attributes = initial_environment_attributes

    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        action_buf.append(action)
        obs, reward, done, info = env.step(action)

        environment_attributes = {'target_position': 0.0,
                                  'target_equilibrium': 1.0}  # This would probably come from info of the environment or augmented state
        mpc_reward = -mpc_cost.get_cost(obs, action, environment_attributes=environment_attributes)
        reward = mpc_reward
        # print(reward)
        obs_buf.append(obs)
        score += reward
        # score_buf.append(score)

    print('Episode:', episode, 'Score:', score)

    action_buf = np.stack(action_buf)
    action_buf = np.squeeze(action_buf)
    plt.figure()
    plt.plot(action_buf)
    plt.title('Actions of RL agent ' + str(episode))
    plt.xlabel('Timesteps')
    plt.ylabel('Action')
    plt.show()

    # score_buf = np.stack(score_buf)
    # score_buf = np.squeeze(score_buf) 
    # plt.figure()
    # plt.plot(score_buf)
    # plt.title('Score of RL agent')
    # plt.xlabel('Timesteps')
    # plt.ylabel('Score')
    # plt.show()

env.close()