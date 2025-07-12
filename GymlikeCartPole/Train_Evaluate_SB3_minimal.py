from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from GymlikeCartPole.EnvGym.CartpoleEnv import CartPoleEnv

import matplotlib.pyplot as plt

import numpy as np

TRAINING_TYPE = "ppo"  # or "sac"


def make_env(render_mode=None):

    # env_name = "MountainCarContinuous-v0" # Not working
    # env_name = "Pendulum-v1"
    # env_name = "CartPole-v0"
    # env = gym.make(env_name, render_mode=render_mode)

    new_env = CartPoleEnv(render_mode=render_mode)

    return new_env


env = DummyVecEnv([lambda: make_env(render_mode=None)])

if TRAINING_TYPE == "ppo":
    model = PPO('MlpPolicy', env, verbose=1)
elif TRAINING_TYPE == "sac":
    model = SAC('MlpPolicy', env, verbose=1)
else:
    raise ValueError("Unsupported training type. Use 'ppo' or 'sac'.")

model.learn(total_timesteps=500000, progress_bar=True)

model.save(f"{TRAINING_TYPE} model")



# VISUAL EVALUATION
env = DummyVecEnv([lambda: make_env(render_mode="human")])
evaluate_policy(model, env, n_eval_episodes=2, render=True)
env.close()

# EVALUATION WITHOUT RENDERING
env = DummyVecEnv([lambda: make_env(render_mode=None)])

for episode in range(1, 3):
    score = 0
    action_buf = []
    obs_buf = []
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action, _ = model.predict(obs)
        action_buf.append(action)
        obs, reward, done, info = env.step(action)
        obs_buf.append(obs)
        score += reward

    print('Episode:', episode, 'Score:', score)

    action_buf = np.stack(action_buf)
    action_buf = np.squeeze(action_buf)
    plt.figure()
    plt.plot(action_buf)
    plt.title('Actions of RL agent')
    plt.xlabel('Timesteps')
    plt.ylabel('Action')
    plt.show()

env.close()
