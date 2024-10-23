from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from GymlikeCartPole.EnvGym.CartpoleEnv import CartPoleEnv

import matplotlib.pyplot as plt

import numpy as np

# env_name = "MountainCarContinuous-v0" # Not working
# env_name = "Pendulum-v1"
# env_name = "CartPole-v0"
# env = gym.make(env_name)


# env = gym.make(env_name, render_mode=None)
env = CartPoleEnv(render_mode=None)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=500000, progress_bar=True)
# model.learn(total_timesteps=100000, progress_bar=True)

model.save('ppo_cartpole_swing_up_origin_laptop_stricter')


# env = gym.make(env_name, render_mode="human")
env = CartPoleEnv(render_mode="human")
env = DummyVecEnv([lambda: env])
evaluate_policy(model, env, n_eval_episodes=2, render=True)
env.close()


env = CartPoleEnv(render_mode=None)
env = DummyVecEnv([lambda: env])

for episode in range(1, 3):
    score = 0
    action_buf = []
    obs_buf = []
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
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