from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from GymlikeCartPole.EnvGym.CartpoleEnv import CartPoleEnv

import matplotlib.pyplot as plt

import numpy as np

# env_name = "MountainCarContinuous-v0" # Not working
# env_name = "Pendulum-v1"
# env_name = "CartPole-v0"
# env = gym.make(env_name, render_mode=None)

TRAINING_TYPE = "ppo"  # or "sac"

env = CartPoleEnv(render_mode=None)

env = DummyVecEnv([lambda: env])

if TRAINING_TYPE == "ppo":
    model = PPO('MlpPolicy', env, verbose=1)
elif TRAINING_TYPE == "sac":
    model = SAC('MlpPolicy', env, verbose=1)
else:
    raise ValueError("Unsupported training type. Use 'ppo' or 'sac'.")

model.learn(total_timesteps=500000, progress_bar=True)

model.save(f"{TRAINING_TYPE} model")

