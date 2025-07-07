# Evaluate_RL_Cartpole.py

"""
Standalone evaluation for custom CartPoleEnv.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from GymlikeCartPole.EnvGym.CartpoleEnv import CartPoleEnv

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MODEL_DIR = "models"
N_EVAL    = 5
N_RENDER  = 2
SEED      = 42

# ─── MODEL SELECTION AND ALGO INFERENCE ───────────────────────────────────────
pattern     = os.path.join(MODEL_DIR, "*_cartpole_*.zip")
model_files = sorted(glob.glob(pattern))
if not model_files:
    raise FileNotFoundError(f"No model found in {MODEL_DIR}")
MODEL_FILE = model_files[-1]

# infer “sac” vs “ppo” from filename prefix
algo      = os.path.basename(MODEL_FILE).split("_")[0].lower()
AlgoClass = SAC if algo == "sac" else PPO

VEC_FILE = MODEL_FILE.replace(".zip", "_vecnorm.pkl")
print(f"Detected algorithm:  {algo.upper()}")
print(f"Loading {algo.upper()} model:  {MODEL_FILE}")
print(f"Using VecNormalize stats: {VEC_FILE}")


def make_raw_env(render_mode=None):
    """
    Instantiate the unwrapped CartPoleEnv. Wrapping with Monitor and
    VecNormalize happens downstream.
    """
    return CartPoleEnv(render_mode=render_mode)

# ─── 1) BATCH EVALUATION
batch_env = DummyVecEnv([lambda: Monitor(make_raw_env(render_mode=None))])
batch_env = VecNormalize.load(VEC_FILE, batch_env)
batch_env.training = False
batch_env.seed(SEED)

mean_reward, std_reward = evaluate_policy(
    AlgoClass.load(MODEL_FILE),
    batch_env,
    n_eval_episodes=N_EVAL,
    render=False,
)

print(f"[Batch Eval] {N_EVAL} episodes → mean {mean_reward:.2f} ± {std_reward:.2f}")
batch_env.close()

# ─── 2) HUMAN-RENDERED DEMOS
model = AlgoClass.load(MODEL_FILE)
for episode in range(1, N_RENDER + 1):
    # Create a fresh env for visualization
    raw_env = make_raw_env(render_mode="human")
    vis = DummyVecEnv([lambda: Monitor(raw_env)])
    vis = VecNormalize.load(VEC_FILE, vis)
    vis.training = False
    vis.seed(SEED)

    obs = vis.reset()  # returns only observations
    done = False
    actions, rewards = [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vis.step(action)  # VecEnv step returns 4-tuple
        actions.append(action)
        rewards.append(float(reward))

    total_reward = sum(rewards)
    print(f"[Render] Episode {episode} → Total Reward: {total_reward:.2f}")

    # Plot cumulative reward
    cum_rewards = np.cumsum(rewards)
    plt.figure()
    plt.plot(cum_rewards)
    plt.title(f"Cumulative Reward – Episode {episode}")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.show()

    vis.close()
    raw_env.close()
