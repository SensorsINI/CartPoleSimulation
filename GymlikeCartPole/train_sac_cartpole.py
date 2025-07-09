#!/usr/bin/env python3
"""
train_sac_cartpole.py

Training script for SAC on custom CartPoleEnv with timestamped model saving.
"""

import os
import random
from datetime import datetime

import numpy as np
import torch

from gymnasium.wrappers import TimeLimit

from typing import Callable
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from GymlikeCartPole.EnvGym.CartpoleEnv import CartPoleEnv
from GymlikeCartPole.success_count_callback import RollingSuccessCountCallback

# ─── 0) CONFIGURATION ─────────────────────────────────────────────────────────

#  - "stabilization" → balance only from near-upright starts
#  - "swingup"      → random starts + swing-up reward shaping
TASK = "stabilization"
CARTPOLE_TYPE = "custom_sim"  # "openai", "custom_sim", "remote"
MAX_EPISODE_STEPS = 500

SEED = 42
N_ENVS = 16
TOTAL_TIMESTEPS = 300_000

NET_ARCH    = [32, 32]
BATCH_SIZE  = 256
INITIAL_LR  = 3e-4

# ─── Run-specific directory setup ─────────────────────────────────────────────
# Generate a unique folder for everything produced this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir   = os.path.join("runs", f"sac_cartpole_{CARTPOLE_TYPE}_{TASK}_{timestamp}")

print(f"Creating run directory: {run_dir}")

# Inside that, separate subfolders for models and various logs
MODEL_DIR = os.path.join(run_dir, "models")
LOG_DIR   = os.path.join(run_dir, "logs")

# Create the entire hierarchy in one go
for d in (MODEL_DIR,
          os.path.join(LOG_DIR, "tensorboard"),
          os.path.join(LOG_DIR, "best_model"),
          os.path.join(LOG_DIR, "eval_logs"),
          os.path.join(LOG_DIR, "checkpoints")):
    os.makedirs(d, exist_ok=True)

# ─── 1) REPRODUCIBILITY ────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_random_seed(SEED)

# ─── 2) ENV FACTORY ───────────────────────────────────────────────────────────
def make_env():
    """
    Instantiate the CartPoleEnv, then wrap with Monitor.
    Monitor records episode reward/length for logging callbacks.
    """
    env = CartPoleEnv(render_mode=None, task=TASK, cartpole_type=CARTPOLE_TYPE)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return Monitor(env)

# ─── 3) VECTORIZED TRAINING ENV ───────────────────────────────────────────────
train_env = SubprocVecEnv([make_env for _ in range(N_ENVS)], start_method="fork")
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
)
train_env.seed(SEED)

# ─── 4) MODEL SETUP ───────────────────────────────────────────────────────────
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear decay from initial_value to zero over training."""
    def lr_fn(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return lr_fn

model = SAC(
    "MlpPolicy",
    train_env,
    policy_kwargs=dict(net_arch=NET_ARCH),
    buffer_size=100_000,
    batch_size=BATCH_SIZE,
    learning_starts=1_000,
    train_freq=(1, "step"),
    gradient_steps=4,
    tau=0.02,
    ent_coef="auto",
    target_update_interval=1,
    use_sde=True,
    sde_sample_freq=4,
    learning_rate=linear_schedule(INITIAL_LR),
    verbose=1,
    tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
    seed=SEED,
    device="auto",
)

# ─── 5) CALLBACKS & EVAL ENV ───────────────────────────────────────────────────
# Create evaluation env with identical normalization (without loading any prior stats)
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
)
eval_env.training = False
eval_env.seed(SEED)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(LOG_DIR, "best_model"),
    log_path=os.path.join(LOG_DIR, "eval_logs"),
    eval_freq=5_000,
    deterministic=True,
)
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=os.path.join(LOG_DIR, "checkpoints"),
    name_prefix="sac_cp"
)

rolling_cb = RollingSuccessCountCallback(n_episodes=20, verbose=1)

callbacks = CallbackList([eval_callback, checkpoint_callback, rolling_cb])

# ─── 6) TRAINING & SAVING ────────────────────────────────────────────────────
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

model_name = f"sac_cartpole_arch{'x'.join(map(str, NET_ARCH))}_bs{BATCH_SIZE}_lr{INITIAL_LR:.0e}"
model_path = os.path.join(MODEL_DIR, f"{model_name}.zip")
vec_path   = os.path.join(MODEL_DIR, f"{model_name}_vecnorm.pkl")

model.save(model_path)
train_env.save(vec_path)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"🔍 Mean reward after training: {mean_reward:.1f} ± {std_reward:.1f}")


# ─── 7) CLEANUP ─────────────────────────────────────────────────────────────
train_env.close()
eval_env.close()