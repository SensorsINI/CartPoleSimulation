import sys
import os
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import TrainFreq

from GymlikeCartPole.EnvGym.PhysCartpoleEnv import PhysCartpoleEnv

from gymnasium import logger, spaces
import numpy as np
import math

sys.path.insert(0, os.path.abspath(os.path.join(".", "Driver")))
sys.path.insert(1, os.path.abspath(os.path.join(".", "Driver", "CartPoleSimulation")))

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # TF: If uncommented, only uses CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

os.chdir("Driver")

import tensorflow as tf
from DriverFunctions.PhysicalCartPoleDriver import PhysicalCartPoleDriver
from CartPoleSimulation.CartPole import CartPole
from globals import CONTROL_PERIOD_MS

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.

print("TF Devices:", tf.config.list_physical_devices())
print("TF Device Placement:", tf.config.get_soft_device_placement())
print("TF Float Type:", tf.keras.backend.floatx())

CartPoleInstance = CartPole()
CartPoleInstance.dt_controller = float(CONTROL_PERIOD_MS)/1000.0
PhysicalCartPoleDriverInstance = PhysicalCartPoleDriver(CartPoleInstance)
# PhysicalCartPoleDriverInstance.run()

import threading
import time
def other():
    for i in range(10000):
        # os.system(('cls' if os.name == 'nt' else 'clear'))
        print("angle: ", PhysicalCartPoleDriverInstance.s[0])
        time.sleep(0.02)


phys = threading.Thread(target=PhysicalCartPoleDriverInstance.run)
env = PhysCartpoleEnv(phys, PhysicalCartPoleDriverInstance)



'''load model'''
from SI_Toolkit.computation_library import TensorType, NumpyLibrary
from CartPole.cartpole_equations import CartPoleEquations
cpe = CartPoleEquations(lib=NumpyLibrary())
theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = cpe.params.TrackHalfLength
high = np.array(
            [
                theta_threshold_radians * 2,
                np.inf,
                1.0,
                1.0,
                x_threshold * 2,
                np.inf,
            ],dtype=np.float32,)

# # ### Custom Actor Model Size:
# policy_kwargs = dict(net_arch=[16, 16])
# model = SAC('MlpPolicy', env, policy_kwargs=policy_kwargs,
#             train_freq=(10, 'step'), batch_size=10000, verbose=1)

# path = "Driver/CartPoleSimulation/GymlikeCartPole/"
# model = SAC.load(path + "sac_cartpole_64size_10kbatch_timescale_1011")

# SAC only updates action about once every 3 controller step calls
model = SAC.load('/home/marcin/PycharmProjects/physical-cartpole/Driver/CartPoleSimulation/GymlikeCartPole/sac_cartpole_32size.zip',
                 env = env,custom_objects={"train_freq": (7, 'step')}, bcustom_objects={"action_space": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                                "observation_space": spaces.Box(-high, high, dtype=np.float32)}, force_reset=True)

# print(model.train_freq)

# model.batch_size = 256
# model.replay_buffer = model.replay_buffer_class(
#     model.replay_buffer.buffer_size,
#     model.observation_space,
#     model.action_space,
#     device=model.device,
#     n_envs=model.n_envs,
#     optimize_memory_usage=model.optimize_memory_usage,
# )
# model.train_freq = TrainFreq(n=5, unit='step')

# model.ent_coef = 0.001
# model.optimize_memory_usage = True
# model.device = "cuda"
# PPO updates pretty much every controller step call
# model = PPO.load('/home/marcin/PycharmProjects/physical-cartpole/Driver/CartPoleSimulation/GymlikeCartPole/ppo_cartpole_angle_dependent_reward.zip',
#                  env = env, custom_objects={"action_space": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
#                                 "observation_space": spaces.Box(-high, high, dtype=np.float32)})

# policy_kwargs = dict(net_arch=[32, 32])
# model = SAC('MlpPolicy', env, policy_kwargs=policy_kwargs,
#              batch_size=1024, verbose=1, train_freq=(5, 'step'))

# policy_kwargs = dict(net_arch=[32, 32])
# model = SAC('MlpPolicy', env, policy_kwargs=policy_kwargs,
#              batch_size=1024, verbose=1)
# model =
'''load model'''

env.open_connection()
# print(model.relpay_buffer)
# model.learn(total_timesteps=150000, progress_bar=True)


# learn_thread = threading.Thread(target=model.learn, kwargs={'total_timesteps': 1000, 'progress_bar': True})
# learn_thread.start()

# timesteps: 30000
model.learn(total_timesteps=50000, progress_bar=True)
model.save("sac_from_sac_cartpole_32size")

# print("JOINING")
# env.close_connection()
# other = threading.Thread(target=env.step)
# # other = threading.Thread(target=env.run_physical_cartpole)
# env.open_connection()
# other.start()

