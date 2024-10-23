import numpy as np

from time import sleep

from stable_baselines3 import SAC, PPO

from others.globals_and_utils import load_config, my_logger
from Control_Toolkit.others.globals_and_utils import import_controller_by_name

from GymlikeCartPole.EnvGym.CartpoleEnv import CartPoleEnv

import matplotlib.pyplot as plt


config = load_config("cartpole_physical_parameters.yml")

logger = my_logger(__name__)

# model_path = 'GymlikeCartPole/sac_cartpole'
# model = SAC.load(model_path)

# model_path = 'GymlikeCartPole/sac_cartpole_random_angle'
# model = SAC.load(model_path)

# model_path = 'GymlikeCartPole/sac_cartpole_random_angle_speed'
# model = SAC.load(model_path)

# model_path = 'GymlikeCartPole/ppo model'
# model = PPO.load(model_path)
#
# model_path = 'GymlikeCartPole/ppo model swing up'
# model = PPO.load(model_path)

# model_path = 'GymlikeCartPole/sac_cartpole_swing_up_origin'
# model = SAC.load(model_path)

# model_path = 'GymlikeCartPole/sac_cartpole_swing_up_origin_laptop'
# model = SAC.load(model_path)

# model_path = 'GymlikeCartPole/ppo_cartpole_swing_up_origin'
# model = PPO.load(model_path)

model_path = 'GymlikeCartPole/ppo_cartpole_swing_up_origin_laptop_stricter'
model = PPO.load(model_path)

# model_path = 'GymlikeCartPole/ppo_cartpole_swing_up_origin_5'
# model = PPO.load(model_path)



model_name = model_path.split('/')[-1]
print(model_name)

env = CartPoleEnv(render_mode="human")

done = False
env.reset()
state = env.state
print(state)
# state = env.reset()
action_buf = []
obs_buf = []
total_reward = 0

for k in range(500):
    env.render()
    sleep(0.01)
    # action = np.clip(np.reshape(controller.step(state), (-1,)), -1.0, 1.0)
    action, _states = model.predict(state, deterministic=True)
    action_buf.append(action)
    # action = env.action_space.sample()  # take a random action
    state, reward, terminated, truncated, _ = env.step(action)
    logger.info(f"Iteration {k}: Reward = {reward}")
    total_reward += reward
    if done:
        break

print("total reward: " + str(total_reward))
action_buf = np.stack(action_buf)
action_buf = np.squeeze(action_buf)
plt.figure()
plt.plot(action_buf, color='green')
plt.title('Actions of RL Agent ' + model_name)
plt.xlabel('Timesteps')
plt.ylabel('Action')
plt.show()

env.close()
