import numpy as np

from time import sleep

from others.globals_and_utils import load_config, my_logger
from Control_Toolkit.others.globals_and_utils import import_controller_by_name

from GymlikeCartPole.EnvGym.CartpoleEnv import CartPoleEnv

import matplotlib.pyplot as plt

controller_name = "neural-imitator"
# controller_name = "mpc"
controller_full_name = f"controller-{controller_name}".replace("-", "_")
Controller = import_controller_by_name(controller_full_name)

config = load_config("cartpole_physical_parameters.yml")

logger = my_logger(__name__)

env = CartPoleEnv(render_mode="human")
controller = Controller('CartPole', (-1, 1), {'target_position': 0.0, 'target_equilibrium': 1.0})
controller.configure()
done = False
# env.reset()
state = env.state
action_buf = []
obs_buf = []
for k in range(1000):
    env.render()
    sleep(0.01)
    action = np.clip(np.reshape(controller.step(state), (-1,)), -1.0, 1.0)
    action_buf.append(action)
    # action = env.action_space.sample()  # take a random action
    state, reward, terminated, truncated, _ = env.step(action)
    logger.info(f"Iteration {k}: Reward = {reward}")
    if done:
        break

action_buf = np.stack(action_buf)
action_buf = np.squeeze(action_buf)
plt.figure()
plt.plot(action_buf, color='green')
plt.title('Actions of our controller')
plt.xlabel('Timesteps')
plt.ylabel('Action')
plt.show()

env.close()
