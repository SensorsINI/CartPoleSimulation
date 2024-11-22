import numpy as np

from time import sleep

from others.globals_and_utils import load_config, my_logger
from Control_Toolkit.others.globals_and_utils import import_controller_by_name

from GymlikeCartPole.EnvGym.CartpoleEnv import CartPoleEnv
from GymlikeCartPole.mpc_cost_function import MPC_CostFunction

from SI_Toolkit.computation_library import NumpyLibrary

import matplotlib.pyplot as plt

controller_name = "neural-imitator"
# controller_name = "mpc"
controller_full_name = f"controller-{controller_name}".replace("-", "_")
Controller = import_controller_by_name(controller_full_name)

config = load_config("cartpole_physical_parameters.yml")

logger = my_logger(__name__)

env = CartPoleEnv(render_mode="human")
initial_environment_attributes = {'target_position': 0.0, 'target_equilibrium': 1.0}

controller = Controller('CartPole', (-1, 1), initial_environment_attributes)
controller.configure()

lib = NumpyLibrary()
mpc_cost = MPC_CostFunction(lib, initial_environment_attributes)

done = False
# env.reset()
state = env.state
action_buf = []
obs_buf = []
environment_attributes = initial_environment_attributes
for k in range(1000):
    env.render()
    sleep(0.01)
    action = np.clip(np.reshape(controller.step(state, updated_attributes=environment_attributes), (-1,)), -1.0, 1.0)
    action_buf.append(action)
    # action = env.action_space.sample()  # take a random action
    state, reward, terminated, truncated, _ = env.step(action)
    environment_attributes = {'target_position': 0.0, 'target_equilibrium': 1.0}  # This would probably come from info of the environment or augmented state
    mpc_reward = -mpc_cost.get_cost(state, action, environment_attributes=environment_attributes)
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
