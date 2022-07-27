from importlib import import_module
import os
import yaml
from GymlikeCartPole.CartPoleEnv_LTC import CartPoleEnv_LTC
from time import sleep

from Controllers.controller_lqr import controller_lqr
from Controllers.controller_mppi import controller_mppi
from Controllers.controller_mppi_tf import controller_mppi_tf

controller_name = "mppi-tf"
controller_full_name = f"controller-{controller_name}".replace("-", "_")

Controller = getattr(import_module(f"Controllers.{controller_full_name}"), controller_full_name)

config = yaml.load(open(os.path.join(os.path.dirname(__file__), "..", "config.yml"), "r"), Loader=yaml.FullLoader)

from others.globals_and_utils import my_logger
logger = my_logger(__name__)

env = CartPoleEnv_LTC()
controller = Controller(env.CartPoleInstance, **{**config["controller"][controller_name], **{"num_control_inputs": config["cartpole"]["num_control_inputs"]}})
done = False
state = env.state
for k in range(1000):
    env.render()
    sleep(0.01)
    action = controller.step(state)
    # action = env.action_space.sample()  # take a random action
    state, reward, done, _ = env.step(action)
    logger.info(f"Iteration {k}: Reward = {reward}")
    if done:
        break
env.close()