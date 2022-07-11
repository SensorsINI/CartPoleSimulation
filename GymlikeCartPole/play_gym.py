from GymlikeCartPole.CartPoleEnv_LTC import CartPoleEnv_LTC
from time import sleep

from Controllers.controller_lqr import controller_lqr
from Controllers.controller_mppi import controller_mppi
from Controllers.controller_mppi_tf import controller_mppi_tf

from others.globals_and_utils import my_logger
logger = my_logger(__name__)

env = CartPoleEnv_LTC()
controller = controller_mppi_tf()
done = False
state = env.state
target = env.target
for k in range(1000):
    env.render()
    sleep(0.01)
    action = controller.step(state, target)
    # action = env.action_space.sample()  # take a random action
    state, target, reward, done, _ = env.step(action)
    logger.info(f"Iteration {k}: Reward = {reward}")
    if done:
        break
env.close()