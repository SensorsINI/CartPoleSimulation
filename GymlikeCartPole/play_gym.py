from time import sleep

from others.globals_and_utils import load_config, my_logger
from Control_Toolkit.others.globals_and_utils import import_controller_by_name

from GymlikeCartPole.CartPoleEnv_LTC import CartPoleEnv_LTC

controller_name = "mppi"
controller_full_name = f"controller-{controller_name}".replace("-", "_")
Controller = import_controller_by_name(controller_full_name)

config = load_config("cartpole_physical_parameters.yml")

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
