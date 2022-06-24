from GymlikeCartPole.CartPoleEnv_LTC import CartPoleEnv_LTC
from time import sleep

from Controllers.controller_lqr import controller_lqr
from Controllers.controller_mppi import controller_mppi

env = CartPoleEnv_LTC()
controller = controller_mppi()
done = False
state = env.state
target = env.target
while not done:
    env.render()
    sleep(0.01)
    action = controller.step(state, target)
    # action = env.action_space.sample()  # take a random action
    state, target, reward, done, _ = env.step(action)
env.close()