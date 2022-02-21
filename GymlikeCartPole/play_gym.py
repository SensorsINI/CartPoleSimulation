from GymlikeCartPole.CartPoleEnv_LTC import CartPoleEnv_LTC
from time import sleep

from Controllers.controller_lqr import controller_lqr

env = CartPoleEnv_LTC()
controller = controller_lqr()
done = False
state = env.state
target = env.target
while not done:
    env.render()
    sleep(0.01)
    action = controller.step(state, target)
    state, target, reward, done, _ = env.step(action)  # take a random action
env.close()