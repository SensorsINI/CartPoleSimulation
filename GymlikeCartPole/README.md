Cartpole RL

At this commit the working files are:
[gymRL_trial1.py](gymRL_trial1.py) - launching script, it runs cartpole with PPO
[CartpoleEnvGym.py](CartpoleEnvGym.py) - the environment class
Other files are obsolete but kept for reference.

To obtain current environment class I copied standard gymnasium cartpole environment (stand 17.10.2024)
and modified by copying taking action definition from pendulum environment.
The action is now a continuous value in range [-1, 1] which is then multiplied by force magnitude.
I deleted CartPoleEnv vectorized, as it seemed not to work for me and I don't have time time nor need to fix it now.
I added code to truncate experiment after some number of steps.
I added lines for rendering the target position but not the logic to change it yet.

