import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from stable_baselines3 import SAC, PPO
from tensorflow.python.keras.backend import dtype

# Define state grid
x_values = np.linspace(-0.2, 0.2, 50)  # Position range
theta_values = np.linspace(-math.pi, math.pi, 50)  # Angle range
value_estimates = np.zeros((len(x_values), len(theta_values)))

model_path = 'sac_cartpole_timescale_normalized_batch1024_withoutTerminal_0711'

model_type = model_path.split('_')[0]

if model_type == 'ppo':
    model = PPO.load(model_path)

if model_type == 'sac':
    model = SAC.load(model_path)
# print(model.critic)

# Loop over grid and estimate value
for i, x in enumerate(x_values):
    for j, theta in enumerate(theta_values):
        # print(theta)
        # Create a state tensor, fixing other state components as zero
        state = torch.tensor([[theta, 0, math.cos(theta), math.sin(theta), x, 0]], dtype=torch.float32)  # ["angle", "angleD", "angle_cos", "angle_sin", "position", "positionD",]
        action = torch.tensor([[0]], dtype=torch.float32)

        if model_type == 'ppo':
            q_value = model.policy.predict_values(state)
        if model_type == 'sac':
            q_value = model.critic(state, action)
        # print(min(q_value).item())
        # print(q1_value)
        value_estimates[i, j] = min(q_value).item()

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(value_estimates, extent=[-0.2, 0.2, -math.pi, math.pi], origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Estimated Value')
plt.xlabel('Position (x)')
plt.ylabel('Angle (theta)')
plt.title('Critic Value Function Heatmap')
plt.show()