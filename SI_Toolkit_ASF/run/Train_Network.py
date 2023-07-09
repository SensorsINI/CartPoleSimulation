import os
from SI_Toolkit.Training.Train import train_network

config_dir = os.path.join('SI_Toolkit_ASF', 'experiment_configs')
for file in sorted(os.listdir(config_dir), key=lambda x: int(x[1:].split('.')[0])):
    config_path = os.path.join(config_dir, file)
    train_network(config_path=config_path)

# config_path = os.path.join('SI_Toolkit_ASF', 'config_training.yml')
# train_network(config_path=config_path)