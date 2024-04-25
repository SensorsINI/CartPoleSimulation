import os

from SI_Toolkit.load_and_normalize import calculate_normalization_info, load_yaml

path_to_config = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config_training.yml'))

config = load_yaml(path_to_config)


calculate_normalization_info(config=config)
