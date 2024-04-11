import os

from SI_Toolkit.load_and_normalize import calculate_normalization_info

from others.globals_and_utils import load_config
config = load_config(os.path.join("SI_Toolkit_ASF", "config_training.yml"))


calculate_normalization_info(config=config)
