import os
import yaml

from SI_Toolkit.load_and_normalize import calculate_normalization_info

config = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_training.yml'), 'r'), Loader=yaml.FullLoader)

calculate_normalization_info(config=config)
