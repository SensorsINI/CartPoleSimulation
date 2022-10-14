import os
import yaml
from SI_Toolkit.Functions.TF.Testing import test_network_pole_length, test_network_control_input

config = yaml.load(open(os.path.join("SI_Toolkit_ASF", "config_training.yml"), "r"), yaml.FullLoader)

test_network_pole_length()

for output in config['training_default']['outputs']:
    if output == 'Q':
        test_network_control_input()
