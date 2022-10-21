from types import SimpleNamespace
from SI_Toolkit.computation_library import PyTorchLibrary, TensorType

import numpy as np
import torch
import yaml
import os
from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.load_and_normalize import normalize_numpy_array

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_INDICES
except ModuleNotFoundError:
    print("SI_Toolkit_ASF not yet created")

from SI_Toolkit.Functions.General.Initialization import (get_net,
                                                         get_norm_info_for_net)
from SI_Toolkit.Functions.Pytorch.Network import get_device

config_controller = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml"), "r"), Loader=yaml.FullLoader)
NET_NAME = config_controller["neural-imitator-pytorch"]["net_name"]
PATH_TO_MODELS = config_controller["neural-imitator-pytorch"]["PATH_TO_MODELS"]


class controller_neural_imitator_pytorch(template_controller):
    _computation_library = PyTorchLibrary
    
    def configure(self):
        a = SimpleNamespace()
        self.batch_size = 1  # It makes sense only for testing (Brunton plot for Q) of not rnn networks to make bigger batch, this is not implemented
        
        a.path_to_models = PATH_TO_MODELS
        a.net_name = NET_NAME

        self.device = get_device()

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        self.normalization_info = get_norm_info_for_net(self.net_info)

        self.net.reset()
        self.net.eval()

    def step(self, s: np.ndarray, time=None, updated_attributes: dict[str, TensorType]={}):
        self.update_attributes(updated_attributes)

        net_input = s[
            ..., [STATE_INDICES.get(key) for key in self.net_info.inputs[:-1]]
        ]  # -1 is a fix to exclude target position
        net_input = np.append(net_input, self.target_position)

        net_input = normalize_numpy_array(
            net_input, self.net_info.inputs, self.normalization_info
        )

        net_input = np.reshape(net_input, [-1, 1, len(self.net_info.inputs)])

        net_input = torch.tensor(net_input).float().to(self.device)

        net_output = self.net(net_input)

        Q = float(net_output)

        return Q
