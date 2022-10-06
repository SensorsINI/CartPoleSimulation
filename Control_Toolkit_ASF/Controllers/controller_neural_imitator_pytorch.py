import numpy as np

from types import SimpleNamespace

import torch

from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.load_and_normalize import normalize_numpy_array

from others.globals_and_utils import load_config

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_INDICES, \
        CONTROL_INPUTS
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')

from SI_Toolkit.Functions.Pytorch.Network import get_device

from SI_Toolkit.Functions.General.Initialization import get_net, get_norm_info_for_net

config = load_config("config.yml")
NET_NAME = config['controller']['neural-imitator-pytorch']['net_name']
PATH_TO_MODELS = config['controller']['neural-imitator-pytorch']['PATH_TO_MODELS']

class controller_neural_imitator_pytorch(template_controller):
    def __init__(self, predictor, batch_size=1, **kwargs):

        a = SimpleNamespace()
        self.batch_size = batch_size  # It makes sense only for testing (Brunton plot for Q) of not rnn networks to make bigger batch, this is not implemented

        a.path_to_models = PATH_TO_MODELS
        a.net_name = NET_NAME

        self.device = get_device()

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True, library='Pytorch')

        self.normalization_info = get_norm_info_for_net(self.net_info)

        self.net.reset()
        self.net.eval()

        super().__init__(predictor)

    def step(self, s, time=None):

        net_input = s[..., [STATE_INDICES.get(key) for key in self.net_info.inputs[:-1]]]  # -1 is a fix to exclude target position
        net_input = np.append(net_input, self.predictor.target_position)

        net_input = normalize_numpy_array(net_input,
                                          self.net_info.inputs,
                                          self.normalization_info)

        net_input = np.reshape(net_input, [-1, 1, len(self.net_info.inputs)])

        net_input = torch.tensor(net_input).float().to(self.device)

        net_output = self.net(net_input)

        Q = float(net_output)

        return Q
