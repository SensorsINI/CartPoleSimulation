import tensorflow as tf
import numpy as np

from types import SimpleNamespace

from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.load_and_normalize import normalize_numpy_array

from others.globals_and_utils import load_config

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_INDICES, \
        CONTROL_INPUTS
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')

from SI_Toolkit.Functions.General.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.Functions.TF.Compile import CompileTF

config = load_config("config.yml")
NET_NAME = config['controller']['neural-imitator-tf']['net_name']
PATH_TO_MODELS = config['controller']['neural-imitator-tf']['PATH_TO_MODELS']

class controller_neural_imitator_tf(template_controller):
    def __init__(self, environment, batch_size=1, **kwargs):

        a = SimpleNamespace()
        self.batch_size = batch_size  # It makes sense only for testing (Brunton plot for Q) of not rnn networks to make bigger batch, this is not implemented

        a.path_to_models = PATH_TO_MODELS
        a.net_name = NET_NAME

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        self.normalization_info = get_norm_info_for_net(self.net_info)

        # Make a prediction

        self.net_initial_input_without_Q = np.zeros([len(self.net_info.inputs) - len(CONTROL_INPUTS)], dtype=np.float32)

        net_input_type = tf.TensorSpec((self.batch_size, 1, len(self.net_info.inputs)), tf.float32)

        # Retracing tensorflow functions
        try:
            self.evaluate_net = self.evaluate_net_f.get_concrete_function(net_input=net_input_type)
        except:
            self.evaluate_net = self.evaluate_net_f

        super().__init__(environment)

    def step(self, s, time=None):

        net_input = s[..., [STATE_INDICES.get(key) for key in self.net_info.inputs[:-1]]]  # -1 is a fix to exclude target position
        net_input = np.append(net_input, self.env_mock.target_position)

        net_input = normalize_numpy_array(net_input,
                                          self.net_info.inputs,
                                          self.normalization_info)

        net_input = np.reshape(net_input, [-1, 1, len(self.net_info.inputs)])

        net_input = tf.convert_to_tensor(net_input, tf.float32)

        net_output = self.evaluate_net(net_input)

        Q = float(net_output)

        return Q

    @CompileTF
    def evaluate_net_f(self, net_input):
        # print('retracing evaluate_net_f')
        net_output = self.net(net_input)
        return net_output
