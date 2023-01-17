import tensorflow as tf
import numpy as np

from types import SimpleNamespace

from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.load_and_normalize import normalize_numpy_array
from others.p_globals import L

from others.globals_and_utils import load_config

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_INDICES, \
        CONTROL_INPUTS
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')

from SI_Toolkit.Functions.General.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.Functions.TF.Compile import Compile

from Control_Toolkit.Controllers.controller_mppi_tf import controller_mppi_tf
from SI_Toolkit.Functions.TF.Testing import denormalize, normalize

config = load_config("config.yml")
NET_NAME = config['controller']['neural-imitator-tf']['net_name']
PATH_TO_MODELS = config['controller']['neural-imitator-tf']['PATH_TO_MODELS']

SEED = config['controller']['mppi-tf']['seed']
NUM_CONTROL_INPUTS = 1
CC_WEIGHT = config['controller']['mppi-tf']['cc_weight']
R = config['controller']['mppi-tf']['R']
LBD = config['controller']['mppi-tf']['LBD']
MPC_HORIZON = config['controller']['mppi-tf']['mpc_horizon']
NUM_ROLLOUTS = config['controller']['mppi-tf']['num_rollouts']
DT = config['controller']['mppi-tf']['dt']
PREDICTOR_INTERM_STEPS = config['controller']['mppi-tf']['predictor_intermediate_steps']
NU = config['controller']['mppi-tf']['NU']
SQRTRHOINV = config['controller']['mppi-tf']['SQRTRHOINV']
GAMMA = config['controller']['mppi-tf']['GAMMA']
SAMPLING_TYPE = config['controller']['mppi-tf']['SAMPLING_TYPE']
NET_NAME_MPPI = config['controller']['mppi-tf']['NET_NAME']
PREDICTOR_NAME = config['controller']['mppi-tf']['predictor_name']

# Batch size for autoregressive mode 1 testing
if 'Autoregressive' == NET_NAME.split('-')[0] and NET_NAME.split('-')[-1] == '0':
    batch_size = 32
else:
    batch_size = 1

class controller_neural_imitator_tf(template_controller):

    """Parameters needed for autoregressive testing of mode 1"""
    number_iteration = 0
    use_mppi = True
    last_pole_length_prediction = 0.3
    input_history = None
    mppi_tf_for_autoregressive = None
    mppi_Q_predictions = None

    def __init__(self, environment, batch_size=batch_size, **kwargs):

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

        self.mppi_tf_for_autoregressive = controller_mppi_tf(environment,SEED,NUM_CONTROL_INPUTS,CC_WEIGHT,R,LBD,MPC_HORIZON,NUM_ROLLOUTS,DT,PREDICTOR_INTERM_STEPS,NU,SQRTRHOINV,GAMMA,SAMPLING_TYPE,NET_NAME_MPPI,PREDICTOR_NAME)

    def step(self, s, time=None, target_equilibrium=1.0):

        if 'Autoregressive' not in self.net_info.net_name:
            net_input = s[..., [STATE_INDICES.get(key) for key in self.net_info.inputs[:-1]]]  # -1 is a fix to exclude target position
            net_input = np.append(net_input, self.env_mock.target_position)

            net_input = normalize_numpy_array(net_input,
                                              self.net_info.inputs,
                                              self.normalization_info)

            net_input = np.reshape(net_input, [-1, 1, len(self.net_info.inputs)])

            net_input = tf.convert_to_tensor(net_input, tf.float32)

            net_output = self.evaluate_net(net_input)

            Q = float(net_output)
        else:
            if self.net_info.training_mode != 1:
                cartpole_state = s[..., [STATE_INDICES.get(key) for key in self.net_info.inputs[1:-1]]]  # 1 to exclude pole_length, -1 is a fix to exclude target position
                target_position = self.env_mock.target_position
                if self.number_iteration == 0:
                    pole_length = L
                else:
                    pole_length = self.last_pole_length_prediction

                net_input = np.append(np.append(pole_length, cartpole_state), target_position)

                net_input = normalize_numpy_array(net_input,
                                                  self.net_info.inputs,
                                                  self.normalization_info)

                net_input = np.reshape(net_input, [-1, 1, len(self.net_info.inputs)])

                net_input = tf.convert_to_tensor(net_input, tf.float32)

                net_output = self.evaluate_net(net_input)

                Q = float(denormalize(net_output[-1, :, 0].numpy(), self.normalization_info, what='Q'))
                self.last_pole_length_prediction = float(denormalize(net_output[-1, :, 1].numpy(), self.normalization_info))
                # print(self.last_pole_length_prediction)

            else:
                cartpole_state = s[..., [STATE_INDICES.get(key) for key in self.net_info.inputs[-6:-1]]]
                target_position = self.env_mock.target_position
                network_input_incomplete = np.append(cartpole_state, target_position)

                network_input_incomplete = normalize_numpy_array(network_input_incomplete,
                                                                 self.net_info.inputs[-6:],
                                                                 self.normalization_info)
                if len(self.net_info.inputs) == 8:
                    network_input_incomplete = np.append(target_equilibrium, network_input_incomplete)

                if self.input_history is None:
                    self.input_history = network_input_incomplete
                else:
                    self.input_history = np.vstack((self.input_history, network_input_incomplete))

                if self.number_iteration < self.net_info.total_washout + batch_size:
                    self.use_mppi = True
                else:
                    self.use_mppi = False

                if self.use_mppi:
                    Q = self.mppi_tf_for_autoregressive.step(s, time=None)
                    self.mppi_Q_predictions = Q
                else:
                    net_input = self.compose_network_input()
                    net_output = self.evaluate_net(net_input)

                    Q = float(denormalize([net_output[-1, -1, 0]], self.normalization_info, what='Q'))
                    self.last_pole_length_prediction = float(denormalize([net_output[-1, -1, 1]], self.normalization_info))

                    self.mppi_Q_predictions = self.mppi_tf_for_autoregressive.step(s, time=None)

            self.number_iteration += 1

        return Q

    @Compile
    def evaluate_net_f(self, net_input):
        # print('retracing evaluate_net_f')
        net_output = self.net(net_input)
        return net_output

    def compose_network_input(self):

        batches = []
        value = float(normalize([self.last_pole_length_prediction], self.normalization_info))
        pole_vec = np.full(self.net_info.total_washout, value).reshape(-1, 1)
        pole_vec = np.expand_dims(pole_vec, axis=0)
        for i in range(batch_size):
            start = (self.number_iteration-self.net_info.total_washout-batch_size)+i
            end = (self.number_iteration-batch_size)+i
            batch = self.input_history[start:end]
            batch = np.expand_dims(batch, axis=0)
            batch = np.concatenate((pole_vec, batch), axis=2)
            batches.append(batch)

        net_input = np.concatenate(batches, axis=0)

        return net_input
