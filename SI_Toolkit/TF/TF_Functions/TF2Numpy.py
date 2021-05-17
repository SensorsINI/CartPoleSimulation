# These are RNN functions as used in tensorflow

# These imports should be deleted after development process is concluded
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_math_ops

# These imports should be left
import numpy as np
from Modeling.SI_Toolkit.TF.TF_Functions.Network import create_rnn_instance


# Take care there is also other GRU implementation in TensorFlow
# However I checked predictor autoregressive uses this one
def step_gru(cell_inputs,
             cell_state,
             kernel,
             recurrent_kernel,
             input_bias,
             recurrent_bias
             ):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_state

    # inputs projected by all gate matrices at once
    matrix_x = K.dot(cell_inputs, kernel)
    matrix_x = K.bias_add(matrix_x, input_bias)

    x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=1)

    # hidden state projected by all gate matrices at once
    matrix_inner = K.dot(h_tm1, recurrent_kernel)
    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)

    recurrent_z, recurrent_r, recurrent_h = array_ops.split(matrix_inner, 3,
                                                            axis=1)
    z = nn.sigmoid(x_z + recurrent_z)
    r = nn.sigmoid(x_r + recurrent_r)
    hh = nn.tanh(x_h + r * recurrent_h)

    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    return h, [h]


def step_dense(inputs, kernel, bias=None, activation=None):
    outputs = gen_math_ops.mat_mul(inputs, kernel)

    if bias is not None:
        outputs = nn_ops.bias_add(outputs, bias)

    if activation is not None:
        outputs = activation(outputs)

    return outputs


class layer_np:
    def __init__(self,
                 name,
                 initial_state=None,
                 kernel=None,
                 recurrent_kernel=None,
                 bias=None,
                 recurrent_bias=None,
                 activation=None
                 ):

        # Todo: fix variable for internal state
        self.name = name

        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.input_bias = bias
        self.recurrent_bias = recurrent_bias

        self.cell_state = initial_state

        self.activation = activation

    def __call__(self, layer_input, *args, **kwargs):
        if 'gru' in self.name:
            output = step_gru(cell_inputs=layer_input,
                              cell_state=self.cell_state,
                              kernel=self.kernel,
                              recurrent_kernel=self.recurrent_kernel,
                              input_bias=self.input_bias,
                              recurrent_bias=self.recurrent_bias
                              )
            self.cell_state = output
            return output
        elif 'dense' in self.name:
            output = step_dense(inputs=layer_input)
            return output
        elif 'rnn' in self.name:
            raise NotImplementedError('Simple RNN Numpy layer not implemented yet')
        elif 'lstm' in self.name:
            raise NotImplementedError('LSTM Numpy layer not implemented yet')


class myNN_np:
    def __init__(self, layers_dicts):

        # It should have
        # self.layers list of layers
        self.layers = []
        for layer_dict in layers_dicts:
            layer = layer_np(**layer_dict)
            self.layers.append(layer)

    def __call__(self, net_input, *args, **kwargs):
        inout = net_input
        for layer in self.layers:
            inout = layer(inout)
        return inout

def get_activation_function(layer_output_name):
    if 'Tanh' in layer_output_name:
        return np.tanh
    else:
        return lambda x: x


from copy import copy
def create_rnn_instance_numpy(net_template):
    layers = net_template.layers
    layers_dicts = []
    layer_dict_template = {}
    dict_keys = ['name', 'kernel', 'recurrent_kernel', 'bias', 'recurrent_bias', 'initial_state', 'activation']
    layer_dict_template = {}
    layer_dict_template.fromkeys(dict_keys)
    for layer in layers:
        print(layer)
        layer_dict = copy(layer_dict_template)
        # Collect info about a layer to the dict
        name = layer.name
        layer_dict['name'] = name

        if 'gru' in name:
            layer_dict['kernel'] = layer.cell.kernel.numpy()
            layer_dict['bias'] = layer.cell.bias.numpy()
            layer_dict['recurrent_kernel'] = layer.cell.recurrent_kernel.numpy()
            layer_dict['recurrent_bias'] = layer.cell.bias.numpy()
            layer_dict['initial_state'] = layer.states[0].numpy()
        elif 'dense' in name:
            layer_dict['kernel'] = layer.kernel.numpy()
            layer_dict['bias'] = layer.bias.numpy()
            layer_dict['activation'] = get_activation_function(layer.output.name)

        layers_dicts.append(layer_dict)
    # get layer info from net_template
    return myNN_np(layers_dicts)


RNN_FULL_NAME = 'GRU-6IN-64H1-64H2-5OUT-0'  # DT = 0.1s for this net
# RNN_PATH = './save_tf/long_3_55/'
RNN_PATH = './save_tf/'
# RNN_PATH = './controllers/nets/mpc_on_rnn_tf/'
PREDICTION_FEATURES_NAMES = ['angle_cos', 'angle_sin', 'angleD', 'position', 'positionD']

if __name__ == '__main__':
    prediction_features_names = PREDICTION_FEATURES_NAMES
    rnn_full_name = RNN_FULL_NAME
    rnn_path = RNN_PATH

    # load rnn
    # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
    net, rnn_name, rnn_inputs_names, rnn_outputs_names, normalization_info \
        = create_rnn_instance(load_rnn=rnn_full_name, path_save=rnn_path,
                              return_sequence=False, stateful=True,
                              wash_out_len=1, batch_size=1)

    net_np = create_rnn_instance_numpy(net_template=net)


