
import copy

import numpy as np

from types import SimpleNamespace

import tensorflow as tf
import tensorflow.keras as keras

def load_pretrained_net_weights(net, ckpt_path):
    """
    A function loading parameters (weights and biases) from a previous training to a net RNN instance
    :param net: An instance of RNN
    :param ckpt_path: path to .ckpt file storing weights and biases
    :return: No return. Modifies net in place.
    """
    print("Loading Model: ", ckpt_path)
    print('')

    net.load_weights(ckpt_path).expect_partial()


def compose_net_from_net_name(net_name,
                              inputs_list,
                              outputs_list,
                              time_series_length,
                              batch_size=None,
                              stateful=False):

    # Get the information about network architecture from the network name
    # Split the names into "LSTM/GRU", "128H1", "64H2" etc.
    names = net_name.split('-')
    layers = ['H1', 'H2', 'H3', 'H4', 'H5']
    h_size = []  # Hidden layers sizes
    for name in names:
        for index, layer in enumerate(layers):
            if layer in name:
                # assign the variable with name obtained from list layers.
                h_size.append(int(name[:-2]))

    if not h_size:
        raise ValueError('You have to provide the size of at least one hidden layer in rnn name')

    h_number = len(h_size)

    if 'GRU' in names:
        net_type = 'GRU'
        layer_type = keras.layers.GRU
    elif 'LSTM' in names:
        net_type = 'LSTM'
        layer_type = keras.layers.LSTM
    elif 'Dense' in names:
        net_type = 'Dense'
        layer_type = keras.layers.Dense
    else:
        net_type = 'RNN-Basic'
        layer_type = keras.layers.SimpleRNN

    net = tf.keras.Sequential()

    # Construct network
    # Either dense...
    if net_type == 'Dense':
        net.add(tf.keras.Input(shape=(time_series_length, len(inputs_list))))
        for i in range(h_number - 1):
            net.add(layer_type(
                units=h_size[i], activation='tanh', batch_size=batch_size
            ))
    else:
        # Or RNN...
        net.add(layer_type(
            units=h_size[0],
            batch_input_shape=(batch_size, time_series_length, len(inputs_list)),
            return_sequences=True,
            stateful=stateful
        ))
        # Define following layers
        for i in range(1, len(h_size)):
            net.add(layer_type(
                units=h_size[i],
                return_sequences=True,
                stateful=stateful
            ))

    # net.add(keras.layers.Dense(units=len(outputs_list), activation='tanh'))
    net.add(keras.layers.Dense(units=len(outputs_list)))

    print('Constructed a neural network of type {}, with {} hidden layers with sizes {} respectively.'
          .format(net_type, len(h_size), ', '.join(map(str, h_size))))

    # Compose net_info
    net_info = SimpleNamespace()
    net_info.net_name = net_name
    net_info.inputs = inputs_list
    net_info.outputs = outputs_list
    net_info.net_type = net_type

    return net, net_info


def get_internal_states(net):
    states_list = []
    for layer in net.layers:
        if (('gru' in layer.name) or
                ('lstm' in layer.name) or
                ('rnn' in layer.name)):
            single_states = []
            for single_state in layer.states:
                captured_single_state = copy.deepcopy(single_state).numpy()
                single_states.append(captured_single_state)

            states_list.append(single_states)
        else:
            states_list.append(None)
    return states_list


from tensorflow.python.keras import backend as K
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape


def my_reset_states(layer, states=None):
    spec_shape = None

    if layer.input_spec is not None:
        spec_shape = nest.flatten(layer.input_spec[0])[0].shape

    if spec_shape is None:
        batch_size = None
    else:
        batch_size = spec_shape[1] if layer.time_major else spec_shape[0]

    if states is None:
        for state, size in zip(nest.flatten(layer.states),
                               nest.flatten(layer.cell.state_size)):
            K.set_value(state, np.zeros([batch_size] +
                                        tensor_shape.as_shape(size).as_list()))
    else:
        flat_states = nest.flatten(layer.states)
        flat_input_states = nest.flatten(states)
        set_value_tuples = []
        for i, (value, state) in enumerate(zip(flat_input_states,
                                               flat_states)):
            set_value_tuples.append((state, value))
        K.batch_set_value(set_value_tuples)


def load_internal_states(net, states):
    for layer, state in zip(net.layers, states):
        if (('gru' in layer.name) or
                ('lstm' in layer.name) or
                ('rnn' in layer.name)):
            layer.reset_states(state[0])
            # my_reset_states(layer, state[0])
