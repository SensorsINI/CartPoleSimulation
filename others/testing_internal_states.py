import tensorflow as tf
from SI_Toolkit.Predictors.predictor_autoregressive_neural import \
    predictor_autoregressive_neural
from SI_Toolkit.Functions.TF.Compile import CompileTF

from others.globals_and_utils import create_rng


@CompileTF
def copy_internal_states_to_ref(net, memory_states_ref):
    for layer, layer_ref in zip(net.layers, memory_states_ref):
        if (('gru' in layer.name) or
                ('lstm' in layer.name) or
                ('rnn' in layer.name)):

            for single_state, single_state_ref in zip(layer.states, layer_ref.states):
                single_state_ref.assign(tf.convert_to_tensor(single_state))
        else:
            pass


@CompileTF
def copy_internal_states_from_ref(net, memory_states_ref):
    for layer, layer_ref in zip(net.layers, memory_states_ref):
        if (('gru' in layer.name) or
                ('lstm' in layer.name) or
                ('rnn' in layer.name)):

            for single_state, single_state_ref in zip(layer.states, layer_ref.states):
                single_state.assign(tf.convert_to_tensor(single_state_ref))
        else:
            pass

if __name__ == '__main__':
    import numpy as np
    from SI_Toolkit_ASF.predictors_customization import CONTROL_INPUTS

    batch_size = 8
    horizon = 10

    predictor = predictor_autoregressive_neural(horizon=horizon, batch_size=batch_size,
                                                net_name='LSTM-6IN-8H1-4H2-16H3-5OUT-0')

    rng = create_rng(__name__, None)
    initial_state = rng.random(size=(batch_size, 6))
    # initial_state = rng.random(size=(1, 6))
    Q = np.float32(rng.random(size=(batch_size, horizon, len(CONTROL_INPUTS))))

    net = predictor.net
    memory_states_ref = predictor.memory_states_ref
    _r = memory_states_ref[1].states
    _o = net.layers[1].states
    copy_internal_states_from_ref(net, memory_states_ref)

    predictor.predict(initial_state, Q)

    pass  # Check layers in both

    copy_internal_states_from_ref(net, memory_states_ref)

    pass  # Check if both are zero

    predictor.predict(initial_state, Q)

    pass  # check if ref is still 0

    copy_internal_states_to_ref(net, memory_states_ref)

    pass # Check if both are equal

    predictor.predict(initial_state, Q)

    pass  # Check if they are different and nonzero
