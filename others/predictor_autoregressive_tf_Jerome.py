# "Command line" parameters
from SI_Toolkit.TF.TF_Functions.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.TF.TF_Functions.Compile import Compile
from SI_Toolkit.load_and_normalize import *
from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, CONTROL_INPUTS, augment_predictor_output
from types import SimpleNamespace
import yaml
import os
import tensorflow as tf

class predictor_autoregressive_tf:
    def __init__(self, horizon=None, batch_size=None, net_name=None):
        self.net_name = net_name
        self.net_type = net_name
        self.batch_size = batch_size
        self.horizon = horizon

        self.initial_state = None
        self.initial_state_tf = None
        self.prev_initial_state_tf = None

        # Neural Network
        config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config_testing.yml'), 'r'), Loader=yaml.FullLoader)
        if '/' in net_name:
            self.model_path = net_name.rsplit('/', 1)[0]
            self.net_name = self.net_name.split("/")[-1]
        else:
            self.model_path = config["testing"]["PATH_TO_NN"]
            self.net_name = self.net_name

        self.net_type = self.net_name.split("-")[0]

        print(self.model_path, self.net_name, self.net_type)

        a = SimpleNamespace()
        a.path_to_models = self.model_path
        a.net_name = self.net_name
        self.net, self.net_info = get_net(a, time_series_length=1, batch_size=batch_size, stateful=True)
        self.normalization_info = get_norm_info_for_net(self.net_info)[self.net_info.outputs]

        # Network sizes
        self.net_input_length = len(self.net_info.inputs)
        self.control_length = len(CONTROL_INPUTS)
        self.state_length = self.net_input_length - self.control_length

        # Helpers
        self.state_indices_list = [STATE_INDICES.get(key) for key in self.net_info.outputs]

        # De/Normalization:
        # normalized = 2 * (denormalize - min) / (max-min) - 1 = denormalized * 2 / (max-min) - 2 * min / (max-min) - 1 = denormalized * 2 / (max-min) -
        # denormalized = (normalized + 1) / 2 *  (max-min) + min = normalized * (max-min) / 2 + (max-min) / 2 + min = normalized * (max-min) / 2 + (max+min) / 2
        min = tf.convert_to_tensor(self.normalization_info.loc['min'].to_numpy(), dtype=tf.float32)
        max = tf.convert_to_tensor(self.normalization_info.loc['max'].to_numpy(), dtype=tf.float32)
        self.normalization_offset = -2 * tf.ones(shape=(self.batch_size, self.state_length)) * tf.math.divide(min, max-min) - tf.ones(shape=(self.batch_size, self.state_length))
        self.normalization_scale = 2 * tf.ones(shape=(self.batch_size, self.state_length)) * tf.math.reciprocal(max-min)
        self.denormalization_offset = tf.ones(shape=(self.batch_size, self.horizon, self.state_length)) * (max+min) / 2
        self.denormalization_scale = tf.ones(shape=(self.batch_size, self.horizon, self.state_length)) * (max-min) / 2


        # RNN States
        if self.net_type == 'GRU':
            self.gru_layers = [layer for layer in self.net.layers if (('gru' in layer.name) or ('lstm' in layer.name) or ('rnn' in layer.name))]
            self.num_gru_layers = len(self.gru_layers)
            self.rnn_internal_states = [tf.zeros(shape=(self.batch_size, layer.units)) for layer in self.gru_layers]
        self.Q_prev = tf.zeros(shape=(self.batch_size, 1, len(CONTROL_INPUTS)), dtype=tf.float32)


    # TODO: replace everywhere with predict_tf
    # DEPRECATED: This version is in-efficient since it copies all batches to GPU
    def predict(self, initial_state, Q):
        # print('************\n************\n************\n************\n************\n************\n')
        # Predict TF

        self.initial_state = initial_state

        net_output = self.predict_tf(tf.convert_to_tensor(self.initial_state[0,...], dtype=tf.float32), tf.convert_to_tensor(Q, dtype=tf.float32))

        # Prepare Deprecated Output
        output_array = np.zeros([self.batch_size, self.horizon + 1, len(STATE_VARIABLES)], dtype=np.float32)
        output_array[:, 0, :] = self.initial_state
        output_array[:, 1:, :] = net_output.numpy()

        return output_array

    # Predict (Euler: 6.8ms, RNN:10.5ms)
    @Compile
    def predict_tf(self, initial_state, Q):
        # assert tf.rank(Q) == 3
        # Select States

        initial_state = tf.gather(initial_state, self.state_indices_list, axis=0)

        self.initial_state_tf = tf.tile(tf.expand_dims(initial_state, axis=0), [self.batch_size, 1])

        # Normalization
        self.initial_state_tf = self.normalization_offset + tf.math.multiply(self.normalization_scale, self.initial_state_tf)

        # Run Last Input for RNN States
        if self.net_type == 'GRU' and self.prev_initial_state_tf is not None:
            # Set RNN States
            for j in range(self.num_gru_layers):
                self.gru_layers[j].states[0].assign(self.rnn_internal_states[j])

            # Apply Last Input
            Q0 = self.Q_prev
            # assert tf.shape(Q0)[1] == 1 # Just one timestep
            if tf.shape(Q0)[0] != self.batch_size:
                # assert tf.shape(Q0)[0] == 1 # Batch size == 1
                Q0 = tf.tile(Q0, tf.constant([self.batch_size, 1, 1]))
            Q0 = tf.squeeze(Q0, axis=1)

            net_input = tf.concat([Q0, self.initial_state_tf], axis=1)
            self.net(tf.expand_dims(net_input, axis=1))

            # Get RNN States
            for j in range(self.num_gru_layers):
                self.rnn_internal_states[j] = tf.identity(self.gru_layers[j].states[0])

        # Run Iterations (Euler:ms, RNN:ms)
        net_outputs = self.iterate_net(initial_state=self.initial_state_tf, Q=Q)
        self.prev_initial_state_tf = tf.identity(self.initial_state_tf)

        net_outputs = self.denormalization_offset + tf.math.multiply(self.denormalization_scale, net_outputs)

        # Network Output: angleD, angle_cos, angle_sin, position, positionD
        # Return: angle, angleD, angle_cos, angle_sin, position, positionD
        net_outputs = tf.stack([tf.math.atan2(net_outputs[...,2], net_outputs[...,1]), net_outputs[...,0], net_outputs[...,1], net_outputs[...,2], net_outputs[...,3], net_outputs[...,4]], axis=2)

        return net_outputs

    def update_internal_state_tf(self, Q0):
        self.Q_prev = Q0

    # TODO: replace everywhere with update_internal_state_tf
    # DEPRECATED: This version is in-efficient since it copies all batches to GPU
    def update_internal_state(self, Q, s=None):

        if tf.is_tensor(Q):
            self.update_internal_state_tf(tf.convert_to_tensor(Q[0,...], dtype=tf.float32))

    @Compile
    def iterate_net(self, Q, initial_state):

        net_output = tf.zeros(shape=(self.batch_size, self.state_length), dtype=tf.float32)
        net_outputs = tf.TensorArray(tf.float32, size=self.horizon, dynamic_size=False)

        for i in tf.range(self.horizon):
            Q_current = Q[..., i, :]

            if i == 0:
                net_input = tf.concat([Q_current, initial_state], axis=1)
            else:
                net_input = tf.concat([Q_current, net_output], axis=1)

            net_output = self.net(tf.expand_dims(net_input, axis=1))
            net_output = tf.squeeze(net_output, axis=1)

            net_outputs = net_outputs.write(i, net_output)

        # Stacking
        output = tf.transpose(net_outputs.stack(), perm=[1, 0, 2])

        return output


if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_autoregressive_tf_Jerome import predictor_autoregressive_tf
predictor = predictor_autoregressive_tf(horizon, batch_size=batch_size, net_name='GRU-6IN-32H1-32H2-5OUT-0')
'''

    timer_predictor(initialisation)
