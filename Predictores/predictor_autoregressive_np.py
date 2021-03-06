"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""


#TODO: for the moment it is not possible to update RNN more often than mpc dt
#   Updating it more often will lead to false results.


from Modeling.TF.TF_Functions.Network import *
import numpy as np
from Modeling.TF.TF_Functions.TF2Numpy import create_rnn_instance_numpy


RNN_FULL_NAME = 'GRU-6IN-64H1-64H2-5OUT-0' # DT = 0.1s for this net
# RNN_PATH = './save_tf/long_3_55/'
RNN_PATH = './save_tf/'
# RNN_PATH = './controllers/nets/mpc_on_rnn_tf/'
PREDICTION_FEATURES_NAMES = ['angle_cos', 'angle_sin', 'angleD', 'position', 'positionD']

class predictor_autoregressive_tf:
    def __init__(self, horizon, dt=None):

        prediction_features_names = PREDICTION_FEATURES_NAMES
        rnn_full_name = RNN_FULL_NAME
        rnn_path = RNN_PATH

        # load rnn
        # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
        net, self.rnn_name, self.rnn_inputs_names, self.rnn_outputs_names, self.normalization_info \
            = create_rnn_instance(load_rnn=rnn_full_name, path_save=rnn_path,
                                  return_sequence=False, stateful=True,
                                  wash_out_len=1, batch_size=1)

        # TODO: Load it from SaveModel TF
        # SAVEPATH = rnn_path + rnn_full_name + '/1/'
        # net_predict = keras.models.load_model(SAVEPATH)
        # net_predict.set_weights(self.net.get_weights())
        #
        # self.net = net_predict

        self.net = create_rnn_instance_numpy(net)
        del net

        self.rnn_internal_states = get_internal_states(self.net)

        print(self.net.summary())

        self.horizon = horizon
        # self.rnn_inout = np.zeros([horizon+1, 1, 1, len(self.rnn_inputs_names)], dtype=np.float32)

        self.rnn_current_input_without_Q = np.zeros([len(self.rnn_inputs_names)-1], dtype=np.float32)
        self.rnn_current_input = np.zeros([1, 1, len(self.rnn_inputs_names)], dtype=np.float32)

        self.prediction_rnn = pd.DataFrame(columns=self.rnn_outputs_names, index=range(horizon + 1))

        self.prediction_features_names = prediction_features_names
        self.prediction_denorm = False
        Q_type = tf.TensorSpec((self.horizon,), tf.float32)
        initial_input_type = tf.TensorSpec((len(self.rnn_inputs_names)-1,), tf.float32)

        rnn_input_type = tf.TensorSpec((1, 1, len(self.rnn_inputs_names)), tf.float32)

        # Retracing tensorflow functions
        try:
            self.evaluate_rnn = self.evaluate_rnn_f.get_concrete_function(rnn_input=rnn_input_type)
        except:
            self.evaluate_rnn = self.evaluate_rnn_f

        try:
            self.iterate_rnn = self.iterate_rnn_f.get_concrete_function(Q=Q_type,
                                                                        initial_input=initial_input_type)
            print(self.iterate_rnn)
        except:
            self.iterate_rnn = self.iterate_rnn_f

    def setup(self, initial_state: pd.DataFrame, prediction_denorm=False):

        self.rnn_internal_states = get_internal_states(self.net)
        initial_state_normed = normalize_df(copy.copy(initial_state[self.rnn_inputs_names[1:]]), self.normalization_info)
        self.rnn_current_input_without_Q = initial_state_normed.to_numpy(dtype=np.float32, copy=True).squeeze()
        if prediction_denorm:
            self.prediction_denorm=True
        else:
            self.prediction_denorm = False

    def predict(self, Q) -> pd.DataFrame:

        # load internal RNN state

        if len(Q) != self.horizon:
            raise IndexError('Number of provided control inputs does not match the horizon')
        else:
            Q = tf.squeeze(tf.convert_to_tensor(Q, dtype=tf.float32))

        # don't forget to reload internal state of RNN

        load_internal_states(self.net, self.rnn_internal_states)
        initial_input = tf.convert_to_tensor(self.rnn_current_input_without_Q, tf.float32)
        # t0 = timeit.default_timer()
        rnn_inout = self.iterate_rnn(Q, initial_input)
        # t1 = timeit.default_timer()
        # iterate_t = (t1-t0)/self.horizon
        # print('Iterate {} us/eval'.format(iterate_t * 1.0e6))
        # compose the pandas output DF
        # Later: if necessary add sin, cos, derivatives
        # First version let us assume net returns all state except for angle
        rnn_inout_np = rnn_inout.numpy()
        rnn_inout = rnn_inout[:self.horizon+1, 0, :].numpy()
        self.prediction_rnn.iloc[:, :] = rnn_inout

        if self.prediction_denorm:
            predictions = copy.copy(denormalize_df(self.prediction_rnn[self.prediction_features_names], self.normalization_info))
            predictions['angle'] = np.arctan2(predictions['angle_sin'], predictions['angle_cos'])
        else:
            predictions = copy.copy(self.prediction_rnn[self.prediction_features_names])
            predictions['angle'] = np.arctan2(predictions['angle_sin'], predictions['angle_cos'])/np.pi

        return predictions

    # @tf.function
    def update_internal_state(self, Q0):

        # load internal RNN state
        load_internal_states(self.net, self.rnn_internal_states)
        self.rnn_current_input[0, 0, 0] = Q0
        self.rnn_current_input[0, 0, 1:] = self.rnn_current_input_without_Q
        # self.evaluate_rnn(self.rnn_current_input) # Using tf.function to compile net
        self.net(self.rnn_current_input) # Using net directly

    # @tf.function
    def iterate_rnn_f(self, Q, initial_input):
        print('retracing iterate_rnn_f')
        # Iterate over RNN -
        # rnn_input = tf.zeros(shape=(1, 1, len(self.rnn_inputs_names),), dtype=tf.float32)
        rnn_output = tf.zeros(shape=(1,len(self.rnn_outputs_names)), dtype=tf.float32)
        rnn_inout = tf.TensorArray(tf.float32, size=self.horizon + 1)
        # Q_current = tf.zeros(shape=(1,), dtype=tf.float32)

        rnn_inout = rnn_inout.write(0, tf.reshape(initial_input, [1, len(initial_input)]))
        for i in tf.range(0, self.horizon):
            Q_current = (tf.reshape(Q[i], [1]))
            if i == 0:
                rnn_input = (tf.reshape(tf.concat([Q_current, initial_input], axis=0), [1, 1, len(self.rnn_inputs_names)]))
            else:
                rnn_input = tf.reshape(tf.concat([Q_current, tf.squeeze(rnn_output)], axis=0), [1, 1, len(self.rnn_inputs_names)])
            # rnn_output = (self.net(rnn_input))
            rnn_output = (self.evaluate_rnn(rnn_input))
            #tf.print(rnn_output)

            rnn_inout = rnn_inout.write(i+1, rnn_output)
            # tf.print(rnn_inout.read(i+1))
        # print(rnn_inout)
        rnn_inout = rnn_inout.stack()
        return rnn_inout

    # @tf.function
    def evaluate_rnn_f(self, rnn_input):
        print('retracing evaluate_rnn_f')
        rnn_output = self.net(rnn_input)
        return rnn_output