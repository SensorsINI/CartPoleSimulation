"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""

"""
This is a predictor for autoregressive RNNs constructed in tensorflowrol
This predictor is good only for one control input being first rnn input, all other rnn inputs in the same order
as rnn outputs, and all rnn outputs being closed loop, no dt, no target position
horizon cannot be changed in runtime
"""


"""
Using predictor:
1. Initialize while initializing controller
    This step load the RNN - it make take quite a bit of time
    During initialization you only need to provide RNN which should be loaded
2. Call iterativelly three functions
    a) setup(initial_state, horizon, etc.)
    b) predict(Q)
    c) update_rnn
    
    ad a) at this stage you can change the parameters for prediction like e.g. horizon, dt
            It also prepares 0 state of the prediction, and tensors for saving the results,
            to make b) max performance. This function should be called BEFORE starting solving an optim
    ad b) predict is optimized to get the prediction of future states of the system as fast as possible.
        It accepts control input (vector) as its only input and is intended to be used at every evaluation of the cost functiomn
    ad c) this method updates the internal state of RNN. It accepts control input for current time step (scalar) as its only input
            it should be called only after the optimization problem is solved with the control input used in simulation
            
"""

#TODO: for the moment it is not possible to update RNN more often than mpc dt
#   Updating it more often will lead to false results.


from Modeling.TF.TF_Functions.Network import *
from Modeling.load_and_normalize import load_normalization_info, normalize_df, denormalize_df

import numpy as np
import pandas as pd

RNN_FULL_NAME = 'GRU-6IN-64H1-64H2-5OUT-0'  # DT = 0.1s for this net
NET_PATH = './Modeling/TF/Models/' + RNN_FULL_NAME
# RNN_PATH = './controllers/nets/mpc_on_rnn_tf/'
PREDICTION_FEATURES_NAMES = ['s.angle.cos', 's.angle.sin', 's.angleD', 's.position', 's.positionD']

# TODO: This should be taken from log of net:
inputs_list = ['Q', 's.angle.sin', 's.angle.cos', 's.angleD', 's.position', 's.positionD']
outputs_list = ['s.angle.sin', 's.angle.cos', 's.angleD', 's.position', 's.positionD']
PATH_TO_NORMALIZATION_INFO = './Modeling/NormalizationInfo/' + 'NI_2021-03-01_11-51-13.csv'



class predictor_autoregressive_tf:
    def __init__(self, horizon, dt=None):

        prediction_features_names = PREDICTION_FEATURES_NAMES

        self.net = keras.models.load_model(NET_PATH)

        self.rnn_inputs_names = inputs_list
        self.rnn_outputs_names = outputs_list

        self.normalization_info = load_normalization_info(PATH_TO_NORMALIZATION_INFO)

        self.rnn_internal_states = get_internal_states(self.net)

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
            predictions['s.angle'] = np.arctan2(predictions['s.angle.sin'], predictions['s.angle.cos'])
        else:
            predictions = copy.copy(self.prediction_rnn[self.prediction_features_names])
            predictions['s.angle'] = np.arctan2(predictions['s.angle.sin'], predictions['s.angle.cos'])/np.pi

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