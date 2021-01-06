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

import tensorflow as tf
import pandas as pd

HORIZON = 5


class predictor_autoregressive_tf:
    def __init__(self, rnn_full_name, horizon):

        self.net = ...

        self.horizon = horizon
        self.rnn_inputs_names = ...
        self.rnn_inout = tf.zeros([horizon+1, 1, 1, len(self.rnn_inputs_names)], dtype = tf.float32)

        self.rnn_current_input_without_Q = None
        self.rnn_current_input = tf.zeros([1, 1, len(self.rnn_inputs_names)], dtype = tf.float32)

        self.prediction_rnn = pd.DataFrame(columns=self.rnn_inputs_names, index=range(horizon + 1))


        pass
        # Load RNN, either from checkpoints, or from saved model

    def setup(self, initial_state: pd.DataFrame, horizon=None, dt_prediction=None):

        self.rnn_current_input_without_Q = tf.squeeze( tf.convert_to_tensor(initial_state[self.rnn_inputs_names]) )

    # decorate with tf.function - this must work without retracing within one opt problem!!!
    def predict(self, Q, horizon=None, dt_prediction=None) -> pd.DataFrame:
        # print('retracing') # if decorated with tf.function it executes only while retracing
        # Check the input data

        # load internal RNN state

        if len(Q) != horizon:
            raise IndexError('Number of provided control inputs does not match the horizon')
        else:
            Q = tf.convert_to_tensor(Q, dtype=tf.float32)

        self.rnn_inout[:-1, 0, 0, 0] = Q
        self.rnn_inout[0, 0, 0, 1:] = self.rnn_current_input_without_Q

        # don't forget to reload internal state of RNN

        self.load_rnn_internal_state(self.rnn_internal_state)

        # Iterate over RNN -
        for i in range(len(horizon)):
            self.rnn_inout[i + 1, 0, :, :] = self.net.predict_on_batch(self.rnn_inout[i, :, :, :])

        # compose the pandas output DF
        # Later: if necessary add sin, cos, derivatives
        # First version let us assume net returns all state except for angle

        self.prediction.iloc[:,:] = self.rnn_inout[:self.horizon+1, 0, 0, 1:]

        return self.prediction[initial_state.columns]


    def update_internal_rnn_state(self, Q0):

        # load internal RNN state
        self.load_rnn_internal_state(self.rnn_internal_state)
        self.rnn_current_input[0, 0, 0] = Q0
        self.rnn_current_input[0, 0, 1:] = self.rnn_current_input_without_Q
        self.net.predict_on_batch(self.rnn_current_input)
        self.rnn_internal_state = self.get_rnn_internal_state()



    def system_state_2_rnn_input(self, system_state):

        return tf.squeeze(tf.convert_to_tensor(   system_state[self.rnn_inputs_names],   dtype=tf.float32)   )

    # decorate
    def iterate_rnn(self, rnn_inout, horizon):
        # You should make sure that this function retrace only if horizon changes





