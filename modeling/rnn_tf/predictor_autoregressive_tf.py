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


import pandas as pd
import numpy as np
import tensorflow as tf
from modeling.rnn_tf.utilis_rnn import *
from src.utilis import pd_plotter_simple


HORIZON = 5
RNN_FULL_NAME = 'GRU-6IN-64H1-64H2-5OUT-0'
RNN_PATH = './save_tf/long_3_55/'
# RNN_PATH = './controllers/nets/mpc_on_rnn_tf/'
PREDICTION_FEATURES_NAMES = ['s.angle.cos', 's.angle.sin', 's.angleD', 's.position', 's.positionD']


class predictor_autoregressive_tf:
    def __init__(self,
                 rnn_full_name=RNN_FULL_NAME,
                 rnn_path=RNN_PATH, horizon=HORIZON,
                 prediction_features_names=PREDICTION_FEATURES_NAMES):

        # load rnn
        # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
        self.net, self.rnn_name, self.rnn_inputs_names, self.rnn_outputs_names, self.normalization_info \
            = create_rnn_instance(load_rnn=rnn_full_name, path_save=rnn_path,
                                  return_sequence=False, stateful=True,
                                  warm_up_len=1, batchSize=1)

        # SAVEPATH = rnn_path + rnn_full_name + '/1/'
        # net_predict = keras.models.load_model(SAVEPATH)
        # net_predict.set_weights(self.net.get_weights())
        #
        # self.net = net_predict

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

        # TODO: Load it from SaveModel TF
    # @tf.function
    def setup(self, initial_state: pd.DataFrame, prediction_denorm=False):

        self.rnn_internal_states = get_internal_states(self.net)
        initial_state_normed = normalize_df(initial_state[self.rnn_inputs_names[1:]], self.normalization_info)
        self.rnn_current_input_without_Q = initial_state_normed.to_numpy(dtype=np.float32).squeeze()
        if prediction_denorm:
            self.prediction_denorm=True
        else:
            self.prediction_denorm = False

    # decorate with tf.function - this must work without retracing within one opt problem!!!
    # @tf.function
    def predict(self, Q) -> pd.DataFrame:
        # print('retracing') # if decorated with tf.function it executes only while retracing
        # Check the input data

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

        self.prediction_rnn.iloc[:, :] = rnn_inout[:self.horizon+1, 0, :].numpy()
        if self.prediction_denorm:
            return denormalize_df(self.prediction_rnn[self.prediction_features_names], self.normalization_info)
        else:
            return self.prediction_rnn[self.prediction_features_names]

    # @tf.function
    def update_internal_rnn_state(self, Q0):

        # load internal RNN state
        load_internal_states(self.net, self.rnn_internal_states)
        self.rnn_current_input[0, 0, 0] = Q0
        self.rnn_current_input[0, 0, 1:] = self.rnn_current_input_without_Q
        self.evaluate_rnn(self.rnn_current_input) # Using tf.function to compile net
        self.net(self.rnn_current_input) # Using net directly

    @tf.function
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

            rnn_inout = rnn_inout.write(i, rnn_output)
            # tf.print(rnn_inout.read(i+1))
        # print(rnn_inout)
        rnn_inout = rnn_inout.stack()
        return rnn_inout

    @tf.function
    def evaluate_rnn_f(self, rnn_input):
        print('retracing evaluate_rnn_f')
        rnn_output = self.net(rnn_input)
        return rnn_output





if __name__ == '__main__':
    import timeit
    import glob
    horizon = 20
    autoregres_at = 60
    downsampling = 1
    start_at = 190
    # data_path = './data/validate/'
    # filename = 'free.csv'
    datafile = glob.glob('./data/validate/' + '*.csv')[0]
    feature_to_plot = 's.angle.cos'
    # df = pd.read_csv(data_path+filename, comment='#')
    df = pd.read_csv(datafile, comment='#')
    df = df.iloc[::downsampling].reset_index()
    df = df.iloc[start_at:].reset_index()
    pd_plotter_simple(df, 'time', feature_to_plot, idx_range=[0, autoregres_at+horizon])
    predictor = predictor_autoregressive_tf(horizon=horizon)
    t0 = timeit.default_timer()
    for row_number in range(autoregres_at):
        initial_state = pd.DataFrame(df.iloc[row_number, :]).transpose()
        Q = float(df.loc[df.index[row_number], 'Q'])
        predictor.setup(initial_state)
        predictor.update_internal_rnn_state(Q)
    t1 = timeit.default_timer()

    initial_state = pd.DataFrame(df.iloc[autoregres_at, 1:]).transpose()
    predictor.setup(initial_state, prediction_denorm=True)
    print('Predictor ready')
    Q = df.loc[df.index[autoregres_at:autoregres_at+horizon], 'Q'].to_numpy(dtype=np.float32).squeeze()
    t2 = timeit.default_timer()
    for i in range(10):
        prediction = predictor.predict(Q)
    t3 = timeit.default_timer()
    pd_plotter_simple(df, x_name='time', y_name=feature_to_plot, idx_range=[autoregres_at, autoregres_at+horizon])
    pd_plotter_simple(prediction, y_name=feature_to_plot, idx_range=[0, horizon], color='red', dt=0.02)

    update_rnn_t = (t1-t0)/autoregres_at
    print('Update RNN {} us/eval'.format(update_rnn_t*1.0e6))
    predictor_t = (t3-t2)/horizon/10.0
    print('Predict {} us/eval'.format(predictor_t*1.0e6))