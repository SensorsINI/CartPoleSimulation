"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""



"""
Using predictor:
1. Initialize while initializing controller
    This step load the RNN (if applies) - it make take quite a bit of time
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
    ad c) this method updates the internal state of RNN (if applies). It accepts control input for current time step (scalar) as its only input
            it should be called only after the optimization problem is solved with the control input used in simulation
            
"""

#TODO: for the moment it is not possible to update RNN more often than mpc dt
#   Updating it more often will lead to false results.


from modeling.rnn_tf.utilis_rnn import *
from src.utilis import pd_plotter_simple
from src.globals import *
from copy import deepcopy

import matplotlib.pyplot as plt


HORIZON = 5
RNN_FULL_NAME = 'GRU-6IN-64H1-64H2-5OUT-0' # You need it to get normalization info
RNN_PATH = './save_tf/long_3_55/'
# RNN_PATH = './controllers/nets/mpc_on_rnn_tf/'
PREDICTION_FEATURES_NAMES = ['s.angle.cos', 's.angle.sin', 's.angleD', 's.position', 's.positionD']
downsampling = 1
DT = dt_main_simulation_globals*downsampling


class predictor_ideal:
    def __init__(self, horizon=HORIZON,
                 prediction_features_names=PREDICTION_FEATURES_NAMES):

        self.normalization_info = load_normalization_info(RNN_PATH, RNN_FULL_NAME)

        # Physical parameters of the cart
        self.p = SimpleNamespace()  # p like parameters
        self.p.m = m_globals  # mass of pend, kg
        self.p.M = M_globals  # mass of cart, kg
        self.p.L = L_globals  # half length of pend, m
        self.p.u_max = u_max_globals  # max cart force, N
        self.p.M_fric = M_fric_globals  # cart friction, N/m/s
        self.p.J_fric = J_fric_globals  # friction coefficient on angular velocity, Nm/rad/s
        self.p.v_max = v_max_globals  # max DC motor speed, m/s, in absence of friction, used for motor back EMF model
        self.p.controlDisturbance = controlDisturbance_globals  # disturbance, as factor of u_max
        self.p.sensorNoise = sensorNoise_globals  # noise, as factor of max values
        self.p.g = g_globals  # gravity, m/s^2
        self.p.k = k_globals  # Dimensionless factor, for moment of inertia of the pend (with L being half if the length)

        # State of the cart
        self.s = SimpleNamespace()  # s like state


        self.target_position = 0.0
        self.target_position_normed = 0.0

        self.horizon = horizon

        self.dt = DT

        self.prediction_features_names = prediction_features_names
        self.prediction_denorm = False

        self.prediction_list = pd.DataFrame(columns=PREDICTION_FEATURES_NAMES, index=range(horizon + 1))


    def setup(self, initial_state: pd.DataFrame, prediction_denorm=False):

        self.s.angle = initial_state['s.angle'].to_numpy().squeeze()
        self.s.angle_cos = np.cos(initial_state['s.angle']).to_numpy().squeeze()
        self.s.angle_sin = np.sin(initial_state['s.angle']).to_numpy().squeeze()
        self.s.angleD = initial_state['s.angleD'].to_numpy().squeeze()

        self.s.position = initial_state['s.position'].to_numpy().squeeze()
        self.s.positionD = initial_state['s.positionD'].to_numpy().squeeze()

        if prediction_denorm:
            self.prediction_denorm=True
        else:
            self.prediction_denorm = False


    def predict(self, Q) -> pd.DataFrame:

        if len(Q) != self.horizon:
            raise IndexError('Number of provided control inputs does not match the horizon')
        else:
            Q_hat = np.asarray(Q).squeeze()

        # t0 = timeit.default_timer()
        yp_hat = np.zeros(self.horizon + 1, dtype=object)

        for k in range(self.horizon):
            if k == 0:
                yp_hat[0] = deepcopy(self.s)
                s_next = deepcopy(self.s)

            t0 = timeit.default_timer()
            s_next = mpc_next_state(s_next, self.p, Q2u(Q_hat[k], self.p), dt=self.dt)
            s_next.angle_cos = np.cos(s_next.angle)
            s_next.angle_sin = np.sin(s_next.angle)
            t1 = timeit.default_timer()
            # self.eq_eval_time.append((t1 - t0) * 1.0e6)
            yp_hat[k + 1] = s_next

        all_features = []
        for k in range(len(yp_hat)):
            s = yp_hat[k]
            if k<horizon:
                Q = Q_hat[k]
            else:
                Q = Q_hat[k-1]
            timestep_features = [Q, s.angle_cos, s.angle_sin, s.angleD, s.position, s.positionD]
            all_features.append(timestep_features)
        all_features = np.asarray(all_features)
        self.prediction_list = pd.DataFrame(data=all_features, columns=['Q']+PREDICTION_FEATURES_NAMES)
        # self.prediction_list = normalize_df(self.prediction_list, self.normalization_info)

        if self.prediction_denorm:
            pass
            # return denormalize_df(self.prediction_list[PREDICTION_FEATURES_NAMES], self.normalization_info)
        else:
            return self.prediction_list[['Q']+PREDICTION_FEATURES_NAMES]

    # @tf.function
    def update_internal_rnn_state(self, Q0):
        pass



if __name__ == '__main__':
    import timeit
    import glob
    import matplotlib as mpl
    # mpl.use('macosx')
    horizon = 100//downsampling
    autoregres_at = 0
    start_at = 200
    # data_path = './data/validate/'
    # filename = 'free.csv'
    datafile = glob.glob('./data/validate/' + '*.csv')[0]
    feature_to_plot = 's.angle.cos'
    # df = pd.read_csv(data_path+filename, comment='#')
    df = pd.read_csv(datafile, comment='#')
    df = df.iloc[::downsampling].reset_index(drop=True)
    df = df.iloc[start_at:].reset_index(drop=True)
    df = df.iloc[0:autoregres_at+horizon+1, df.columns.get_indexer(['time', 'Q', 's.angle', 's.angle.cos', 's.angle.sin', 's.angleD', 's.position', 's.positionD'])]
    # pd_plotter_simple(df, 'time', feature_to_plot, idx_range=[0, autoregres_at+horizon+1])
    predictor = predictor_ideal(horizon=horizon)
    t0 = timeit.default_timer()
    # for row_number in range(autoregres_at):
    #     initial_state = pd.DataFrame(df.iloc[row_number, :]).transpose()
    #     initial_state = initial_state[['s.angle', 's.angleD', 's.position', 's.positionD']]
    #     Q = float(df.loc[df.index[row_number], 'Q'])
    #     predictor.setup(initial_state)
    #     predictor.update_internal_rnn_state(Q)
    t1 = timeit.default_timer()

    initial_state = pd.DataFrame(deepcopy(df.iloc[autoregres_at, :])).transpose()
    print(initial_state)
    print(df.iloc[autoregres_at, :])
    predictor.setup(initial_state, prediction_denorm=False)
    Q = df.loc[df.index[autoregres_at:autoregres_at+horizon], 'Q'].to_numpy(dtype=np.float32).squeeze()
    print(Q)
    t2 = timeit.default_timer()
    # for i in range(10):
    #     prediction = predictor.predict(Q)
    prediction = predictor.predict(Q)
    t3 = timeit.default_timer()
    fig1 = pd_plotter_simple(df, y_name=feature_to_plot, idx_range=[autoregres_at, autoregres_at+horizon+1], dt=DT*downsampling) # , x_name='time'
    fig2 = pd_plotter_simple(prediction, y_name=feature_to_plot, idx_range=[0, horizon+1], color='red', dt=DT*downsampling)

    target = df[feature_to_plot].to_numpy().squeeze()[autoregres_at : autoregres_at+horizon+1]
    prediction_single = prediction[feature_to_plot].to_numpy().squeeze()
    fig3 = plt.figure()
    plt.plot(target-prediction_single)
    # plt.ion()
    # plt.show(block=False)
    # input("Press [enter] to continue.")
    # plt.show()
    # update_rnn_t = (t1-t0)/autoregres_at
    # print('Update RNN {} us/eval'.format(update_rnn_t*1.0e6))
    # predictor_t = (t3-t2)/horizon/10.0
    # print('Predict {} us/eval'.format(predictor_t*1.0e6))