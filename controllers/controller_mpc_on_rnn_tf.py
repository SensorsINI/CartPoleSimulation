"""do-mpc controller"""

import do_mpc

from src.globals import *
from types import SimpleNamespace

from modeling.rnn_tf.utilis_rnn import *
# tf.compat.v1.disable_eager_execution()

from casadi import Callback, Sparsity
import time

from casadi import Callback

RNN_FULL_NAME = 'GRU-6IN-32H1-32H2-4OUT-0'
INPUTS_LIST = ['s.angle', 's.angleD', 's.position', 's.positionD', 'target_position', 'u']
OUTPUTS_LIST = ['s.angle', 's.angleD', 's.position', 's.positionD']
PATH_SAVE = './controllers/nets/mpc_on_rnn_tf/'


# Package the resulting regression model in a CasADi callback
class RNN(Callback):
  def __init__(self, name, function, opts={}):
    Callback.__init__(self)
    self.construct(name, opts)
    self.function = function

  def eval(self, args):
    x = args[0]
    u = args[1]
    x_next = self.function.predict_y(x, u)
    return x_next


class controller_mpc_on_rnn_tf:
    def __init__(self,
                 position_init=0.0,
                 positionD_init=0.0,
                 angle_init=0.0,
                 angleD_init=0.0,
                 ):

        # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
        self.net, self.rnn_name, self.inputs_list, self.outputs_list \
            = create_rnn_instance(load_rnn=RNN_FULL_NAME, path_save=PATH_SAVE,
                                  return_sequence=False, stateful=True,
                                  warm_up_len=1, batchSize=1)

        self.normalization_info = load_normalization_info(PATH_SAVE, RNN_FULL_NAME)

        self.rnn_input = pd.DataFrame(columns=self.inputs_list)
        self.rnn_output = pd.DataFrame(columns=self.outputs_list)

        """
         Get configured do-mpc modules:
        """

        # Physical parameters of the cart
        p = SimpleNamespace()  # p like parameters
        p.m = m_globals  # mass of pend, kg
        p.M = M_globals  # mass of cart, kg
        p.L = L_globals  # half length of pend, m
        p.u_max = u_max_globals  # max cart force, N
        p.M_fric = M_fric_globals  # cart friction, N/m/s
        p.J_fric = J_fric_globals  # friction coefficient on angular velocity, Nm/rad/s
        p.v_max = v_max_globals  # max DC motor speed, m/s, in absence of friction, used for motor back EMF model
        p.controlDisturbance = controlDisturbance_globals  # disturbance, as factor of u_max
        p.sensorNoise = sensorNoise_globals  # noise, as factor of max values
        p.g = g_globals  # gravity, m/s^2
        p.k = k_globals  # Dimensionless factor, for moment of inertia of the pend (with L being half if the length)

        # State of the cart
        s = SimpleNamespace()  # s like state
        s.position = 0.0
        s.positionD = 0.0
        s.angle = 0.0
        s.angleD = 0.0

        model_type = 'discrete'  # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)

        s.position = self.model.set_variable(var_type='_x', var_name='s.position', shape=(1, 1))
        s.positionD = self.model.set_variable(var_type='_x', var_name='s.positionD', shape=(1, 1))

        s.angle = self.model.set_variable(var_type='_x', var_name='s.angle', shape=(1, 1))
        s.angleD = self.model.set_variable(var_type='_x', var_name='s.angleD', shape=(1, 1))

        Q = self.model.set_variable(var_type='_u', var_name='Q')

        target_position = self.model.set_variable('_tvp', 'target_position')

        position_next, positionD_next, angle_next, angleD_next = \
            self.step_rnn(s, Q2u(Q, p))

        self.model.set_rhs('s.position', position_next)
        self.model.set_rhs('s.angle', angle_next)

        self.model.set_rhs('s.positionD', positionD_next)
        self.model.set_rhs('s.angleD', angleD_next)

        # Simplified, normalized expressions for E_kin and E_pot as a port of cost function
        E_kin_cart = (s.positionD / p.v_max) ** 2
        E_kin_pol = (s.angleD / (2 * np.pi)) ** 2
        E_pot = np.cos(s.angle)

        distance_difference = (((s.position - target_position) / 50.0) ** 2)

        self.model.set_expression('E_kin_cart', E_kin_cart)
        self.model.set_expression('E_kin_pol', E_kin_pol)
        self.model.set_expression('E_pot', E_pot)
        self.model.set_expression('distance_difference', distance_difference)

        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': mpc_horizon_globals * 4,
            't_step': dt_mpc_simulation_globals,
            'n_robust': 0,
            'store_full_solution': False,
            'store_lagr_multiplier': False,
            'store_solver_stats': [],
            'state_discretization': 'discrete'
        }
        self.mpc.set_param(**setup_mpc)
        self.mpc.set_param(nlpsol_opts={'ipopt.linear_solver': 'ma27'})

        lterm = (- self.model.aux['E_pot'] + 10.0 * distance_difference + 5 * self.model.aux['E_kin_pol'])
        mterm = (5 * self.model.aux['E_kin_pol'] - 5 * self.model.aux['E_pot'] + 5 * self.model.aux['E_kin_cart'])

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(Q=0.1)

        self.mpc.bounds['lower', '_u', 'Q'] = -1.0
        self.mpc.bounds['upper', '_u', 'Q'] = 1.0

        self.tvp_template = self.mpc.get_tvp_template()

        self.mpc.set_tvp_fun(self.tvp_fun)

        # Suppress IPOPT outputs
        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.mpc.set_param(nlpsol_opts=suppress_ipopt)

        self.mpc.setup()

        # Set initial state
        self.x0 = self.mpc.x0
        self.x0['s.position'] = position_init
        self.x0['s.positionD'] = positionD_init
        self.x0['s.angle'] = angle_init
        self.x0['s.angleD'] = angleD_init

        self.mpc.x0 = self.x0

        self.mpc.set_initial_guess()

    def tvp_fun(self, t_ind):
        return self.tvp_template

    def step_rnn(self, s, u=None, target_position=None):
        # Copy state and target_position into rnn_input

        if 's.position' in self.rnn_input:
            self.rnn_input['s.position'] = [s.position]
        if 's.angle' in self.rnn_input:
            self.rnn_input['s.angle'] = [s.angle]
        if 's.positionD' in self.rnn_input:
            self.rnn_input['s.positionD'] = [s.positionD]
        if 's.angleD' in self.rnn_input:
            self.rnn_input['s.angleD'] = [s.angleD]
        if 'target_position' in self.rnn_input:
            self.rnn_input['target_position'] = [target_position]
        if 'u' in self.rnn_input:
            self.rnn_input['u'] = [u]


        rnn_input_normed = normalize_df(self.rnn_input, self.normalization_info)
        rnn_input_normed = np.squeeze(rnn_input_normed.to_numpy())
        rnn_input_normed = rnn_input_normed[np.newaxis, np.newaxis, :]
        normalized_rnn_output = self.net.predict_on_batch(rnn_input_normed)
        normalized_rnn_output = np.squeeze(normalized_rnn_output).tolist()
        normalized_rnn_output = copy.deepcopy(pd.DataFrame(data=[normalized_rnn_output], columns=self.outputs_list))
        denormalized_rnn_output = denormalize_df(normalized_rnn_output, self.normalization_info)

        position_next = float(denormalized_rnn_output['s.position'])
        positionD_next = float(denormalized_rnn_output['s.positionD'])
        angle_next = float(denormalized_rnn_output['s.angle'])
        angleD_next = float(denormalized_rnn_output['s.angleD'])

        # position_next, positionD_next, angle_next, angleD_next = casadi.SX(0), casadi.SX(0), casadi.SX(0), casadi.SX(0)

        return position_next, positionD_next, angle_next, angleD_next

    def step(self, s, target_position):

        self.x0['s.position'] = s.position
        self.x0['s.positionD'] = s.positionD

        self.x0['s.angle'] = s.angle
        self.x0['s.angleD'] = s.angleD

        self.tvp_template['_tvp', :, 'target_position'] = target_position

        Q = self.mpc.make_step(self.x0)

        return Q.item()
