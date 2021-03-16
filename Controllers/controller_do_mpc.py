"""do-mpc controller"""

import do_mpc
import numpy as np

from Controllers.template_controller import template_controller
from CartPole.cartpole_model import P_GLOBALS, s0, cartpole_ode_namespace, Q2u
from CartPole._CartPole_mathematical_helpers import create_cartpole_state, cartpole_state_varname_to_index, cartpole_state_namespace_to_vector, cartpole_state_vector_to_namespace

from types import SimpleNamespace

dt_mpc_simulation = 0.2  # s
mpc_horizon = 10

class controller_do_mpc(template_controller):
    def __init__(self,
                 position_init=0.0,
                 positionD_init=0.0,
                 angle_init=0.0,
                 angleD_init=0.0,
                 ):

        """
        Get configured do-mpc modules:
        """

        # Container for the state of the cart
        s = SimpleNamespace()  # s like state

        model_type = 'continuous'  # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)

        s.position = self.model.set_variable(var_type='_x', var_name='s.position', shape=(1, 1))
        s.positionD = self.model.set_variable(var_type='_x', var_name='s.positionD', shape=(1, 1))

        s.angle = self.model.set_variable(var_type='_x', var_name='s.angle', shape=(1, 1))
        s.angleD = self.model.set_variable(var_type='_x', var_name='s.angleD', shape=(1, 1))

        Q = self.model.set_variable(var_type='_u', var_name='Q')

        target_position = self.model.set_variable('_tvp', 'target_position')

        self.model.set_rhs('s.position', s.positionD)
        self.model.set_rhs('s.angle', s.angleD)

        angleD_next, positionD_next = cartpole_ode_namespace(s, Q2u(Q))

        self.model.set_rhs('s.positionD', positionD_next)
        self.model.set_rhs('s.angleD', angleD_next)

        # Simplified, normalized expressions for E_kin and E_pot as a port of cost function
        E_kin_cart = (s.positionD / P_GLOBALS.v_max) ** 2
        E_kin_pol = (s.angleD/(2*np.pi))**2
        E_pot = np.cos(s.angle)

        distance_difference = (((s.position - target_position)/50.0) ** 2)

        self.model.set_expression('E_kin_cart', E_kin_cart)
        self.model.set_expression('E_kin_pol', E_kin_pol)
        self.model.set_expression('E_pot', E_pot)
        self.model.set_expression('distance_difference', distance_difference)


        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': mpc_horizon,
            't_step': dt_mpc_simulation,
            'n_robust': 0,
            'store_full_solution': False,
            'store_lagr_multiplier': False,
            'store_solver_stats': []
        }
        self.mpc.set_param(**setup_mpc)
        # self.mpc.set_param(nlpsol_opts={'ipopt.linear_solver': 'mumps'})
        # Other possible linear solvers from hsl library
        # The give better performance 2-3 times.
        # However if simulating at max speedup the simulation blocks. Issue with memory leak?
        # self.mpc.set_param(nlpsol_opts={'ipopt.linear_solver': 'mumps'})
        self.mpc.set_param(nlpsol_opts = {'ipopt.linear_solver': 'MA57'})

        # # Standard version
        lterm = - self.model.aux['E_pot'] + 20.0 * distance_difference
        mterm = 5 * self.model.aux['E_kin_pol'] - 5 * self.model.aux['E_pot']  + 5 * self.model.aux['E_kin_cart']
        self.mpc.set_rterm(Q=0.1)

        # # Alternative versions of cost function to get more diverse data for learning cartpole model
        # lterm = 20.0 * distance_difference
        # mterm = 5 * self.model.aux['E_kin_pol'] - 5 * self.model.aux['E_pot']  + 5 * self.model.aux['E_kin_cart']
        # self.mpc.set_rterm(Q=0.2)
        #
        # lterm = 20.0 * distance_difference + 5 * self.model.aux['E_kin_cart']
        # mterm = 5 * self.model.aux['E_kin_pol'] - 5 * self.model.aux['E_pot'] + 200.0 * distance_difference
        # self.mpc.set_rterm(Q=0.2)


        self.mpc.set_objective(mterm=mterm, lterm=lterm)


        self.mpc.bounds['lower', '_u', 'Q'] = -1.0
        self.mpc.bounds['upper', '_u', 'Q'] = 1.0

        self.tvp_template = self.mpc.get_tvp_template()

        self.mpc.set_tvp_fun(self.tvp_fun)

        # Suppress IPOPT outputs (optimizer info printed to the console)
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


    def step(self, s, target_position, time=None):

        s = cartpole_state_vector_to_namespace(s)

        self.x0['s.position'] = s.position
        self.x0['s.positionD'] = s.positionD

        self.x0['s.angle'] = s.angle
        self.x0['s.angleD'] = s.angleD

        self.tvp_template['_tvp', :, 'target_position'] = target_position

        Q = self.mpc.make_step(self.x0)

        return Q.item()