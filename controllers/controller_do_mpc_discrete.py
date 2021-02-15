"""do-mpc controller"""

import do_mpc
import numpy as np

from types import SimpleNamespace

from CartPole.cartpole_model import Q2u, p_globals
from src.utilis import mpc_next_state

dt_mpc_simulation = 0.2  # s
mpc_horizon = 10


class controller_do_mpc_discrete:
    def __init__(self,
                 position_init=0.0,
                 positionD_init=0.0,
                 angle_init=0.0,
                 angleD_init=0.0,
                 ):

        """
        Get configured do-mpc modules:
        """

        # Physical parameters of the cart
        p = p_globals

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

        s_next = mpc_next_state(s, p, Q2u(Q,p), dt=dt_mpc_simulation)

        self.model.set_rhs('s.position', s_next.position)
        self.model.set_rhs('s.angle', s_next.angle)

        self.model.set_rhs('s.positionD',s_next.positionD)
        self.model.set_rhs('s.angleD', s_next.angleD)

        # Simplified, normalized expressions for E_kin and E_pot as a port of cost function
        E_kin_cart = (s.positionD / p.v_max) ** 2
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
            'store_solver_stats': [],
            'state_discretization': 'discrete'
        }
        self.mpc.set_param(**setup_mpc)
        self.mpc.set_param(nlpsol_opts = {'ipopt.linear_solver': 'ma27'})

        # Works with horizon 10
        lterm = - self.model.aux['E_pot'] +\
                20 * distance_difference +\
                5 * self.model.aux['E_kin_pol']
        mterm = 5 * self.model.aux['E_kin_pol'] - 5 * self.model.aux['E_pot']  + 5 * self.model.aux['E_kin_cart']
        self.mpc.set_rterm(Q=0.1)

        # Horizon 5
        # weights
        # wr = 0.1  # rterm
        # l1 = 0.0  # -pot
        # l2 = 50.0  # distance
        # l3 = 0.0  # kin_pol
        # m1 = 0.0  # kin_pol
        # m2 = 50.0  # -pot
        # m3 = 0.0  # kin_cart
        # m4 = 0.0# 20.0*10.0  # distance
        #
        # w_sum = wr + l1 + l2 + l3 + m1 + m2 + m3
        #
        # wr /= w_sum
        # l1 /= w_sum
        # l2 /= w_sum
        # l3 /= w_sum
        # m1 /= w_sum
        # m2 /= w_sum
        # m3 /= w_sum
        # m4 /= w_sum

        # print(distance_difference)
        # lterm = -l1 * self.model.aux['E_pot'] +\
        #         l2 * distance_difference +\
        #         l3 * self.model.aux['E_kin_pol']
        # mterm = m1 * self.model.aux['E_kin_pol']\
        #         - m2 * self.model.aux['E_pot']\
        #         + m3 * self.model.aux['E_kin_cart']\
        #         + m4 * distance_difference
        #
        # self.mpc.set_objective(mterm=mterm, lterm=lterm)
        # self.mpc.set_rterm(Q=wr)

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

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


    def step(self, s, target_position):

        self.x0['s.position'] = s.position
        self.x0['s.positionD'] = s.positionD

        self.x0['s.angle'] = s.angle
        self.x0['s.angleD'] = s.angleD

        self.tvp_template['_tvp', :, 'target_position'] = target_position

        Q = self.mpc.make_step(self.x0)

        return Q.item()