"""do-mpc controller"""

from types import SimpleNamespace

import do_mpc
import numpy as np

from CartPole.cartpole_equations import _cartpole_ode, Q2u
from CartPole.state_utilities import cartpole_state_vector_to_namespace
from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.computation_library import NumpyLibrary, TensorType
from CartPole.cartpole_parameters import TrackHalfLength, v_max
from CartPole.cartpole_parameters import k, m_cart, m_pole, g, J_fric, M_fric, L, u_max

def cartpole_ode_namespace(s: SimpleNamespace, u: float):
    angleDD, positionDD = _cartpole_ode(
        np.cos(s.angle), np.sin(s.angle), s.angleD, s.positionD, u,
        k=k, m_cart=m_cart, m_pole=m_pole, g=g, J_fric=J_fric, M_fric=M_fric, L=L
    )
    return angleDD, positionDD

def mpc_next_state(s, u, dt):
    """Wrapper for CartPole ODE. Given a current state (without second derivatives), returns a state after time dt

    TODO: This might be combined with cartpole_integration,
        although the order of cartpole_ode and cartpole_integration is different than in CartClass
        For some reaseon it does not work at least not with do-mpc discreate
    """

    s_next = s

    s_next.angleDD, s_next.positionDD = cartpole_ode_namespace(s_next, u)  # Calculates CURRENT second derivatives

    # Calculate NEXT state:
    s_next = cartpole_integration(s_next, dt)

    return s_next



def cartpole_integration(s, dt):
    """
    Simple single step integration of CartPole state by dt

    Takes state as numpy array.

    :param s: state of the CartPole (position, positionD, angle, angleD must be set). Array order follows global definition.
    :param dt: time step by which the CartPole state should be integrated
    """
    s_next = SimpleNamespace()

    s_next.position = s.position + s.positionD * dt
    s_next.positionD = s.positionD + s.positionDD * dt

    s_next.angle = s.angle + s.angleD * dt
    s_next.angleD = s.angleD + s.angleDD * dt

    return s_next


class controller_do_mpc_discrete(template_controller):
    _computation_library = NumpyLibrary
    
    def configure(self):
        """
        Get configured do-mpc modules:
        """
        # Container for the state of the cart
        s = SimpleNamespace()

        model_type = 'discrete'  # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)

        s.position = self.model.set_variable(var_type='_x', var_name='s.position', shape=(1, 1))
        s.positionD = self.model.set_variable(var_type='_x', var_name='s.positionD', shape=(1, 1))

        s.angle = self.model.set_variable(var_type='_x', var_name='s.angle', shape=(1, 1))
        s.angleD = self.model.set_variable(var_type='_x', var_name='s.angleD', shape=(1, 1))

        Q = self.model.set_variable(var_type='_u', var_name='Q')

        target_position = self.model.set_variable('_tvp', 'target_position')

        s_next = mpc_next_state(s, Q2u(Q, u_max), dt=self.config_controller["dt"])

        self.model.set_rhs('s.position', s_next.position)
        self.model.set_rhs('s.angle', s_next.angle)

        self.model.set_rhs('s.positionD',s_next.positionD)
        self.model.set_rhs('s.angleD', s_next.angleD)

        # Simplified, normalized expressions for E_kin and E_pot as a part of cost function
        E_kin_cart = (s.positionD / v_max) ** 2
        E_kin_pol = (s.angleD/(2*np.pi))**2
        E_pot = np.cos(s.angle)

        distance_difference = (((s.position - target_position)/TrackHalfLength) ** 2)

        self.model.set_expression('E_kin_cart', E_kin_cart)
        self.model.set_expression('E_kin_pol', E_kin_pol)
        self.model.set_expression('E_pot', E_pot)
        self.model.set_expression('distance_difference', distance_difference)

        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': self.config_controller["mpc_horizon"],
            't_step': self.config_controller["dt"],
            'n_robust': 0,
            'store_full_solution': False,
            'store_lagr_multiplier': False,
            'store_solver_stats': [],
            'state_discretization': 'discrete'
        }
        self.mpc.set_param(**setup_mpc)
        self.mpc.set_param(nlpsol_opts = {'ipopt.linear_solver': 'ma27'})

        lterm = - 25 * self.model.aux['E_pot'] +\
                1 * distance_difference +\
                5 * self.model.aux['E_kin_pol']

        mterm = (5 * self.model.aux['E_kin_pol'] - 25 * self.model.aux['E_pot'] + 5 * self.model.aux['E_kin_cart'])

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(Q=0.1)

        self.mpc.bounds['lower', '_u', 'Q'] = self.action_low
        self.mpc.bounds['upper', '_u', 'Q'] = self.action_high

        self.tvp_template = self.mpc.get_tvp_template()

        self.mpc.set_tvp_fun(self.tvp_fun)

        # Suppress IPOPT outputs
        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.mpc.set_param(nlpsol_opts=suppress_ipopt)

        self.mpc.setup()

        # Set initial state
        self.x0 = self.mpc.x0
        self.x0['s.position'] = self.config_controller["position_init"]
        self.x0['s.positionD'] = self.config_controller["positionD_init"]
        self.x0['s.angle'] = self.config_controller["angle_init"]
        self.x0['s.angleD'] = self.config_controller["angleD_init"]

        self.mpc.x0 = self.x0

        self.mpc.set_initial_guess()

    def tvp_fun(self, t_ind):
        return self.tvp_template

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.update_attributes(updated_attributes)

        s = cartpole_state_vector_to_namespace(s)

        self.x0['s.position'] = s.position
        self.x0['s.positionD'] = s.positionD

        self.x0['s.angle'] = s.angle
        self.x0['s.angleD'] = s.angleD

        self.tvp_template['_tvp', :, 'target_position'] = self.variable_parameters.target_position

        Q = self.mpc.make_step(self.x0)

        return Q.item()
