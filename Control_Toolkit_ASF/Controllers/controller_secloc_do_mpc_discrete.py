from types import SimpleNamespace
import do_mpc
import numpy as np
import casadi as ca

from CartPole.cartpole_equations import _cartpole_ode, Q2u
from CartPole.state_utilities import cartpole_state_vector_to_namespace
from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.computation_library import NumpyLibrary, TensorType
from CartPole.cartpole_parameters import TrackHalfLength, v_max
from CartPole.cartpole_parameters import k, m_cart, m_pole, g, J_fric, M_fric, L, u_max
from others.globals_and_utils import load_config

def cartpole_ode_namespace(s: SimpleNamespace, u: float):
    cos_angle = ca.cos(s.angle) if isinstance(s.angle, ca.SX) else np.cos(s.angle)
    sin_angle = ca.sin(s.angle) if isinstance(s.angle, ca.SX) else np.sin(s.angle)

    angleDD, positionDD = _cartpole_ode(
        cos_angle, sin_angle, s.angleD, s.positionD, u,
        k=k, m_cart=m_cart, m_pole=m_pole, g=g, J_fric=J_fric, M_fric=M_fric, L=L
    )
    return angleDD, positionDD

def mpc_next_state(s, u, dt):
    s_next = SimpleNamespace()

    s_next.position = s.position
    s_next.positionD = s.positionD
    s_next.angle = s.angle
    s_next.angleD = s.angleD

    s_next.angleDD, s_next.positionDD = cartpole_ode_namespace(s_next, u)

    s_next = cartpole_integration(s_next, dt)

    return s_next

def cartpole_integration(s, dt):
    s_next = SimpleNamespace()

    s_next.position = s.position + s.positionD * dt
    s_next.positionD = s.positionD + s.positionDD * dt

    s_next.angle = s.angle + s.angleD * dt
    s_next.angleD = s.angleD + s.angleDD * dt

    return s_next


class controller_secloc_do_mpc_discrete(template_controller):
    _computation_library = NumpyLibrary()

    def configure(self):
        s = SimpleNamespace()

        model_type = 'discrete'
        self.model = do_mpc.model.Model(model_type)

        s.position = self.model.set_variable(var_type='_x', var_name='s.position', shape=(1, 1))
        s.positionD = self.model.set_variable(var_type='_x', var_name='s.positionD', shape=(1, 1))
        s.angle = self.model.set_variable(var_type='_x', var_name='s.angle', shape=(1, 1))
        s.angleD = self.model.set_variable(var_type='_x', var_name='s.angleD', shape=(1, 1))

        Q = self.model.set_variable(var_type='_u', var_name='Q')

        target_position = self.model.set_variable('_tvp', 'target_position')
        dt = self.model.set_variable('_tvp', 'dt')

        s_next = mpc_next_state(s, Q2u(Q, u_max), dt=dt)

        self.model.set_rhs('s.position', s_next.position)
        self.model.set_rhs('s.angle', s_next.angle)
        self.model.set_rhs('s.positionD', s_next.positionD)
        self.model.set_rhs('s.angleD', s_next.angleD)

        E_kin_cart = (s.positionD / v_max) ** 2
        E_kin_pol = (s.angleD / (2 * np.pi)) ** 2
        E_pot = ca.cos(s.angle)

        distance_difference = (((s.position - target_position) / TrackHalfLength) ** 2)

        self.model.set_expression('E_kin_cart', E_kin_cart)
        self.model.set_expression('E_kin_pol', E_kin_pol)
        self.model.set_expression('E_pot', E_pot)
        self.model.set_expression('distance_difference', distance_difference)

        self.model.setup()

        configT = load_config("config_gui.yml")
        self.dt = configT["time_scales"]["controller_update_interval"]

        self.mpc = do_mpc.controller.MPC(self.model)
        self.ref_period = self.config_controller["ref_period"]
        self.log_base = self.config_controller["log_base"]
        self.dead_ang = self.config_controller["dead_ang"]
        self.dead_pos = self.config_controller["dead_pos"]
        self.ang_last_shift = 0.0001
        self.pos_last_shift = 0.0001
        self.last_Q = 0
        self.time_last = None
        self.tick_control = 0
        self.tick_total = 0
        self.tick_last = 0

        self.pos_events = 0
        self.ang_events = 0
        self.total_events = 0

        setup_mpc = {
            'n_horizon': self.config_controller["mpc_horizon"],
            't_step': self.config_controller["ref_period"],
            'n_robust': 0,
            'store_full_solution': False,
            'store_lagr_multiplier': False,
            'store_solver_stats': [],
            'state_discretization': 'discrete'
        }
        self.mpc.set_param(**setup_mpc)

        lterm = - 25 * self.model.aux['E_pot'] + \
                1 * distance_difference + \
                5 * self.model.aux['E_kin_pol']

        mterm = (5 * self.model.aux['E_kin_pol'] - 25 * self.model.aux['E_pot'] + 5 * self.model.aux['E_kin_cart'])

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(Q=0.1)

        self.mpc.bounds['lower', '_u', 'Q'] = self.action_low
        self.mpc.bounds['upper', '_u', 'Q'] = self.action_high

        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_fun)

        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.mpc.set_param(nlpsol_opts=suppress_ipopt)

        self.mpc.setup()

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
        if self.time_last is None:
            time_difference = self.ref_period
        else:
            time_difference = time - self.time_last

        self.tick_last = self.tick_last + 1
        self.tick_total = self.tick_total + 1

        self.update_attributes(updated_attributes)
        s = cartpole_state_vector_to_namespace(s)

        if self.tick_last >= self.ref_period:
            spike = 0
            ang_shift = s.angle
            ang_shift_sign = 1
            if ang_shift < 0:
                ang_shift_sign = -1
                ang_shift = -ang_shift
            if (ang_shift > self.dead_ang) and (self.ang_last_shift != 0):
                ang_ratio_inc = ang_shift / self.ang_last_shift
                ang_ratio_dec = 1.0 / ang_ratio_inc
                if (ang_ratio_inc >= self.log_base) or (ang_ratio_dec >= self.log_base):
                    self.ang_last_shift = ang_shift
                    spike = 1
                    self.ang_events = self.ang_events + 1
            if spike == 0:
                pos_shift = s.position - self.variable_parameters.target_position
                pos_shift_sign = 1
                if pos_shift < 0:
                    pos_shift_sign = -1
                    pos_shift = -pos_shift
                if (pos_shift > self.dead_pos) and (self.pos_last_shift != 0):
                    pos_ratio_inc = pos_shift / self.pos_last_shift
                    pos_ratio_dec = 1.0 / pos_ratio_inc
                    if (pos_ratio_inc >= self.log_base) or (pos_ratio_dec >= self.log_base):
                        self.pos_last_shift = pos_shift
                        spike = 1
                        self.pos_events = self.pos_events + 1
            if spike == 1:
                self.tick_last = 0
                self.tick_control = self.tick_control + 1
                self.time_last = time
                self.x0['s.position'] = s.position
                self.x0['s.positionD'] = s.positionD
                self.x0['s.angle'] = s.angle
                self.x0['s.angleD'] = s.angleD

                if time_difference > 10 * self.ref_period * self.dt:
                    time_difference = 10 * self.ref_period * self.dt
                self.tvp_template['_tvp', :, 'target_position'] = self.variable_parameters.target_position

                dt_nominal = self.ref_period * self.dt

                for k in range(0, self.config_controller["mpc_horizon"] + 1):
                    self.tvp_template['_tvp', k, 'dt'] = dt_nominal

                Q = self.mpc.make_step(self.x0)
                self.last_Q = Q.item()
                self.total_events = self.total_events + 1
                return Q.item()
        return self.last_Q
